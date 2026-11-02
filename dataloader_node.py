import copy
import math
from pathlib import Path

import torch
import numpy as np

from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import add_self_loops, remove_self_loops, \
    to_undirected, to_scipy_sparse_matrix
from torch_geometric.data import Data

from graph_infiltrator_core import apply_graph_infiltrator_poison, prepare_graph_infiltrator_data

INF = np.iinfo(np.int64).max


def _keep_largest_connected_component(data):
    """Match the paper's 7,487-node Amazon-Photo graph preprocessing."""
    from scipy.sparse.csgraph import connected_components

    adjacency = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    component_count, labels = connected_components(adjacency, directed=False)
    if component_count <= 1:
        return data
    counts = np.bincount(labels)
    largest = int(counts.argmax())
    subset = torch.from_numpy(np.flatnonzero(labels == largest)).long()
    result = data.subgraph(subset)
    result['source_num_components'] = int(component_count)
    result['largest_component_applied'] = True
    return result


def _sample_undirected_edge_columns(edge_index, ratio, seed, num_nodes):
    """Sample relation pairs, retaining all stored directions of each relation."""
    if edge_index.numel() == 0 or ratio <= 0:
        return edge_index[:, :0].clone(), {
            'undirected_total': 0,
            'undirected_kept': 0,
            'directed_columns_kept': 0,
        }

    edge_cpu = edge_index.detach().cpu().long()
    low = torch.minimum(edge_cpu[0], edge_cpu[1])
    high = torch.maximum(edge_cpu[0], edge_cpu[1])
    relation_keys = low * int(num_nodes) + high
    unique_keys, inverse = torch.unique(relation_keys, sorted=True, return_inverse=True)
    keep_count = min(
        unique_keys.numel(),
        max(1, math.floor(float(ratio) * unique_keys.numel())),
    )
    generator = torch.Generator().manual_seed(int(seed))
    selected_groups = torch.randperm(unique_keys.numel(), generator=generator)[:keep_count]
    keep_mask = torch.isin(inverse, selected_groups)
    sampled = edge_cpu[:, keep_mask].contiguous()
    stats = {
        'undirected_total': int(unique_keys.numel()),
        'undirected_kept': int(keep_count),
        'directed_columns_kept': int(sampled.shape[1]),
    }
    return sampled.to(edge_index.device), stats


def _as_numpy_index(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64)


def _feature_importance_order(dataset, seed, task='node'):
    train_mask = getattr(dataset, 'train_mask', None)
    if train_mask is None:
        raise ValueError('[Split] feature_importance requires an explicit train_mask.')
    train_mask = train_mask.detach().cpu().bool().reshape(-1)
    if train_mask.numel() != dataset.x.shape[0] or not bool(train_mask.any()):
        raise ValueError('[Split] feature_importance requires a non-empty train_mask aligned with dataset.x.')

    # MDI is an offline experimental partitioning aid. It must not inspect
    # validation/test labels, which are unavailable to the attacker.
    x = dataset.x.detach().cpu()[train_mask].numpy()
    y = dataset.y.detach().cpu().reshape(-1)[train_mask].numpy()
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError:
        print('[Split] scikit-learn is unavailable; falling back to random feature ordering.')
        return torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(int(seed)))

    unique_y = np.unique(y)
    use_regressor = str(task).lower() in {'reg', 'regression', 'regress'}
    if not use_regressor and np.issubdtype(y.dtype, np.floating) and unique_y.size > max(20, x.shape[0] // 2):
        use_regressor = True

    estimator = RandomForestRegressor(
        n_estimators=100,
        random_state=int(seed),
        n_jobs=-1,
    ) if use_regressor else RandomForestClassifier(
        n_estimators=100,
        random_state=int(seed),
        n_jobs=-1,
    )
    try:
        estimator.fit(x, y)
        importances = estimator.feature_importances_
        order = np.argsort(-importances, kind='mergesort')
        return torch.from_numpy(order.astype(np.int64, copy=False))
    except Exception as exc:
        print(f'[Split] Feature-importance training failed ({exc}); falling back to random ordering.')
        return torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(int(seed)))


def _strict_feature_importance_partitions(feature_order, feature_dim, seed):
    """Build complementary 70/30 and 30/70 feature partitions."""
    order = _as_numpy_index(feature_order)
    if np.unique(order).size != int(feature_dim) or order.size != int(feature_dim):
        raise ValueError('[Split] feature importance order must contain every feature exactly once.')

    half = int(feature_dim) // 2
    high_pool = order[:half].copy()
    normal_pool = order[half:].copy()
    first_count = half
    rng = np.random.default_rng(int(seed))
    rng.shuffle(high_pool)
    rng.shuffle(normal_pool)

    first_high_count = int(round(first_count * 0.7))
    first_high_count = max(0, min(first_high_count, high_pool.size, first_count))
    first_normal_count = first_count - first_high_count
    if first_normal_count > normal_pool.size:
        shortfall = first_normal_count - normal_pool.size
        first_high_count += shortfall
        first_normal_count -= shortfall

    party_1 = np.concatenate([
        high_pool[:first_high_count],
        normal_pool[:first_normal_count],
    ])
    # Party 2 receives the exact complement. This guarantees both disjointness
    # and complete coverage of the global feature set.
    party_2 = np.concatenate([
        high_pool[first_high_count:],
        normal_pool[first_normal_count:],
    ])
    rng.shuffle(party_1)
    rng.shuffle(party_2)

    stats = {
        'global_features': int(feature_dim),
        'important_features': int(high_pool.size),
        'general_features': int(normal_pool.size),
        'party_1_features': int(party_1.size),
        'party_1_important': int(first_high_count),
        'party_2_features': int(party_2.size),
        'party_2_important': int(high_pool.size - first_high_count),
        'overlap_features': 0,
        'uncovered_features': 0,
    }
    return {
        1: torch.from_numpy(party_1.astype(np.int64, copy=False)),
        2: torch.from_numpy(party_2.astype(np.int64, copy=False)),
    }, stats


def _strict_degree_edge_partitions(edge_index, num_nodes, seed):
    """Split original relation categories 80/20 and 20/80 without overlap.

    Party 1 receives 80% of the original frequent-relation pool and 20% of the
    original long-tail pool. Party 2 receives the exact complements. This is a
    category-allocation ratio, not a within-local-graph composition ratio.
    """
    if edge_index.numel() == 0 or int(num_nodes) == 0:
        empty = torch.empty((0,), dtype=torch.long)
        stats = {
            'global_relations': 0,
            'frequent_relations': 0,
            'long_tail_relations': 0,
            'party_1_relations': 0,
            'party_2_relations': 0,
            'overlap_relations': 0,
            'uncovered_relations': 0,
            'uncovered_ratio': 0.0,
        }
        return {1: empty, 2: empty.clone()}, stats

    edge_cpu = edge_index.detach().cpu().long()
    low = torch.minimum(edge_cpu[0], edge_cpu[1])
    high = torch.maximum(edge_cpu[0], edge_cpu[1])
    relation_keys = low * int(num_nodes) + high
    unique_keys, inverse = torch.unique(relation_keys, sorted=True, return_inverse=True)

    degree = torch.bincount(edge_cpu.reshape(-1), minlength=int(num_nodes)).numpy()
    node_order = np.lexsort((np.arange(int(num_nodes)), -degree))
    active_nodes = np.zeros(int(num_nodes), dtype=bool)
    active_nodes[node_order[:max(1, int(num_nodes) // 2)]] = True
    relation_src = (unique_keys // int(num_nodes)).numpy()
    relation_dst = (unique_keys % int(num_nodes)).numpy()
    frequent_mask = active_nodes[relation_src] | active_nodes[relation_dst]
    frequent_groups = np.nonzero(frequent_mask)[0].astype(np.int64, copy=False)
    tail_groups = np.nonzero(~frequent_mask)[0].astype(np.int64, copy=False)
    if frequent_groups.size == 0 or tail_groups.size == 0:
        raise ValueError('[Split] degree-biased partition requires both frequent and long-tail relations.')

    party_1_frequent = int(round(frequent_groups.size * 0.8))
    party_1_tail = int(round(tail_groups.size * 0.2))
    party_2_frequent = int(frequent_groups.size - party_1_frequent)
    party_2_tail = int(tail_groups.size - party_1_tail)

    rng = np.random.default_rng(int(seed) + 17)
    rng.shuffle(frequent_groups)
    rng.shuffle(tail_groups)
    party_1_groups = np.concatenate([
        frequent_groups[:party_1_frequent],
        tail_groups[:party_1_tail],
    ])
    party_2_groups = np.concatenate([
        frequent_groups[party_1_frequent:],
        tail_groups[party_1_tail:],
    ])
    rng.shuffle(party_1_groups)
    rng.shuffle(party_2_groups)

    inverse_np = inverse.numpy()
    party_1_columns = np.nonzero(np.isin(inverse_np, party_1_groups))[0]
    party_2_columns = np.nonzero(np.isin(inverse_np, party_2_groups))[0]
    global_relations = int(unique_keys.numel())
    party_1_relations = int(party_1_groups.size)
    party_2_relations = int(party_2_groups.size)
    uncovered_relations = global_relations - party_1_relations - party_2_relations
    stats = {
        'global_relations': global_relations,
        'frequent_relations': int(frequent_groups.size),
        'long_tail_relations': int(tail_groups.size),
        'party_1_relations': party_1_relations,
        'party_1_frequent': party_1_frequent,
        'party_1_long_tail': party_1_tail,
        'party_2_relations': party_2_relations,
        'party_2_frequent': party_2_frequent,
        'party_2_long_tail': party_2_tail,
        'overlap_relations': 0,
        'uncovered_relations': int(uncovered_relations),
        'uncovered_ratio': float(uncovered_relations / global_relations),
        'party_1_directed_columns': int(party_1_columns.size),
        'party_2_directed_columns': int(party_2_columns.size),
    }
    return {
        1: torch.from_numpy(party_1_columns.astype(np.int64, copy=False)),
        2: torch.from_numpy(party_2_columns.astype(np.int64, copy=False)),
    }, stats

# 数据集中本身没有self_loop
def load_nodelevel_dataset(config=None):
    r"""
    :returns:
        data_dict
    :rtype:
        Dict: dict{'client_id': Data()}
    """
    path = str(Path(__file__).resolve().parent / 'data')
    name = config.dataset
    name_aliases = {"amazon-photo": "photo", "amazon_photo": "photo"}
    name = name_aliases.get(name, name)

    # Transforms 默认不使用transform_func
    # Dataset
    if name in ["cora", "citeseer", "pubmed"]:
        num_split = {
            'cora': [170, 0, INF], #232
            'citeseer': [220, 0, INF], #332
            'pubmed': [3000, 0, INF], #3943
        }
        dataset = Planetoid(path,
                            name,
                            split='random',
                            num_train_per_class=num_split[name][0],
                            num_val=num_split[name][1],
                            num_test=num_split[name][2])
        dataset = dataset[0]
    elif name in ["photo", "amazon-photo"]:
        dataset = Amazon(path, name='Photo')
        dataset = dataset[0]
        if bool(getattr(config, 'photo_largest_connected_component', True)):
            dataset = _keep_largest_connected_component(dataset)
        # Amazon-Photo does not provide predefined splits; create train/test masks
        num_nodes = dataset.num_nodes
        num_classes = int(dataset.y.max().item()) + 1
        num_train_per_class = 200  # Target: 200 training nodes per class (approx. 1600 total for 8 classes if balanced)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for c in range(num_classes):
            class_idx = (dataset.y == c).nonzero(as_tuple=False).view(-1)
            perm = torch.randperm(class_idx.size(0))[:num_train_per_class]
            train_mask[class_idx[perm]] = True
        test_mask[~train_mask] = True
        dataset.train_mask = train_mask
        dataset.test_mask = test_mask
    else:
        raise ValueError(
            f'Unsupported dataset: {name}. Supported datasets: cora, citeseer, pubmed, '
            f'photo (amazon-photo)'
        )

    #将dataset中的边随机打乱顺序排列，方便进行边的切割
    feature_generator = torch.Generator().manual_seed(int(config.seed))
    edge_generator = torch.Generator().manual_seed(int(config.seed) + 1)
    dataset.edge_index = dataset.edge_index[:, torch.randperm(dataset.edge_index.shape[1], generator=edge_generator)]
    # 将dataset中的结点特征随机打乱顺序排列，方便进行结点特征的切割
    dataset.x = dataset.x[:, torch.randperm(dataset.x.shape[1], generator=feature_generator)]

    # 分割节点特征。主实验对属性做互斥、均匀的垂直划分。
    data = {}
    feature_partition = [
        (i + 1) / config.num_of_clients for i in range(config.num_of_clients - 1)
    ]
    edge_partition = [
        (i + 1) / config.num_of_clients for i in range(config.num_of_clients - 1)
    ]
    print(feature_partition, edge_partition)
    feature_dim = dataset.x.shape[1]
    feature_indexs = torch.randperm(feature_dim, generator=feature_generator)
    edge_dim = dataset.edge_index.shape[1]
    if config.no_edge == 1:
        edge_dim = 0

    # 主实验：每个参与方独立采样指定比例的无向关系。不能复用同一份列索引，
    # 也不能把 PyG 中一条无向关系的正反向列拆开。
    edge_ratio = getattr(config, "edge_ratio", None)
    edge_split_strategy = getattr(config, 'edge_split_strategy', 'random')
    if edge_ratio is not None and edge_split_strategy == 'random':
        edge_ratio = max(0.0, min(1.0, float(edge_ratio)))
        independently_sampled_edges = {}
        edge_sampling_stats = {}
        for client_id in range(1, config.num_of_clients + 1):
            sampled, stats = _sample_undirected_edge_columns(
                dataset.edge_index,
                edge_ratio,
                int(config.seed) + 10007 * client_id,
                dataset.num_nodes,
            )
            independently_sampled_edges[client_id] = sampled
            edge_sampling_stats[client_id] = stats
        print(
            f"[Main Edge Split] directed_columns={edge_dim}, edge_ratio={edge_ratio}, "
            f"per_party={[v['directed_columns_kept'] for v in edge_sampling_stats.values()]}"
        )
    else:
        independently_sampled_edges = {}
        edge_sampling_stats = {}

    feature_split_strategy = getattr(config, 'feature_split_strategy', 'random')

    feature_partitions = {}
    edge_partitions = {}
    feature_partition_stats = {}
    edge_partition_stats = {}

    if config.num_of_clients == 2 and feature_split_strategy == 'feature_importance':
        feature_order = _feature_importance_order(dataset, config.seed, getattr(config, 'task', 'node'))
        feature_partitions, feature_partition_stats = _strict_feature_importance_partitions(
            feature_order, feature_dim, config.seed
        )
        print(f'[Split] Using disjoint feature-importance partition: {feature_partition_stats}.')
    elif config.num_of_clients == 2:
        feature_indexs = torch.randperm(feature_dim, generator=feature_generator)

    if config.num_of_clients == 2 and edge_split_strategy == 'biased_degree':
        edge_partitions, edge_partition_stats = _strict_degree_edge_partitions(
            dataset.edge_index[:, :edge_dim], dataset.num_nodes, config.seed
        )
        print(f'[Split] Using disjoint degree-biased edge partition: {edge_partition_stats}.')

    # i=0是总数据，其他是各个客户端的数据
    for i in range(config.num_of_clients + 1):
        data[i] = copy.deepcopy(dataset)

    for i in range(1, config.num_of_clients + 1):
        if i in feature_partitions:
            selected_feature_indices = feature_partitions[i]
            data[i].x = data[i].x[:, selected_feature_indices]
        elif feature_partition:
            assert len(feature_partition) == config.num_of_clients - 1, "feature_partition的长度应该为num_of_clients-1"
            if i == 1:
                selected_feature_indices = feature_indexs[:math.floor(feature_dim*feature_partition[0])]
            elif i == config.num_of_clients:
                selected_feature_indices = feature_indexs[math.floor(feature_dim*feature_partition[i-2]):feature_dim]
            else:
                selected_feature_indices = feature_indexs[
                    math.floor(feature_dim * feature_partition[i - 2]):
                    math.floor(feature_dim * feature_partition[i - 1])
                ]
            data[i].x = data[i].x[:, selected_feature_indices]
        else:
            selected_feature_indices = torch.arange(feature_dim)

        data[i]['feature_indices'] = selected_feature_indices.clone()
        if feature_partition_stats:
            data[i]['feature_partition_stats'] = copy.deepcopy(feature_partition_stats)
        if i in independently_sampled_edges:
            data[i].edge_index = independently_sampled_edges[i]
            data[i]['edge_sampling_stats'] = edge_sampling_stats[i]
        elif i in edge_partitions:
            data[i].edge_index = data[i].edge_index[:, edge_partitions[i]]
            data[i]['edge_partition_stats'] = copy.deepcopy(edge_partition_stats)
        elif edge_partition:
            if config.num_of_clients == 2:
                assert len(edge_partition) == config.num_of_clients - 1, "edge_partition的长度应该为num_of_clients-1"
                if i == 1:
                    data[i].edge_index = data[i].edge_index[:, :math.floor(edge_dim * edge_partition[0])]
                elif i == config.num_of_clients:
                    data[i].edge_index = data[i].edge_index[:, math.floor(edge_dim * edge_partition[i - 2]):edge_dim]
                else:
                    data[i].edge_index = data[i].edge_index[:, math.floor(edge_dim * edge_partition[i - 2]):
                                             math.floor(edge_dim * edge_partition[i - 1])]
            else:
                # 从data[i].edge_index中随机选择一半边，将这些边放到data[i].edge_index中
                data[i].edge_index = data[i].edge_index[:, torch.randperm(data[i].edge_index.shape[1])[:math.floor(edge_dim/2)]]
    return data


def setup_data(data, config):
    """Prepare Graph-Infiltrator anchors and evaluation nodes."""
    return prepare_graph_infiltrator_data(data, config)


def poison_data(data, config):
    """Build the persistent Graph-Infiltrator training view."""
    return apply_graph_infiltrator_poison(data, config)
