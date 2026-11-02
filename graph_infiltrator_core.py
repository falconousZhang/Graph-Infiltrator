"""Core data operations for the integrated Graph-Infiltrator attack.

The label tensor is used only to sample the explicitly known anchors, build the
ASR evaluation set, and evaluate LIA after inference. Candidate selection itself
depends only on the malicious participant's embeddings and the anchor IDs.
"""

import math
import warnings

import torch

from utils import distribute_quota, merge_multi_target_nodes, resolve_target_labels


def _cpu_long(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long()
    return torch.as_tensor(values, dtype=torch.long)


def _mapping_value(mapping, label):
    value = mapping.get(label)
    if value is None:
        value = mapping.get(str(label))
    return value


def _sample_without_replacement(pool, count, generator):
    pool = _cpu_long(pool)
    count = int(count)
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if count > pool.numel():
        raise ValueError(f'Cannot sample {count} distinct items from a pool of {pool.numel()}.')
    order = torch.randperm(pool.numel(), generator=generator)[:count]
    return pool[order]


def _default_anchor_count(config):
    requested = int(getattr(config, 'num_of_start_node', 0) or 0)
    if requested > 0:
        return requested
    dataset = str(getattr(config, 'dataset', '')).lower()
    defaults = {'cora': 3, 'citeseer': 5, 'pubmed': 5, 'photo': 5, 'amazon-photo': 5}
    fallback = defaults.get(dataset, 5)
    warnings.warn(
        f'num_of_start_node was not set; using K={fallback} for dataset={dataset!r}.',
        RuntimeWarning,
    )
    return fallback


def _build_trigger_attributes(feature_dims, target_labels, trigger_count, seed):
    if trigger_count <= 0:
        return {label: torch.empty(0, dtype=torch.long) for label in target_labels}
    if trigger_count > feature_dims:
        raise ValueError(
            f'Trigger requests {trigger_count} attributes, but the malicious party has only {feature_dims}.'
        )

    generator = torch.Generator().manual_seed(int(seed) + 2003)
    feature_order = torch.randperm(feature_dims, generator=generator)
    attributes = {}
    stride = max(1, trigger_count)
    for index, label in enumerate(target_labels):
        # Different targets receive different trigger patterns. When the feature
        # space is too small for disjoint patterns, rolling keeps them distinct
        # whenever trigger_count < feature_dims while allowing controlled overlap.
        attributes[label] = torch.roll(feature_order, shifts=index * stride)[:trigger_count].clone()
    return attributes


def _select_asr_test_groups(data, config, excluded_nodes=None):
    """Select disjoint ASR groups outside the supplied node set.

    K-anchor inference ranks the training-only candidate set
    ``V_train \\ V_anc``. Once that inference is complete, this helper is called
    again with ``V_inf union V_anc`` so ASR evaluation nodes cannot also be
    training-stage poisoned nodes.
    """

    target_labels = resolve_target_labels(config)
    labels = data.y.detach().cpu().long()
    test_nodes = torch.where(data.test_mask.detach().cpu())[0]
    test_total = math.floor(
        float(config.num_test_poisoning_ratio) * int(test_nodes.numel())
    )
    test_quotas = distribute_quota(test_total, len(target_labels))
    seed = int(getattr(config, 'seed', 0))

    if excluded_nodes is None:
        excluded_nodes = torch.empty(0, dtype=torch.long)
    else:
        excluded_nodes = torch.unique(_cpu_long(excluded_nodes))

    remaining_test_nodes = test_nodes[
        ~torch.isin(test_nodes, excluded_nodes)
    ]
    poisoned_test_nodes_by_target = {}
    for index, label in enumerate(target_labels):
        eligible = remaining_test_nodes[
            labels[remaining_test_nodes] != int(label)
        ]
        count = min(int(test_quotas[index]), int(eligible.numel()))
        generator = torch.Generator().manual_seed(seed + 3001 * (index + 1))
        selected = _sample_without_replacement(eligible, count, generator)
        poisoned_test_nodes_by_target[label] = selected
        if selected.numel() > 0:
            remaining_test_nodes = remaining_test_nodes[
                ~torch.isin(remaining_test_nodes, selected)
            ]

    poisoned_test_nodes = merge_multi_target_nodes(
        poisoned_test_nodes_by_target
    ).cpu()
    overlap = poisoned_test_nodes[
        torch.isin(poisoned_test_nodes, excluded_nodes)
    ]
    if overlap.numel() > 0:
        raise RuntimeError(
            'ASR test selection overlaps V_inf union V_anc: '
            f'{overlap.tolist()}'
        )

    # A test node used by the training-stage attack is not a clean test node,
    # even when it is no longer part of the ASR evaluation set.
    excluded_from_clean_test = torch.unique(
        torch.cat([excluded_nodes, poisoned_test_nodes])
    )
    retained_test_nodes = test_nodes[
        ~torch.isin(test_nodes, excluded_from_clean_test)
    ]

    data['poisoned_test_nodes_by_target'] = poisoned_test_nodes_by_target
    data['poisoned_test_nodes'] = poisoned_test_nodes
    data['retained_test_nodes'] = retained_test_nodes
    data['asr_test_requested'] = test_total
    data['asr_test_selected'] = int(poisoned_test_nodes.numel())
    data['asr_excluded_nodes'] = excluded_nodes
    return poisoned_test_nodes_by_target


def prepare_graph_infiltrator_data(data, config):
    """Prepare anchors, inference budgets, test nodes, and target-specific triggers.

    No training node is selected for poisoning here. The poisoned training set is
    created later by :func:`infer_k_anchor_groups` from learned embeddings.
    """

    target_labels = resolve_target_labels(config)
    if not target_labels:
        raise ValueError('Graph-Infiltrator requires at least one target label.')

    labels = data.y.detach().cpu().long()
    train_nodes = torch.where(data.train_mask.detach().cpu())[0]
    train_size = int(train_nodes.numel())
    seed = int(getattr(config, 'seed', 0))
    anchor_count = _default_anchor_count(config)

    total_inferred = math.floor(float(config.num_train_poisoning_ratio) * train_size)
    inference_quotas = distribute_quota(total_inferred, len(target_labels))

    anchor_nodes_by_target = {}
    for index, label in enumerate(target_labels):
        target_pool = train_nodes[labels[train_nodes] == int(label)]
        if target_pool.numel() < anchor_count:
            raise ValueError(
                f'Target label {label} has only {target_pool.numel()} training nodes; '
                f'cannot sample K={anchor_count} distinct anchors.'
            )
        generator = torch.Generator().manual_seed(seed + 1009 * (index + 1))
        anchor_nodes_by_target[label] = _sample_without_replacement(
            target_pool, anchor_count, generator
        )

    feature_dims = int(data.x.shape[1])
    trigger_ratio = float(config.num_poisoning_feature_ratio)
    trigger_count = math.floor(trigger_ratio * feature_dims)
    if trigger_ratio > 0 and trigger_count == 0:
        trigger_count = 1
    poisoned_attributes_by_target = _build_trigger_attributes(
        feature_dims, target_labels, trigger_count, seed
    )

    empty_groups = {label: torch.empty(0, dtype=torch.long) for label in target_labels}
    data['anchor_nodes_by_target'] = anchor_nodes_by_target
    data['start_node_by_target'] = anchor_nodes_by_target
    data['inference_quota_by_target'] = {
        label: int(inference_quotas[index]) for index, label in enumerate(target_labels)
    }
    data['inferred_train_nodes_by_target'] = dict(empty_groups)
    data['poisoned_train_nodes_by_target'] = dict(empty_groups)
    data['poisoned_attributes_by_target'] = poisoned_attributes_by_target
    data['poisoned_train_nodes'] = torch.empty(0, dtype=torch.long)
    data['inferred_train_nodes'] = torch.empty(0, dtype=torch.long)
    data['number_of_inferred'] = 0
    data['number_of_poison'] = 0
    data['poisoned_attributes'] = poisoned_attributes_by_target[target_labels[0]]
    data['graph_infiltrator_poison_applied'] = False

    # This provisional selection is refreshed after K-anchor inference, when
    # V_inf is known and can be explicitly excluded from the ASR test set.
    _select_asr_test_groups(data, config)

    if len(target_labels) == 1:
        label = target_labels[0]
        data['start_node'] = anchor_nodes_by_target[label]

    return data


def infer_k_anchor_groups(data, embeddings, config):
    """Infer target-label nodes from embeddings using the K-anchor distance sum.

    Following the original experiment implementation, the candidate set is
    V_cand = V_train \\ V_anc. In multi-target runs, a round-robin assignment
    over each target's ranking prevents the same inferred node from being
    assigned to more than one target.
    """

    if embeddings.ndim != 2 or embeddings.shape[0] != data.x.shape[0]:
        raise ValueError('Embeddings must have shape [num_nodes, embedding_dim].')

    target_labels = resolve_target_labels(config)
    anchors_by_target = data.get('anchor_nodes_by_target')
    quotas_by_target = data.get('inference_quota_by_target')
    if not anchors_by_target or quotas_by_target is None:
        raise RuntimeError('Call prepare_graph_infiltrator_data before K-anchor inference.')

    device = embeddings.device
    train_mask = data.train_mask.to(device=device, dtype=torch.bool)
    if train_mask.ndim != 1 or train_mask.numel() != embeddings.shape[0]:
        raise ValueError('train_mask must contain one boolean value per node.')
    candidate_mask = train_mask.clone()
    anchor_tensors = {}
    for label in target_labels:
        anchors = _mapping_value(anchors_by_target, label)
        if anchors is None or len(anchors) == 0:
            raise RuntimeError(f'No anchors are available for target label {label}.')
        anchors = _cpu_long(anchors).to(device)
        anchor_tensors[label] = anchors
        candidate_mask[anchors] = False

    candidate_nodes = torch.where(candidate_mask)[0]
    requested_total = sum(int(_mapping_value(quotas_by_target, label) or 0) for label in target_labels)
    if requested_total > candidate_nodes.numel():
        raise ValueError(
            f'Inference requests {requested_total} distinct nodes, but only '
            f'{candidate_nodes.numel()} non-anchor training candidates are available.'
        )

    rankings = {}
    for label in target_labels:
        anchor_embeddings = embeddings[anchor_tensors[label]]
        scores = torch.cdist(embeddings[candidate_nodes], anchor_embeddings, p=2).sum(dim=1)
        rankings[label] = candidate_nodes[torch.argsort(scores)]

    selected_lists = {label: [] for label in target_labels}
    positions = {label: 0 for label in target_labels}
    used = torch.zeros(embeddings.shape[0], dtype=torch.bool, device=device)
    remaining = requested_total
    while remaining > 0:
        progress = False
        for label in target_labels:
            quota = int(_mapping_value(quotas_by_target, label) or 0)
            if len(selected_lists[label]) >= quota:
                continue
            ranking = rankings[label]
            position = positions[label]
            while position < ranking.numel() and used[ranking[position]]:
                position += 1
            positions[label] = position + 1
            if position >= ranking.numel():
                continue
            node = ranking[position]
            selected_lists[label].append(node)
            used[node] = True
            remaining -= 1
            progress = True
        if not progress:
            raise RuntimeError('Unable to construct disjoint K-anchor inference groups.')

    inferred_groups = {}
    poisoned_groups = {}
    lia_by_target = {}
    correct_total = 0
    inferred_total = 0
    labels_for_evaluation = data.y.to(device)
    for label in target_labels:
        if selected_lists[label]:
            inferred = torch.stack(selected_lists[label]).long()
        else:
            inferred = torch.empty(0, dtype=torch.long, device=device)
        anchors = anchor_tensors[label]
        poisoned = torch.unique(torch.cat([inferred, anchors]))
        inferred_groups[label] = inferred
        poisoned_groups[label] = poisoned

        correct = int((labels_for_evaluation[inferred] == int(label)).sum().item())
        count = int(inferred.numel())
        lia_by_target[label] = float(correct / count) if count else float('nan')
        correct_total += correct
        inferred_total += count

    overall_lia = float(correct_total / inferred_total) if inferred_total else float('nan')
    data['inferred_train_nodes_by_target'] = inferred_groups
    data['poisoned_train_nodes_by_target'] = poisoned_groups
    data['inferred_train_nodes'] = merge_multi_target_nodes(inferred_groups, device=device)
    data['poisoned_train_nodes'] = merge_multi_target_nodes(poisoned_groups, device=device)
    data['number_of_inferred'] = inferred_total
    data['number_of_poison'] = int(sum(nodes.numel() for nodes in poisoned_groups.values()))
    data['label_inference_accuracy_by_target'] = lia_by_target
    data['label_inference_accuracy'] = overall_lia

    # Rebuild the ASR set after inference as an explicit safety guard. Since
    # V_cand is training-only, the inferred/anchor set and ASR test set are
    # already disjoint under a valid train/test split.
    _select_asr_test_groups(
        data,
        config,
        excluded_nodes=data['poisoned_train_nodes'],
    )

    return {
        'inferred_groups': inferred_groups,
        'poisoned_groups': poisoned_groups,
        'lia_by_target': lia_by_target,
        'overall_lia': overall_lia,
    }


def _normalized_poison_groups(data, config, device):
    target_labels = resolve_target_labels(config)
    poisoned_train_groups = data.get('poisoned_train_nodes_by_target') or {}
    poisoned_test_groups = data.get('poisoned_test_nodes_by_target') or {}
    attributes_by_target = data.get('poisoned_attributes_by_target') or {}

    normalized_train_groups = {}
    normalized_test_groups = {}
    normalized_attributes = {}
    for label in target_labels:
        train_nodes = _mapping_value(poisoned_train_groups, label)
        test_nodes = _mapping_value(poisoned_test_groups, label)
        attributes = _mapping_value(attributes_by_target, label)
        if attributes is None:
            attributes = data.get('poisoned_attributes')
        normalized_train_groups[label] = _cpu_long(
            train_nodes if train_nodes is not None else []
        ).to(device)
        normalized_test_groups[label] = _cpu_long(
            test_nodes if test_nodes is not None else []
        ).to(device)
        normalized_attributes[label] = _cpu_long(
            attributes if attributes is not None else []
        ).to(device)
    return (
        target_labels,
        normalized_train_groups,
        normalized_test_groups,
        normalized_attributes,
    )


def apply_graph_infiltrator_poison(data, config):
    """Build the persistent training view of the Graph-Infiltrator attack.

    Only poisoned training nodes are replaced, triggered, and disconnected in
    this view. ASR test nodes remain clean so they cannot leak into later
    training rounds; their temporary test-only view is built separately by
    :func:`apply_graph_infiltrator_test_view`.
    """

    if bool(data.get('graph_infiltrator_poison_applied', False)):
        return data

    device = data.x.device
    (
        target_labels,
        normalized_train_groups,
        normalized_test_groups,
        normalized_attributes,
    ) = _normalized_poison_groups(data, config, device)

    ordered_groups = [
        normalized_train_groups[label]
        for label in target_labels
        if normalized_train_groups[label].numel() > 0
    ]
    poisoned_train_nodes = (
        torch.cat(ordered_groups)
        if ordered_groups
        else torch.empty(0, dtype=torch.long, device=device)
    )
    if torch.unique(poisoned_train_nodes).numel() != poisoned_train_nodes.numel():
        raise ValueError('Poisoned training groups must be disjoint across target labels.')
    poisoned_test_nodes = merge_multi_target_nodes(normalized_test_groups, device=device)
    if poisoned_train_nodes.numel() == 0:
        raise RuntimeError('K-anchor inference must select training nodes before poisoning.')
    overlap = poisoned_test_nodes[
        torch.isin(poisoned_test_nodes, poisoned_train_nodes)
    ]
    if overlap.numel() > 0:
        raise ValueError(
            'ASR test nodes must be disjoint from V_inf union V_anc; '
            f'overlap={overlap.detach().cpu().tolist()}.'
        )

    # 1) Node replacement over the complete V_inf union V_anc set. The skip
    # switch is an explicit ablation only; the full-pipeline default is False.
    skip_node_replacement = bool(getattr(config, 'skip_gi_node_replacement', False))
    substitute_by_target = {}
    if skip_node_replacement:
        replacement_nodes = torch.empty(0, dtype=torch.long, device=device)
        for label in target_labels:
            substitute_by_target[label] = torch.empty(0, dtype=torch.long, device=device)
    else:
        # Keep replacement donors inside the training split, matching the
        # legacy implementation's access boundary while still excluding every
        # node in the complete V_inf union V_anc poisoning set.
        train_mask = data.train_mask.to(device=device, dtype=torch.bool)
        if train_mask.ndim != 1 or train_mask.numel() != data.x.shape[0]:
            raise ValueError('train_mask must contain one boolean value per node.')
        train_nodes = torch.where(train_mask)[0]
        replacement_pool = train_nodes[
            ~torch.isin(train_nodes, poisoned_train_nodes)
        ]
        if replacement_pool.numel() < poisoned_train_nodes.numel():
            raise ValueError(
                'Not enough non-poisoned training nodes to perform distinct '
                'node replacement.'
            )
        replacement_generator = torch.Generator().manual_seed(int(getattr(config, 'seed', 0)) + 5003)
        replacement_order = torch.randperm(replacement_pool.numel(), generator=replacement_generator)
        replacement_nodes = replacement_pool.detach().cpu()[replacement_order[:poisoned_train_nodes.numel()]].to(device)
        data.x[poisoned_train_nodes] = data.x[replacement_nodes].detach().clone()

        offset = 0
        for label in target_labels:
            count = normalized_train_groups[label].numel()
            substitute_by_target[label] = replacement_nodes[offset:offset + count]
            offset += count
    data['substitute_train_nodes'] = replacement_nodes
    data['substitute_train_nodes_by_target'] = substitute_by_target

    # 2) Target-specific trigger injection with per-node attribute dropping.
    dropout_ratio = max(0.0, min(1.0, float(getattr(config, 'dropout_ratio', 0.0))))
    train_trigger_values = 0
    train_trigger_slots = 0
    test_trigger_values = 0
    trigger_masks_by_target = {}
    for index, label in enumerate(target_labels):
        attributes = normalized_attributes[label]
        train_group = normalized_train_groups[label]

        if train_group.numel() > 0 and attributes.numel() > 0:
            mask_generator = torch.Generator().manual_seed(
                int(getattr(config, 'seed', 0)) + 7001 * (index + 1)
            )
            keep_mask = torch.rand(
                (train_group.numel(), attributes.numel()), generator=mask_generator
            ) >= dropout_ratio
            keep_mask = keep_mask.to(device)
            rows = train_group[:, None].expand(-1, attributes.numel())
            cols = attributes[None, :].expand(train_group.numel(), -1)
            values = data.x[rows, cols].clone()
            values[keep_mask] = 1
            data.x[rows, cols] = values
            trigger_masks_by_target[label] = keep_mask
            train_trigger_values += int(keep_mask.sum().item())
            train_trigger_slots += int(keep_mask.numel())
        else:
            trigger_masks_by_target[label] = torch.empty(
                (train_group.numel(), attributes.numel()), dtype=torch.bool, device=device
            )

        test_group = normalized_test_groups[label]
        test_trigger_values += int(test_group.numel() * attributes.numel())

    data['trigger_masks_by_target'] = trigger_masks_by_target

    # 3) Persistent training link pruning applies only to poisoned training
    # nodes. Test-node topology is changed only inside the temporary ASR view.
    edge_count_before = int(data.edge_index.shape[1])
    skip_edge_pruning = bool(getattr(config, 'skip_gi_edge_pruning', False))
    if not skip_edge_pruning:
        incident = torch.isin(data.edge_index[0], poisoned_train_nodes) | torch.isin(
            data.edge_index[1], poisoned_train_nodes
        )
        data.edge_index = data.edge_index[:, ~incident].contiguous()
    removed_edges = edge_count_before - int(data.edge_index.shape[1])

    data['poisoned_train_nodes'] = poisoned_train_nodes
    data['poisoned_test_nodes'] = poisoned_test_nodes
    data['number_of_poison'] = int(poisoned_train_nodes.numel())
    data['poisoning_stats'] = {
        'replacement_count': int(replacement_nodes.numel()),
        'node_replacement_skipped': skip_node_replacement,
        'train_trigger_values': train_trigger_values,
        'train_trigger_slots': train_trigger_slots,
        'test_trigger_values': test_trigger_values,
        'test_trigger_values_per_evaluation': test_trigger_values,
        'train_asr_overlap_count': 0,
        'removed_edge_columns': removed_edges,
        'training_removed_edge_columns': removed_edges,
        'edge_pruning_skipped': skip_edge_pruning,
    }
    data['graph_infiltrator_poison_applied'] = True
    return data


def apply_graph_infiltrator_test_view(data, config):
    """Inject and isolate ASR test nodes in a temporary clean graph view.

    The caller must construct this view from the attack-before snapshot and
    discard it after evaluation. This function intentionally has no one-shot
    guard because a fresh ASR view is required at every recorded test round.
    """

    device = data.x.device
    (
        target_labels,
        normalized_train_groups,
        normalized_test_groups,
        normalized_attributes,
    ) = _normalized_poison_groups(data, config, device)
    poisoned_train_nodes = merge_multi_target_nodes(
        normalized_train_groups, device=device
    )
    poisoned_test_nodes = merge_multi_target_nodes(
        normalized_test_groups, device=device
    )
    overlap = poisoned_test_nodes[torch.isin(poisoned_test_nodes, poisoned_train_nodes)]
    if overlap.numel() > 0:
        raise ValueError(
            'ASR test nodes must be disjoint from poisoned training nodes; '
            f'overlap={overlap.detach().cpu().tolist()}.'
        )

    test_trigger_values = 0
    for label in target_labels:
        test_group = normalized_test_groups[label]
        attributes = normalized_attributes[label]
        if test_group.numel() == 0 or attributes.numel() == 0:
            continue
        rows = test_group[:, None].expand(-1, attributes.numel())
        cols = attributes[None, :].expand(test_group.numel(), -1)
        data.x[rows, cols] = 1
        test_trigger_values += int(test_group.numel() * attributes.numel())

    edge_count_before = int(data.edge_index.shape[1])
    skip_edge_pruning = bool(getattr(config, 'skip_gi_edge_pruning', False))
    if not skip_edge_pruning and poisoned_test_nodes.numel() > 0:
        incident = torch.isin(data.edge_index[0], poisoned_test_nodes) | torch.isin(
            data.edge_index[1], poisoned_test_nodes
        )
        data.edge_index = data.edge_index[:, ~incident].contiguous()
    stats = {
        'test_nodes': int(poisoned_test_nodes.numel()),
        'test_trigger_values': test_trigger_values,
        'removed_edge_columns': edge_count_before - int(data.edge_index.shape[1]),
        'edge_pruning_skipped': skip_edge_pruning,
    }
    data['asr_view_stats'] = stats
    return data
