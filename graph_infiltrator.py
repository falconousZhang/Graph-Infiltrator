import csv
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn.functional as F

from dataloader_node import poison_data
from graph_infiltrator_core import (
    apply_graph_infiltrator_test_view,
    infer_k_anchor_groups,
)
from utils import (
    attack_is_active,
    get_multi_target_groups,
    merge_multi_target_nodes,
    resolve_target_labels,
)


def _write_columns(path, header, columns):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip_longest(*columns, fillvalue=''))


def save_metrics(server, backdoor_client):
    output = Path(server.config.basedir)
    output.mkdir(parents=True, exist_ok=True)
    seed = server.config.seed
    _write_columns(
        output / f'loss_{seed}.csv',
        ['train_loss', 'test_loss', 'backdoor_train_loss', 'backdoor_test_loss'],
        [server.train_loss, server.test_loss,
         backdoor_client.backdoor_train_loss, backdoor_client.backdoor_test_loss],
    )
    _write_columns(
        output / f'accuracy_{seed}.csv',
        ['train_accuracy', 'test_accuracy', 'backdoor_train_accuracy',
         'backdoor_test_accuracy', 'label_inference_accuracy'],
        [server.train_accuracy, server.test_accuracy,
         backdoor_client.backdoor_train_accuracy,
         backdoor_client.backdoor_test_accuracy,
         backdoor_client.inference_acc],
    )


def calculate_mse(h_clean, h_poisoned, target_nodes):
    """MSE stealthiness metric on the specified triggered nodes."""
    return ((h_clean[target_nodes] - h_poisoned[target_nodes]) ** 2).mean()


def _safe_mean(values, device):
    if not values:
        return torch.tensor(0.0, device=device)
    return torch.stack(values).mean()


def _set_train_mode(workers):
    for worker in workers:
        worker.model.train()


def _capture_malicious_embeddings(client):
    """Capture embeddings for label inference without changing train/eval state."""
    was_training = client.model.training
    client.model.eval()
    with torch.no_grad():
        embeddings = client.model(client.data).detach().clone()
    client.model.train(was_training)
    return embeddings


def graph_infiltrator(workers, config):
    for worker in workers:
        worker.data.to(worker.config.device)

    server = workers[0]
    backdoor_client = workers[config.attack_client_id]
    train_mask = server.data.train_mask
    test_mask = server.data.test_mask
    attack_enabled = 'backdoor' in config.training_mode

    target_labels = resolve_target_labels(config)
    if not target_labels:
        raise ValueError('Graph-Infiltrator requires at least one target label.')
    if attack_enabled and backdoor_client.data.get('anchor_nodes_by_target') is None:
        raise RuntimeError('Call setup_data on the malicious client before graph_infiltrator.')

    poisoned_train_groups = {}
    poisoned_train_nodes = torch.empty(0, dtype=torch.long, device=config.device)
    if attack_enabled:
        _, poisoned_test_groups = get_multi_target_groups(
            backdoor_client.data, config, split='test', device=config.device
        )
        poisoned_test_nodes = merge_multi_target_nodes(
            poisoned_test_groups, device=config.device
        )
    else:
        poisoned_test_groups = {}
        poisoned_test_nodes = torch.empty(0, dtype=torch.long, device=config.device)

    # Immutable attack-before view and the persistent training-only poisoned
    # view. ASR evaluation is always built temporarily from the former.
    clean_x = backdoor_client.data.x.detach().clone()
    clean_edge_index = backdoor_client.data.edge_index.detach().clone()
    training_x = clean_x
    training_edge_index = clean_edge_index
    attack_started = False

    for epoch in range(config.num_epoch):
        _set_train_mode(workers)

        # Candidate labels are not consulted here. data.y is read only after the
        # embedding-based ranking has finished, solely to evaluate LIA.
        if attack_enabled and not attack_started and attack_is_active(epoch, config):
            embeddings = _capture_malicious_embeddings(backdoor_client)
            inference = infer_k_anchor_groups(backdoor_client.data, embeddings, config)
            backdoor_client.inference_acc.append(inference['overall_lia'])
            backdoor_client.inference_acc_by_target.append(inference['lia_by_target'])

            poison_data(backdoor_client.data, config)
            _, poisoned_train_groups = get_multi_target_groups(
                backdoor_client.data, config, split='train', device=config.device
            )
            poisoned_train_nodes = merge_multi_target_nodes(
                poisoned_train_groups, device=config.device
            )
            # K-anchor inference rebuilds the ASR set after V_inf is known.
            # Refresh the pre-loop cache so evaluation uses that disjoint set.
            _, poisoned_test_groups = get_multi_target_groups(
                backdoor_client.data, config, split='test', device=config.device
            )
            poisoned_test_nodes = merge_multi_target_nodes(
                poisoned_test_groups, device=config.device
            )
            # poison_data builds only the persistent training view. Preserve it
            # verbatim so ASR evaluation can be discarded without resampling
            # node replacements or trigger-dropout masks.
            training_x = backdoor_client.data.x.detach().clone()
            training_edge_index = backdoor_client.data.edge_index.detach().clone()
            attack_started = True

            print(
                '[Graph-Infiltrator] K-anchor inference complete: '
                f"LIA={inference['overall_lia']:.4f}, "
                f"inferred={backdoor_client.data.number_of_inferred}, "
                f"poisoned={backdoor_client.data.number_of_poison}, "
                f"stats={backdoor_client.data.poisoning_stats}"
            )

        passive_outputs = []
        for client_id in range(1, config.num_of_clients + 1):
            cur_client = workers[client_id]
            cur_client.data.to(cur_client.config.device)
            passive_outputs.append(cur_client.model(cur_client.data))

        for passive_output in passive_outputs:
            passive_output.retain_grad()

        active_output = server.model(passive_outputs)
        server.optimizer.zero_grad()
        loss = F.cross_entropy(active_output[train_mask], server.data.y[train_mask])
        loss.backward(retain_graph=True)
        server.optimizer.step()

        for client_id in range(1, config.num_of_clients + 1):
            cur_client = workers[client_id]
            passive_loss = torch.mul(
                passive_outputs[client_id - 1][train_mask],
                passive_outputs[client_id - 1].grad[train_mask],
            )
            cur_client.optimizer.zero_grad()
            passive_loss.backward(torch.ones_like(passive_loss), retain_graph=True)
            cur_client.optimizer.step()

        server.train_loss.append(loss.item())
        train_accuracy = (
            (server.data.y == active_output.argmax(dim=1))[train_mask].sum().item()
            / train_mask.sum().item()
        )
        server.train_accuracy.append(train_accuracy)

        if attack_started and poisoned_train_nodes.numel() > 0:
            per_target_train_loss = {}
            per_target_train_accuracy = {}
            train_loss_values = []
            for label, nodes in poisoned_train_groups.items():
                if nodes.numel() == 0:
                    continue
                cur_loss = F.cross_entropy(
                    active_output[nodes],
                    torch.full((nodes.shape[0],), label, device=config.device),
                )
                train_loss_values.append(cur_loss)
                per_target_train_loss[label] = cur_loss.item()
                per_target_train_accuracy[label] = (
                    (active_output.argmax(dim=1) == label)[nodes].sum().item()
                    / nodes.shape[0]
                )

            backdoor_train_loss = _safe_mean(train_loss_values, config.device)
            backdoor_train_accuracy = (
                sum(per_target_train_accuracy.values()) / len(per_target_train_accuracy)
                if per_target_train_accuracy
                else 0.0
            )
            backdoor_client.backdoor_train_loss.append(backdoor_train_loss.item())
            backdoor_client.backdoor_train_accuracy.append(backdoor_train_accuracy)
            backdoor_client.backdoor_train_loss_by_target.append(per_target_train_loss)
            backdoor_client.backdoor_train_accuracy_by_target.append(per_target_train_accuracy)
        else:
            # Preserve epoch alignment while marking pre-attack values undefined.
            backdoor_client.backdoor_train_loss.append(float('nan'))
            backdoor_client.backdoor_train_accuracy.append(float('nan'))
            backdoor_client.backdoor_train_loss_by_target.append({})
            backdoor_client.backdoor_train_accuracy_by_target.append({})

        server.model.eval()
        for client_id in range(1, config.num_of_clients + 1):
            workers[client_id].model.eval()

        try:
            with torch.no_grad():
                if attack_started and poisoned_test_nodes.numel() > 0:
                    # ASR view: start from the immutable attack-before graph,
                    # inject only test triggers, and prune only test-node edges.
                    backdoor_client.data.x = clean_x.clone()
                    backdoor_client.data.edge_index = clean_edge_index.clone()
                    apply_graph_infiltrator_test_view(backdoor_client.data, config)
                    attack_passive_outputs = [
                        workers[client_id].model(workers[client_id].data)
                        for client_id in range(1, config.num_of_clients + 1)
                    ]
                    attack_test_output = server.model(attack_passive_outputs)

                    per_target_test_loss = {}
                    per_target_test_accuracy = {}
                    test_loss_values = []
                    for label, nodes in poisoned_test_groups.items():
                        if nodes.numel() == 0:
                            continue
                        cur_loss = F.cross_entropy(
                            attack_test_output[nodes],
                            torch.full((nodes.shape[0],), label, device=config.device),
                        )
                        test_loss_values.append(cur_loss)
                        per_target_test_loss[label] = cur_loss.item()
                        per_target_test_accuracy[label] = (
                            (attack_test_output.argmax(dim=1) == label)[nodes].sum().item()
                            / nodes.shape[0]
                        )

                    backdoor_test_loss = _safe_mean(test_loss_values, config.device)
                    backdoor_test_accuracy = (
                        sum(per_target_test_accuracy.values())
                        / len(per_target_test_accuracy)
                        if per_target_test_accuracy
                        else 0.0
                    )
                    backdoor_client.backdoor_test_loss.append(backdoor_test_loss.item())
                    backdoor_client.backdoor_test_accuracy.append(backdoor_test_accuracy)
                    backdoor_client.backdoor_test_loss_by_target.append(per_target_test_loss)
                    backdoor_client.backdoor_test_accuracy_by_target.append(
                        per_target_test_accuracy
                    )
                    h_poisoned_eval = attack_passive_outputs[
                        config.attack_client_id - 1
                    ]

                    # Clean evaluation view: no training poison and no ASR
                    # trigger/topology changes are visible to the test inputs.
                    backdoor_client.data.x = clean_x
                    backdoor_client.data.edge_index = clean_edge_index
                    clean_passive_outputs = [
                        workers[client_id].model(workers[client_id].data)
                        for client_id in range(1, config.num_of_clients + 1)
                    ]
                    clean_test_output = server.model(clean_passive_outputs)
                    h_clean_eval = clean_passive_outputs[config.attack_client_id - 1]

                    per_target_mse = {}
                    mse_values = []
                    for label, nodes in poisoned_test_groups.items():
                        if nodes.numel() == 0:
                            continue
                        mse = calculate_mse(h_clean_eval, h_poisoned_eval, nodes)
                        per_target_mse[label] = mse.item()
                        mse_values.append(mse)
                    backdoor_client.backdoor_mse.append(
                        torch.stack(mse_values).mean().item()
                        if mse_values
                        else float('nan')
                    )
                    backdoor_client.backdoor_mse_by_target.append(per_target_mse)
                else:
                    clean_passive_outputs = [
                        workers[client_id].model(workers[client_id].data)
                        for client_id in range(1, config.num_of_clients + 1)
                    ]
                    clean_test_output = server.model(clean_passive_outputs)
                    backdoor_client.backdoor_test_loss.append(float('nan'))
                    backdoor_client.backdoor_test_accuracy.append(float('nan'))
                    backdoor_client.backdoor_test_loss_by_target.append({})
                    backdoor_client.backdoor_test_accuracy_by_target.append({})
                    backdoor_client.backdoor_mse.append(float('nan'))
                    backdoor_client.backdoor_mse_by_target.append({})

                test_loss = F.cross_entropy(
                    clean_test_output[test_mask], server.data.y[test_mask]
                )
                server.test_loss.append(test_loss.item())
                clean_test_nodes = (
                    backdoor_client.data.retained_test_nodes
                    if attack_enabled
                    else torch.where(test_mask)[0]
                )
                correct = (
                    clean_test_output.argmax(dim=1) == server.data.y
                )[clean_test_nodes].sum().item()
                test_accuracy = correct / clean_test_nodes.shape[0]
                server.test_accuracy.append(test_accuracy)
        finally:
            # Discard every test-only mutation. The next round receives exactly
            # the persistent training view, including only train-node poison.
            if attack_started:
                backdoor_client.data.x = training_x
                backdoor_client.data.edge_index = training_edge_index

        print(
            'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(
                epoch + 1,
                loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
            )
        )
        if attack_started and poisoned_train_nodes.numel() > 0:
            print(
                'Poisoned {}, Backdoor Train Loss: {}, Backdoor Train Accuracy: {}, '
                'Backdoor Test Loss: {}, Backdoor Test Accuracy: {}, MSE: {}'.format(
                    poisoned_train_nodes.shape[0],
                    backdoor_client.backdoor_train_loss[-1],
                    backdoor_client.backdoor_train_accuracy[-1] * 100,
                    backdoor_client.backdoor_test_loss[-1],
                    backdoor_client.backdoor_test_accuracy[-1] * 100,
                    backdoor_client.backdoor_mse[-1],
                )
            )
            print(
                'Per-target train ASR: {}'.format(
                    backdoor_client.backdoor_train_accuracy_by_target[-1]
                )
            )
            print(
                'Per-target test ASR: {}'.format(
                    backdoor_client.backdoor_test_accuracy_by_target[-1]
                )
            )

    save_metrics(server, backdoor_client)
    return server.test_accuracy, backdoor_client.backdoor_test_accuracy
