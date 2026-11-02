import numpy as np
import torch


def resolve_attack_round(config):
    """Return the one-based round in which poisoning becomes active."""
    attack_round = getattr(config, 'attack_round', None)
    if attack_round is not None:
        attack_round = int(attack_round)
        if attack_round < 1:
            raise ValueError('attack_round must be a positive one-based round number.')
        return attack_round
    # Backward-compatible interpretation for legacy configurations.
    return int(getattr(config, 'start_attack_epoch', 0)) + 2


def attack_is_active(epoch_zero_based, config):
    return int(epoch_zero_based) + 1 >= resolve_attack_round(config)


def resolve_target_labels(config):
    target_labels = getattr(config, 'target_labels', None)
    if target_labels is None or target_labels == []:
        target_label = getattr(config, 'target_label', None)
        if target_label is None:
            return []
        target_labels = [target_label]
    elif isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.detach().cpu().tolist()
    elif not isinstance(target_labels, (list, tuple)):
        target_labels = [target_labels]

    normalized = []
    for label in target_labels:
        label = int(label)
        if label not in normalized:
            normalized.append(label)
    return normalized


def distribute_quota(total_count, num_groups):
    if num_groups <= 0:
        return []
    base = total_count // num_groups
    remainder = total_count % num_groups
    return [base + (1 if idx < remainder else 0) for idx in range(num_groups)]


def get_multi_target_groups(data, config, split='train', device=None):
    """Return label -> node-index tensors for single-target and multi-target runs.

    In multi-target mode, the dataloader stores per-label node buckets under
    `poisoned_{split}_nodes_by_target`. In single-target mode, this falls back to
    the legacy flat tensor keys so existing experiments keep working unchanged.
    """
    target_labels = resolve_target_labels(config)
    grouped_nodes = data.get(f'poisoned_{split}_nodes_by_target')

    if grouped_nodes is None:
        legacy_nodes = data.get(f'poisoned_{split}_nodes')
        if legacy_nodes is None:
            return target_labels, {}
        if not isinstance(legacy_nodes, torch.Tensor):
            legacy_nodes = torch.tensor(legacy_nodes, dtype=torch.long)
        if device is not None:
            legacy_nodes = legacy_nodes.to(device)
        label = target_labels[0] if target_labels else getattr(config, 'target_label')
        return [label], {label: legacy_nodes}

    normalized_groups = {}
    for label in target_labels:
        nodes = grouped_nodes.get(label)
        if nodes is None:
            nodes = grouped_nodes.get(str(label))
        if nodes is None:
            continue
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        if device is not None:
            nodes = nodes.to(device)
        normalized_groups[label] = nodes
    return target_labels, normalized_groups


def merge_multi_target_nodes(group_map, device=None):
    tensors = []
    for nodes in group_map.values():
        if nodes is None:
            continue
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        if nodes.numel() == 0:
            continue
        tensors.append(nodes)
    if not tensors:
        return torch.empty(0, dtype=torch.long, device=device)
    merged = torch.unique(torch.cat(tensors))
    if device is not None:
        merged = merged.to(device)
    return merged
