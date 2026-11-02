"""Graph-Infiltrator configuration."""

import argparse
from pathlib import Path

from result_paths import RESULTS_ROOT


DATASETS = {
    'cora': dict(classes=7, epochs=100, attack_round=16, anchors=3, ratio=0.05, lr_factor=5.0),
    'citeseer': dict(classes=6, epochs=100, attack_round=16, anchors=5, ratio=0.05, lr_factor=5.0),
    'pubmed': dict(classes=3, epochs=100, attack_round=41, anchors=5, ratio=0.05, lr_factor=5.0),
    'photo': dict(classes=8, epochs=50, attack_round=41, anchors=5, ratio=0.15, lr_factor=10.0),
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Run Graph-Infiltrator.')
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--model', choices=['gcn', 'gat', 'sage'], default='gcn')
    parser.add_argument('--aggregation', dest='combine_strategy',
                        choices=['concat', 'mean', 'regression'], default='concat')
    parser.add_argument('--epochs', dest='num_epoch', type=int)
    parser.add_argument('--attack-round', type=int)
    parser.add_argument('--num-clients', dest='num_of_clients', type=int, default=2)
    parser.add_argument('--attack-client-id', type=int, default=1)
    parser.add_argument('--target-labels', type=int, nargs='+', default=[0])
    parser.add_argument('--anchors', dest='num_of_start_node', type=int)
    parser.add_argument('--poison-ratio', dest='num_train_poisoning_ratio', type=float)
    parser.add_argument('--test-ratio', dest='num_test_poisoning_ratio', type=float, default=0.05)
    parser.add_argument('--trigger-ratio', dest='num_poisoning_feature_ratio', type=float)
    parser.add_argument('--lr-factor', dest='attack_lr_boost', type=float)
    parser.add_argument('--edge-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    config = parser.parse_args(args)
    defaults = DATASETS[config.dataset]
    for field, value in {
        'num_epoch': defaults['epochs'],
        'attack_round': defaults['attack_round'],
        'num_of_start_node': defaults['anchors'],
        'num_train_poisoning_ratio': defaults['ratio'],
        'num_poisoning_feature_ratio': defaults['ratio'],
        'attack_lr_boost': defaults['lr_factor'],
    }.items():
        if getattr(config, field) is None:
            setattr(config, field, value)

    if not 1 <= config.attack_round <= config.num_epoch:
        parser.error('--attack-round must be between 1 and --epochs.')
    if config.num_of_clients < 2:
        parser.error('--num-clients must be at least 2.')
    if not 1 <= config.attack_client_id <= config.num_of_clients:
        parser.error('--attack-client-id must identify an existing client.')
    if any(label < 0 or label >= defaults['classes'] for label in config.target_labels):
        parser.error('--target-labels must be valid class indices for the dataset.')
    if config.num_of_start_node < 1 or config.threads < 1:
        parser.error('--anchors and --threads must be positive.')
    if not 0 < config.attack_lr_boost < float('inf'):
        parser.error('--lr-factor must be finite and positive.')
    for field in ('num_train_poisoning_ratio', 'num_test_poisoning_ratio',
                  'num_poisoning_feature_ratio', 'edge_ratio'):
        if not 0 <= getattr(config, field) <= 1:
            parser.error(f'{field} must be between 0 and 1.')

    config.num_of_classes = defaults['classes']
    config.target_label = config.target_labels[0]
    config.hidden = 64
    config.out_channels = 32
    config.layer = 2
    config.dropout = 0.5
    config.dropout_ratio = 0.5
    config.lr = 0.005
    config.no_edge = 0
    config.task = 'node'
    config.training_mode = 'backdoor'
    config.attack_mode = 'graph_infiltrator'
    config.enable_lr_augmentation = True
    config.basedir = str(config.output or (
        RESULTS_ROOT / config.dataset / config.model / config.combine_strategy / str(config.seed)
    ))
    del config.output
    return config
