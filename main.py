"""Run Graph-Infiltrator."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from client import Client
from cmd_args import parse_args
from dataloader_node import load_nodelevel_dataset, setup_data
from graph_infiltrator import graph_infiltrator
from server import Server


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    config = parse_args(argv)
    if config.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu.')
    config.device = torch.device(config.device)
    if config.dry_run:
        print(json.dumps(vars(config), indent=2, default=str))
        return 0

    torch.set_num_threads(config.threads)
    setup_seed(config.seed)
    data = load_nodelevel_dataset(config)
    workers = [Server(data[0], config)]
    workers.extend(Client(i, data[i], config) for i in range(1, config.num_of_clients + 1))
    setup_data(data[config.attack_client_id], config)
    for participant in data.values():
        participant.to(config.device)
    setup_seed(config.seed + 900001)

    output = Path(config.basedir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(
        json.dumps(vars(config), indent=2, default=str) + '\n', encoding='utf-8'
    )
    graph_infiltrator(workers, config)
    attacker = workers[config.attack_client_id]

    def last_finite(values):
        return next((float(value) for value in reversed(values) if math.isfinite(value)), None)

    result = {
        'retained_test_accuracy': last_finite(workers[0].test_accuracy),
        'asr': last_finite(attacker.backdoor_test_accuracy),
        'lia': last_finite(attacker.inference_acc),
    }
    (output / 'result.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print(f'Results: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
