from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from model.gcn import GCN_Net
from model.sage import SAGE_Net
from model.gat import GAT_Net


def get_model(config,input_shape):

    x_shape, num_edge_features = input_shape
    if config.task.startswith('node'):
        if config.model == 'gcn':
            # assume `data` is a dict where key is the client index,
            # and value is a PyG object
            model = GCN_Net(x_shape[-1],
                            config.out_channels,
                            hidden=config.hidden,
                            max_depth=config.layer,
                            dropout=config.dropout)
        elif config.model == 'sage':
            model = SAGE_Net(x_shape[-1],
                             config.out_channels,
                             hidden=config.hidden,
                             max_depth=config.layer,
                             dropout=config.dropout)
        elif config.model == 'gat':
            model = GAT_Net(x_shape[-1],
                            config.out_channels,
                            hidden=config.hidden,
                            max_depth=config.layer,
                            dropout=config.dropout)
        else:
            raise ValueError('not recognized gnn model {}'.format(
                config.model))
    else:
        raise ValueError('not recognized data task {}'.format(
            config.task))
    return model
