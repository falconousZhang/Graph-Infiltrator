import torch.nn as nn
import torch


class VFLActiveModel(nn.Module):
    def __init__(self, config):
        super(VFLActiveModel, self).__init__()
        self.combine_strategy = config.combine_strategy
        self.embedding_clip_threshold = getattr(config, 'embedding_clip_threshold', None)
        self.gradient_noise_scale = float(getattr(config, 'gradient_noise_scale', 0.0))
        if self.combine_strategy=='concat':
            self.d1 = nn.Linear(in_features=config.out_channels*config.num_of_clients, out_features=32)
        elif self.combine_strategy=='mean':
            self.d1 = nn.Linear(in_features=config.out_channels, out_features=32)
        elif self.combine_strategy=='regression':
            self.weights = nn.Parameter(torch.rand(config.num_of_clients,config.out_channels))
            self.d1 = nn.Linear(in_features=config.out_channels, out_features=32)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(in_features=32, out_features=config.num_of_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Paper defense experiments are applied at the active party, which is
        # the only party that observes every submitted embedding and returns
        # its corresponding gradient.
        if self.embedding_clip_threshold is not None:
            threshold = float(self.embedding_clip_threshold)
            if threshold <= 0:
                raise ValueError('embedding_clip_threshold must be positive.')
            clipped = []
            for embedding in x:
                norm = torch.linalg.vector_norm(embedding, ord=2, dim=1, keepdim=True)
                scale = torch.clamp(threshold / norm.clamp_min(1e-12), max=1.0)
                clipped.append(embedding * scale)
            x = clipped

        if self.training and self.gradient_noise_scale > 0:
            noise_scale = self.gradient_noise_scale

            def add_laplace_noise(gradient):
                # Inverse-CDF sampling avoids constructing CPU distribution
                # parameters and preserves the gradient's device and dtype.
                uniform = torch.rand_like(gradient).clamp_(1e-7, 1.0 - 1e-7) - 0.5
                noise = -noise_scale * torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs())
                return gradient + noise

            for embedding in x:
                if embedding.requires_grad:
                    embedding.register_hook(add_laplace_noise)

        if self.combine_strategy == 'concat':
            x = torch.cat(x, dim=1)
        elif self.combine_strategy == 'mean':
            x = torch.mean(torch.stack(x,dim=0), dim=0)
        elif self.combine_strategy == 'regression':
            # TODO 检查是否符合结果
            # 将self.weigthts和x相乘
            x = torch.stack(x,dim=1)
            x = torch.sum(x*self.weights,dim=1)
        x = self.d1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.out(x)
        # CrossEntropyLoss expects unnormalized logits. Apply softmax only in
        # presentation code when probabilities are explicitly needed.
        return x
