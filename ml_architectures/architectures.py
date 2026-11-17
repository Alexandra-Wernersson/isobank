"""The networks of the three designs, with the same layers and initialisation as in our full model.

1. SVD + network (FlatDirectNetwork): theta -> n_layers hidden layers of one width, each Linear -> LayerNorm ->
   GELU, then a linear output layer giving the SVD coefficients. The curve is coefficients @ basis + mean.
2. SVD + autoencoder (FunnelAutoencoder + ParameterToDecoder):
   encoder  coefficients -> 8h -> 4h -> 2h -> h -> L (latent),      LeakyReLU(0.3), L2 latent penalty
   decoder  L -> h -> 2h -> 4h -> 8h -> coefficients,               LeakyReLU(0.3)
   parameter network  theta -> h -> 4h -> 4h -> h -> L,            LayerNorm + GELU, feeding the pretrained decoder
3. Average = the mean of two models:
   A  the same funnel autoencoder on the data itself (every sample), with its parameter network;
   B  the direct network of design 1.

Only torch is needed.
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------------------------ layers
def linear_block(n_in: int, n_out: int, activation: bool = True) -> nn.Sequential:
    """Linear + LeakyReLU(0.3); Kaiming-uniform weights, zero bias (the encoder and decoder block)"""
    layer = nn.Linear(n_in, n_out)
    if activation:
        nn.init.kaiming_uniform_(layer.weight, a=0.3, nonlinearity="leaky_relu")
    else:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
    nn.init.zeros_(layer.bias)
    return nn.Sequential(layer, nn.LeakyReLU(negative_slope=0.3)) if activation else nn.Sequential(layer)


def linear_block_param(n_in: int, n_out: int, layernorm: bool = True, activation: bool = True) -> nn.Sequential:
    """Linear + LayerNorm + GELU (the parameter-network block; its first block has no GELU)"""
    layer = nn.Linear(n_in, n_out)
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu" if activation else "linear")
    nn.init.zeros_(layer.bias)
    layers = [layer]
    if layernorm:
        layers.append(nn.LayerNorm(n_out))
    if activation:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def latent_penalty(z: torch.Tensor, regulariser: float) -> torch.Tensor:
    """L2 shrinkage of the latent vector, regulariser * <|z|^2> over the batch"""
    return regulariser * torch.mean(torch.sum(z ** 2, dim=1)) if regulariser else z.new_zeros(())


# ------------------------------------------------------------------------------------------------ 2./3A. funnel autoencoder
class Encoder(nn.Module):
    """signal -> 8h -> 4h -> 2h -> h -> latent L; returns (z, latent penalty)"""

    def __init__(self, n_signal: int, latent: int = 32, hidden: int = 64, regulariser: float = 1e-3):
        super().__init__()
        self.regulariser = regulariser
        self.net = nn.Sequential(linear_block(n_signal, 8 * hidden), linear_block(8 * hidden, 4 * hidden),
                                 linear_block(4 * hidden, 2 * hidden), linear_block(2 * hidden, hidden))
        self.latent = nn.Linear(hidden, latent)
        nn.init.kaiming_uniform_(self.latent.weight, nonlinearity="linear")
        nn.init.zeros_(self.latent.bias)

    def forward(self, x):
        z = self.latent(self.net(x))
        return z, latent_penalty(z, self.regulariser)


class Decoder(nn.Module):
    """latent L -> h -> 2h -> 4h -> 8h -> signal"""

    def __init__(self, n_signal: int, latent: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(linear_block(latent, hidden), linear_block(hidden, 2 * hidden),
                                 linear_block(2 * hidden, 4 * hidden), linear_block(4 * hidden, 8 * hidden))
        self.out = nn.Linear(8 * hidden, n_signal)
        nn.init.kaiming_uniform_(self.out.weight, nonlinearity="linear")

    def forward(self, z):
        return self.out(self.net(z))


class FunnelAutoencoder(nn.Module):
    """stage 1: signal -> encoder -> z -> decoder -> signal, trained on the signal alone (no parameters)"""

    def __init__(self, n_signal: int, latent: int = 32, hidden: int = 64, regulariser: float = 1e-3):
        super().__init__()
        self.encoder = Encoder(n_signal, latent, hidden, regulariser)
        self.decoder = Decoder(n_signal, latent, hidden)

    def forward(self, x):
        z, penalty = self.encoder(x)
        return self.decoder(z), penalty


class ParameterToDecoder(nn.Module):
    """stage 2: theta -> parameter network h -> 4h -> 4h -> h -> z (L) -> the pretrained decoder -> signal. The encoder is not used any more;
    the decoder keeps training (the default) unless freeze_decoder=True."""

    def __init__(self, decoder: Decoder, n_inputs: int, latent: int = 32, hidden: int = 64, freeze_decoder: bool = False):
        super().__init__()
        self.param_net = nn.Sequential(linear_block_param(n_inputs, hidden, activation=False), linear_block_param(hidden, 4 * hidden),
                                       linear_block_param(4 * hidden, 4 * hidden), linear_block_param(4 * hidden, hidden))
        for m in self.param_net.modules():  # the parameter network is re-initialised Xavier-uniform
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.latent_layer = nn.Linear(hidden, latent)
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(not freeze_decoder)

    def forward(self, theta):
        return self.decoder(self.latent_layer(self.param_net(theta)))


# ------------------------------------------------------------------------------------------------ 1. and 3B. direct network
class FlatDirectNetwork(nn.Module):
    """theta -> SVD coefficients through n_layers hidden layers of one width: Linear -> LayerNorm -> GELU each (without the LayerNorm the
    plain flat stack collapsed to a constant output in our tests)"""

    def __init__(self, n_inputs: int, n_outputs: int, width: int = 256, n_layers: int = 6):
        super().__init__()
        layers = [nn.Linear(n_inputs, width), nn.LayerNorm(width)]
        for _ in range(n_layers - 1):
            layers += [nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)]
        self.body = nn.Sequential(*layers)
        self.act = nn.GELU()
        self.out = nn.Linear(width, n_outputs)

    def forward(self, theta):
        return self.out(self.act(self.body(theta)))


def count_weights(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
