import torch
from torch import nn

# TODO: think of a better name
class Blurrer(nn.Module):
    def __init__(self, L=10) -> None:
        super().__init__()

        self.L = L

        self.in_dim = 2*L*3 + 4 + 3 + 2*L*3
        self.out_dim = 4 + 3

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_dim),
            nn.ReLU()
        )

        for name, param in self.layers.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))

        # Set up embeddings functions
        freqs = torch.pi * 2.**torch.arange(self.L, requires_grad=False, dtype=torch.float32)
        self.emb_fns = []
        for freq in freqs:
            for p_fn in [torch.sin, torch.cos]:
                self.emb_fns.append( lambda x,p_fn=p_fn,freq=freq: p_fn(x*freq) )

    @torch.no_grad()
    def embed(self, vec):
        """
        Positional embeddings

        Args:
        - `vec:Tensor` of shape `N,3`

        Returns:
        - `embeddings:Tensor` of shape `N,3*2*L`
        """
        return torch.concat([fn(vec) for fn in self.emb_fns], dim=-1)

    def forward(self, means, quats, scales, viewdirs):
        # Embed position/direction, shape `N,3*2*L`
        means_emb = self.embed(means)
        viewdirs_emb = self.embed(viewdirs)

        # Stack features, shape `N, 3*2*L + 4 + 3`
        in_features = torch.hstack( (means_emb, quats, scales, viewdirs_emb) )

        # Get MLP output, shape `N, 4 + 3`, make sure no grads flow back using `detach`
        out_features = self.layers(in_features.detach())

        # Make sure blurring is only done with residuals >=1
        out_features = out_features.clip(min=1.)

        # Split output into residuals
        res_quats, res_scales = torch.split(out_features, split_size_or_sections=[4,3], dim=-1)
        return res_quats, res_scales

if __name__ == '__main__':
    L=10
    N=1000
    model = Blurrer(L=L)

    means = torch.randn(N,3)
    quats = nn.functional.normalize( torch.randn(N,4) )
    scales = torch.exp( torch.randn(N, 3) )
    viewdirs = means - torch.randn(1,3)

    means_emb = model.embed(means)
    assert means_emb.shape == (N, 3*2*L)

    viewdirs_emb = model.embed(viewdirs)
    assert viewdirs_emb.shape == (N, 3*2*L)

    res_quats, res_scales = model(means, quats, scales, viewdirs)
    assert res_quats.shape == (N, 4)
    assert (res_quats>0).all()
    assert res_scales.shape == (N, 3)
    assert (res_scales>0).all()
