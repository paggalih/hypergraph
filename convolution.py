import random
import torch
from torch import nn

torch.manual_seed(0)
random.seed(0)


def incidence(n, edges):
    H = torch.zeros(n, len(edges))
    for j, e in enumerate(edges):
        for v in e:
            H[v, j] = 1.0
    return H


def propagation(H):
    """Dv^-1/2 H De^-1 H^T Dv^-1/2 — one round of vertex to hyperedge to vertex averaging."""
    dv, de = H.sum(1), H.sum(0)
    return torch.diag(dv.pow(-0.5)) @ H @ torch.diag(de.reciprocal()) @ H.t() @ torch.diag(dv.pow(-0.5))


def relabel(edges, permutation):
    return tuple(sorted(tuple(sorted(permutation[v] for v in e)) for e in edges))


def features(H, kind):
    degree = H.sum(1, keepdim=True)
    if kind == "ones":
        return torch.ones(H.shape[0], 1)
    if kind == "degree":
        return degree
    if kind == "higher":
        shared = H @ H.t()                                  # how many hyperedges each vertex pair shares
        shared.fill_diagonal_(0)
        strongly_tied = (shared == 2).float().sum(1, keepdim=True)
        return torch.cat([degree, shared.sum(1, keepdim=True), strongly_tied], dim=1)
    raise ValueError(kind)


class HGNN(nn.Module):
    """Feng et al. style hypergraph convolution, mean readout, two-class head."""

    def __init__(self, d_in, d_hidden=32, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(d_in if i == 0 else d_hidden, d_hidden) for i in range(n_layers)
        )
        self.head = nn.Linear(d_hidden, 2)

    def forward(self, x, p):
        for layer in self.layers:
            x = torch.relu(p @ layer(x))
        return self.head(x.mean(0))


# ==============================================================================

def draw(edges, kind):
    permutation = list(range(N))
    random.shuffle(permutation)
    H = incidence(N, relabel(edges, permutation))
    return features(H, kind), propagation(H)


def experiment(kind, steps=400, n_test=200):
    model = HGNN(features(incidence(N, A), kind).shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        loss = 0.0
        for label, edges in ((0, A), (1, B)):
            x, p = draw(edges, kind)
            loss = loss + criterion(model(x, p).unsqueeze(0), torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    with torch.no_grad():
        for _ in range(n_test):
            for label, edges in ((0, A), (1, B)):
                x, p = draw(edges, kind)
                correct += int(model(x, p).argmax().item() == label)
    return correct / (2 * n_test)


for kind in ("ones", "degree", "higher"):
    print(f"{kind:>7}: test accuracy {experiment(kind):.3f}")
