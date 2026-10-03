import itertools
import numpy as np
from collections import defaultdict

N, K, M = 6, 3, 5  # vertices, edge size (3-uniform), number of hyperedges


def incidence_np(n, edges):
    H = np.zeros((n, len(edges)))
    for j, e in enumerate(edges):
        for v in e:
            H[v, j] = 1.0
    return H


def zhou_spectrum(n, edges):
    """Eigenvalues of the normalised hypergraph Laplacian L = I - Dv^-1/2 H De^-1 H^T Dv^-1/2."""
    H = incidence_np(n, edges)
    dv, de = H.sum(1), H.sum(0)
    if (dv == 0).any():
        return None
    Dv, De = np.diag(dv ** -0.5), np.diag(1 / de)
    L = np.eye(n) - Dv @ H @ De @ H.T @ Dv
    return tuple(np.round(np.sort(np.linalg.eigvalsh(L)), 6))


def degree_sequence(n, edges):
    d = [0] * n
    for e in edges:
        for v in e:
            d[v] += 1
    return tuple(sorted(d))


def canonical_form(n, edges):
    """Lexicographically smallest edge set over all vertex relabellings."""
    return min(
        tuple(sorted(tuple(sorted(p[v] for v in e)) for e in edges))
        for p in itertools.permutations(range(n))
    )


def find_cospectral_pair(n=N, k=K, m=M):
    """Exhaustive search for two non-isomorphic k-uniform hypergraphs that agree
    on both the Laplacian spectrum and the vertex degree sequence."""
    candidates = list(itertools.combinations(range(n), k))
    buckets = defaultdict(list)

    for edges in itertools.combinations(candidates, m):
        spectrum = zhou_spectrum(n, edges)
        if spectrum is not None:
            buckets[(spectrum, degree_sequence(n, edges))].append(edges)

    for group in buckets.values():
        if len(group) < 2:
            continue
        representatives = {}
        for edges in group:
            representatives.setdefault(canonical_form(n, edges), edges)
        if len(representatives) > 1:
            return tuple(representatives.values())[:2]
    return None


A, B = find_cospectral_pair()
print("A =", A)
print("B =", B)
print("same degree sequence:", degree_sequence(N, A) == degree_sequence(N, B))
print("max spectral difference:", max(abs(a - b) for a, b in zip(zhou_spectrum(N, A), zhou_spectrum(N, B))))
