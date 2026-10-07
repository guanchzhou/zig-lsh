"""Independent reference outputs for zig-lsh, written to test/golden.json.

NumPy only; no code is shared with the Zig implementation. Each rotation is materialised as a dense
matrix (Hadamard matrices and sign diagonals multiplied out) and checked to be orthogonal before use,
so keys, probes and candidate sets come from matrix products rather than an in-place transform.

Run: uv run --with numpy python test/reference.py
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "golden.json"
MASK = (1 << 64) - 1
RNG = np.random.default_rng(20261007)


def splitmix64(seed: int, i: int) -> int:
    z = (seed + ((i + 1) * 0x9E3779B97F4A7C15)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def padded(d: int) -> int:
    n = 2
    while n < d:
        n *= 2
    return n


_cache: dict = {}


def rotation(seed: int, n: int) -> np.ndarray:
    if (seed, n) in _cache:
        return _cache[(seed, n)]
    H = hadamard(n) / np.sqrt(n)
    R = np.eye(n)
    for r in range(3):
        D = np.diag([-1.0 if splitmix64(seed, r * n + i) & 1 else 1.0 for i in range(n)])
        R = H @ D @ R
    assert np.allclose(R.T @ R, np.eye(n), atol=1e-12)
    _cache[(seed, n)] = R
    return R


def rot_seed(seed: int, k: int, table: int, j: int) -> int:
    return splitmix64(seed, table * k + j)


def rotated(x: np.ndarray, seed: int, k: int, table: int, j: int) -> np.ndarray:
    n = padded(len(x))
    v = np.zeros(n)
    v[: len(x)] = x
    return rotation(rot_seed(seed, k, table, j), n) @ v


def compose(hashes: list[int], n: int) -> int:
    key, mul = 0, 1
    for h in hashes:
        key = (key + h * mul) & MASK
        mul = (mul * 2 * n) & MASK
    return key


def mag(v: float, unit: float) -> float:
    return float(np.floor(abs(v) / unit + 0.5))


def unit_of(y: np.ndarray) -> float:
    s = float(np.linalg.norm(y))
    return 1.0 if s == 0 else 1e-9 * s


def top_index(y: np.ndarray) -> int:
    u = unit_of(y)
    m = [mag(v, u) for v in y]
    return max(range(len(y)), key=lambda i: (m[i], -i))


def keys_and_probes(x: np.ndarray, family: str, k: int, seed: int, table: int, probes: int) -> list[int]:
    n = padded(len(x))
    if family == "hyperplane":
        y = rotated(x, seed, k, table, 0)
        own = sum(1 << j for j in range(k) if y[j] > 1e-12 * np.linalg.norm(y))
        u = unit_of(y)
        alts = sorted((mag(y[j], u), j) for j in range(k))[:probes]
        return [own] + [own ^ (1 << j) for _, j in alts]
    ys = [rotated(x, seed, k, table, j) for j in range(k)]
    tops = [top_index(y) for y in ys]
    hashes = [2 * t + (1 if y[t] < 0 else 0) for y, t in zip(ys, tops)]
    own = compose(hashes, n)
    alts = []
    for j, (y, t) in enumerate(zip(ys, tops)):
        u = unit_of(y)
        others = sorted((i for i in range(n) if i != t), key=lambda i: (-mag(y[i], u), i))[:probes]
        for r, i in enumerate(others):
            alts.append((mag(y[t], u) - mag(y[i], u), j, r, i))
    alts.sort()
    out = [own]
    for _, j, _, i in alts[:probes]:
        h = list(hashes)
        h[j] = 2 * i + (1 if ys[j][i] < 0 else 0)
        out.append(compose(h, n))
    return out


def case(d: int, family: str, k: int, tables: int, probes: int) -> dict:
    n_vec = 12
    data = RNG.standard_normal((n_vec, d)).astype(np.float32)
    queries = (data[:3] + 0.3 * RNG.standard_normal((3, d))).astype(np.float32)
    seed = int(RNG.integers(0, 2**63)) * 2 + int(RNG.integers(0, 2))
    keys = [[keys_and_probes(x.astype(np.float64), family, k, seed, t, 0)[0] for t in range(tables)] for x in data]
    qprobes = [[keys_and_probes(q.astype(np.float64), family, k, seed, t, probes) for t in range(tables)] for q in queries]
    cands = []
    for qp in qprobes:
        s = set()
        for t in range(tables):
            for key in qp[t]:
                s |= {i for i in range(n_vec) if keys[i][t] == key}
        cands.append(sorted(s))
    hexes = lambda v: f"{v:016x}"  # noqa: E731
    return {"d": d, "family": family, "k": k, "tables": tables, "probes": probes, "seed": hexes(seed),
            "data": data.tolist(), "queries": queries.tolist(),
            "keys": [[hexes(v) for v in row] for row in keys],
            "query_probes": [[[hexes(v) for v in tp] for tp in qp] for qp in qprobes],
            "candidates": cands}


def main() -> None:
    cases = []
    for d in (3, 5, 8, 13, 16, 31, 33, 64):
        n = padded(d)
        for family, ks in (("hyperplane", (1, min(4, n), min(8, n))), ("cross-polytope", (1, 2, 3))):
            for k in ks:
                for tables, probes in ((1, 0), (3, 3)):
                    cases.append(case(d, family, k, tables, probes))
    OUT.write_text(json.dumps({"cases": cases}) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
