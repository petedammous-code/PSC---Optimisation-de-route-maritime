"""Dijkstra, visibilite et lissage du chemin."""

import numpy as np
from skimage.graph import MCP_Geometric
from .grille import Grid


def snap_to_navigable(nav: np.ndarray, rc: tuple[int, int]) -> tuple[int, int]:
    """Cellule navigable la plus proche si le port tombe sur terre ou dans la marge."""
    r, c = rc
    if nav[r, c]:
        return r, c
    rows, cols = np.nonzero(nav)
    k = np.argmin((rows - r) ** 2 + (cols - c) ** 2)
    return int(rows[k]), int(cols[k])


def grid_shortest_path(grid: Grid, nav: np.ndarray, start_rc, end_rc, fully_connected=True, speed_kn=None) -> np.ndarray:
    """Dijkstra ; huit voisins par defaut, quatre pour le repli conservateur."""
    costs = np.where(nav, grid.cell_nm[:, None] if speed_kn is None else grid.cell_nm[:, None]/speed_kn, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=fully_connected)
    cumul, _ = mcp.find_costs([start_rc], [end_rc])
    if not np.isfinite(cumul[end_rc]):
        raise RuntimeError("Aucun chemin maritime entre les deux points (grille trop grossiere ou marge trop grande)")
    return np.array(mcp.traceback(end_rc))


def line_of_sight(nav: np.ndarray, p: np.ndarray, q: np.ndarray) -> bool:
    """Teste le segment par echantillonnage a un quart de cellule et quatre voisins."""
    n = int(np.ceil(4 * np.max(np.abs(q - p)))) + 1
    t = np.linspace(0.0, 1.0, n)
    rr = p[0] + t * (q[0] - p[0])
    cc = p[1] + t * (q[1] - p[1])
    r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    r1, c1 = np.ceil(rr).astype(int), np.ceil(cc).astype(int)
    return bool(nav[r0, c0].all() and nav[r0, c1].all() and nav[r1, c0].all() and nav[r1, c1].all())


def string_pull(nav: np.ndarray, path_rc: np.ndarray, seg_cost=None) -> list[int]:
    """Reduit le chemin en escalier ; renvoie les indices des waypoints gardes."""
    path = path_rc.astype(float)
    prefix = None if seg_cost is None else np.r_[0., np.cumsum([seg_cost(a,b) for a,b in zip(path[:-1],path[1:])])]
    keep = [0]
    i = 0
    n = len(path)
    while i < n - 1:
        j = i + 1
        while j + 1 < n and line_of_sight(nav, path[i], path[j + 1]) and (prefix is None or seg_cost(path[i],path[j+1]) <= prefix[j+1]-prefix[i]+1e-9):
            j += 1
        keep.append(j)
        i = j
    return keep


def relax_waypoints(nav: np.ndarray, path_rc: np.ndarray, idx: list[int], seg_len, n_iter: int = 3, respect_cost=False) -> list[int]:
    """Optimisation locale apres le lissage glouton, comme dans le PDF."""
    path = path_rc.astype(float)
    idx = list(idx)
    for _ in range(n_iter):
        changed = False
        k = 1
        while k < len(idx) - 1:
            a, b = path[idx[k - 1]], path[idx[k + 1]]
            if line_of_sight(nav, a, b) and (not respect_cost or seg_len(a,b) <= seg_len(a,path[idx[k]])+seg_len(path[idx[k]],b)+1e-9):
                del idx[k]
                changed = True
                continue
            best_x, best_d = idx[k], seg_len(a, path[idx[k]]) + seg_len(path[idx[k]], b)
            step = max(1, (idx[k + 1] - idx[k - 1]) // 200)
            for x in range(idx[k - 1] + 1, idx[k + 1], step):
                d = seg_len(a, path[x]) + seg_len(path[x], b)
                if d < best_d - 1e-9 and line_of_sight(nav, a, path[x]) and line_of_sight(nav, path[x], b):
                    best_x, best_d = x, d
            if best_x != idx[k]:
                idx[k] = best_x
                changed = True
            k += 1
        if not changed:
            break
    return idx
