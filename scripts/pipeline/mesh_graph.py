#!/usr/bin/env python3
"""
scripts/pipeline/mesh_graph.py

Shared mesh-graph utilities for the round-4 FIX ROUND 4 skin-weight
rebuild (rebuild_face_weights.py, rebuild_arm_body_weights.py,
weld_seam_weights.py). Pure geometry/graph-theory helpers, no glTF I/O of
its own (that stays in gltf_anim.py) and no skin-weight logic (that stays
in the two rebuild_*.py scripts) -- kept here only because all three
post-process scripts need the SAME mesh-adjacency graph and the SAME
weld-group detection, and duplicating either would risk them silently
drifting apart.

  * build_adjacency(positions, indices) -- undirected vertex-adjacency
    graph from triangle indices, edge weight = euclidean edge length
    (used as the graph-Dijkstra metric, i.e. true geodesic-ish distance
    over the mesh surface rather than straight-line euclidean distance,
    which is what makes it possible to tell a cheek tuft (geodesically
    close to the face) from a shoulder cap (geodesically close to the
    torso/arm) even when the two sit close together in 3D space, e.g.
    around the ears).
  * multi_source_dijkstra(adj, seeds) -- standard multi-source shortest
    path; ties among seeds resolve to distance 0 at every seed.
  * laplacian_smooth_field(values, active_idxs, adjacency, iters, alpha,
    fixed=None) -- iterative neighbor-average smoothing of a scalar field
    over a restricted vertex set, with an optional set of indices whose
    value is a fixed Dirichlet boundary condition (used so smoothing a
    band does not also drag in vertices well outside it: the band's own
    outer rim is fixed at its original value, and only the interior of
    the band is free to move toward the local neighbor average).
  * find_weld_groups(positions, precision=1e-5) -- vertices co-located to
    within `precision` (UV/normal seam duplicates, i.e. same 3D point,
    different vertex index because they carry different TEXCOORD_0/
    NORMAL) grouped together.
"""
import heapq
from collections import defaultdict

import numpy as np


def build_adjacency(positions, indices):
    """indices: (M,3) int array of triangle vertex indices.
    Returns a list (len == len(positions)) of dict neighbor_idx -> edge_length."""
    n = positions.shape[0]
    adj = [dict() for _ in range(n)]
    tris = indices.reshape(-1, 3)
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            d = float(np.linalg.norm(positions[a] - positions[b]))
            cur = adj[a].get(b)
            if cur is None or d < cur:
                adj[a][b] = d
            cur = adj[b].get(a)
            if cur is None or d < cur:
                adj[b][a] = d
    return adj


def multi_source_dijkstra(adj, seeds):
    """Shortest-path distance from the nearest of `seeds` to every vertex,
    over the graph `adj` (list of {neighbor: weight})."""
    n = len(adj)
    dist = np.full(n, np.inf)
    pq = []
    for s in seeds:
        s = int(s)
        if dist[s] > 0:
            dist[s] = 0.0
            heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def laplacian_smooth_field(values, active_idxs, adjacency, iters, alpha, fixed=None, tol=None,
                            uniform_weights=True):
    """Laplacian-smooth a scalar field `values` (1-D array, indexed by
    vertex) over `adjacency`, updating only `active_idxs` each iteration
    (neighbors outside `active_idxs` still contribute their CURRENT value
    to the average, they just never get updated themselves unless they
    are also in `active_idxs`). `fixed`, if given, is a set of vertex
    indices within `active_idxs` whose value is clamped back to its
    starting value after every iteration (a Dirichlet boundary ring at
    the edge of the smoothing band, so the band blends into its
    surroundings rather than drifting the whole region toward a flat
    mean).

    `iters` is a HARD CAP; if `tol` is given the iteration stops early
    once the largest per-vertex change in one pass drops below `tol`
    (heat-diffusion convergence to a steady state), so the smoothing
    isn't tied to an arbitrary fixed count that may under- or
    over-smooth depending on how many mesh-edge hops wide a given band
    happens to be.

    `uniform_weights` (default True): each neighbor contributes equally
    to the average, regardless of edge length. An inverse-edge-length
    weighting (closer neighbor = louder vote) is the more common choice
    for smoothing a field that is expected to vary smoothly in SPACE, but
    this pipeline's whole problem is edges that are geometrically SHORT
    yet carry a large skinning-weight jump across them (that is what
    turns into a large edge-length-RATIO strain under animation) -- an
    inverse-length weighting actively discounts exactly the longer edges
    where a real mismatch most needs pulling down, and was measured
    directly on this asset to leave a sternum-area vertex at >50% arm
    weight one 5.7%-h-long edge away from a 0%-arm neighbor because that
    neighbor's vote was discounted for being "far." Equal-vote averaging
    treats every mesh edge as an equally important constraint instead."""
    cur = values.copy()
    fixed = fixed or set()
    active = list(active_idxs)
    active_arr = np.array(active, dtype=np.int64)
    start_vals = {int(i): float(values[i]) for i in fixed}
    for _ in range(iters):
        nxt = cur.copy()
        max_change = 0.0
        for v in active:
            neigh = adjacency[v]
            if not neigh:
                continue
            acc = 0.0
            wsum = 0.0
            for n_idx, elen in neigh.items():
                w = 1.0 if uniform_weights else (1.0 / max(elen, 1e-9))
                acc += w * cur[n_idx]
                wsum += w
            if wsum > 0:
                avg = acc / wsum
                nxt[v] = (1 - alpha) * cur[v] + alpha * avg
                max_change = max(max_change, abs(nxt[v] - cur[v]))
        for i, val in start_vals.items():
            nxt[i] = val
        cur = nxt
        if tol is not None and max_change < tol:
            break
    return cur


def augment_adjacency_with_weld_edges(adj, weld_groups, weight=1e-9):
    """Add near-zero-weight edges between every pair of vertices in each
    weld group (see find_weld_groups) directly into `adj` IN PLACE.

    Triangle-index adjacency alone badly FRAGMENTS this mesh: a UV/normal
    seam means the two sides of the same physical edge are two DIFFERENT
    vertex indices (same 3D position, disjoint triangle fans), so
    build_adjacency's graph has no edge between them at all -- measured
    on this asset, a plain triangle-adjacency BFS/Dijkstra from a single
    vertex reaches only ~89 of 13,186 vertices; the mesh is otherwise cut
    into thousands of texture-island-sized fragments. Since every
    seam-duplicate group is, physically, ONE point on the surface, this
    connects them with an (almost) zero-length edge so a geodesic walk
    can cross a texture-chart boundary the way it would on the real
    surface -- this is what makes multi_source_dijkstra actually
    single-connected-component over the whole mesh, which the
    face/arm-body geodesic classification (rebuild_face_weights.py,
    rebuild_arm_body_weights.py) depends on."""
    for group in weld_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = int(group[i]), int(group[j])
                cur = adj[a].get(b)
                if cur is None or weight < cur:
                    adj[a][b] = weight
                cur = adj[b].get(a)
                if cur is None or weight < cur:
                    adj[b][a] = weight


def find_weld_groups(positions, precision=1e-5):
    """Group vertex indices that are co-located within `precision` (grid
    snap, so it is robust to which member of a pair is 'first'). Returns
    a list of lists of vertex indices, ONLY for groups with >= 2 members
    (singletons are not co-located with anything and need no welding)."""
    groups = defaultdict(list)
    inv = 1.0 / precision
    for i, p in enumerate(positions):
        key = (round(p[0] * inv), round(p[1] * inv), round(p[2] * inv))
        groups[key].append(i)
    return [v for v in groups.values() if len(v) > 1]


def pack_top4(weight_dict, eps=1e-9):
    """dict{joint_slot:int -> weight:float} (any size, weights may be
    zero/negative-noise) -> (joints[4] list[int], weights[4] list[float]),
    renormalized to sum 1, largest 4 entries kept (overflow folded
    proportionally into the kept four so no influence is silently
    dropped -- the same rule fix_arm_shoulder_weights.py used, kept here
    as the ONE shared implementation every round-4 rebuild script calls,
    so "cap at 4 influences, renormalize" behaves identically everywhere
    a vertex's weights are rebuilt)."""
    items = [(j, w) for j, w in weight_dict.items() if w > eps]
    if not items:
        return [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]
    items.sort(key=lambda kv: -kv[1])
    kept = items[:4]
    dropped = items[4:]
    if dropped:
        dropped_sum = sum(w for _, w in dropped)
        kept_sum = sum(w for _, w in kept)
        if kept_sum > eps:
            kept = [(j, w + dropped_sum * (w / kept_sum)) for j, w in kept]
    total = sum(w for _, w in kept)
    joints = [0, 0, 0, 0]
    weights = [0.0, 0.0, 0.0, 0.0]
    for idx, (j, w) in enumerate(kept):
        joints[idx] = int(j)
        weights[idx] = float(w / total)
    return joints, weights
