#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from _common import ROOT, ensure_dir, write_json


EDGE_NEIGHBOR_VOXEL_OFFSET = (
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
)


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def metrics_for_threshold(coords: torch.Tensor, logits: torch.Tensor, threshold: float) -> dict:
    coords_cpu = coords.to(dtype=torch.long, device="cpu")
    logits_cpu = logits.to(dtype=torch.float32, device="cpu")
    coord_rows = [tuple(map(int, row)) for row in coords_cpu.tolist()]
    coord_to_idx = {coord: idx for idx, coord in enumerate(coord_rows)}

    flags = logits_cpu > threshold
    uf = UnionFind(len(coord_rows))
    positive_edges = int(flags.sum().item())
    valid_quads = 0
    dropped_quads = 0
    quads_by_root: Counter[int] = Counter()

    for idx, coord in enumerate(coord_rows):
        z, y, x = coord
        for axis in range(3):
            if not bool(flags[idx, axis]):
                continue
            quad_indices: list[int] = []
            for dz, dy, dx in EDGE_NEIGHBOR_VOXEL_OFFSET[axis]:
                neighbor = (z + dz, y + dy, x + dx)
                neighbor_idx = coord_to_idx.get(neighbor)
                if neighbor_idx is None:
                    dropped_quads += 1
                    quad_indices = []
                    break
                quad_indices.append(neighbor_idx)
            if not quad_indices:
                continue
            root = quad_indices[0]
            for other in quad_indices[1:]:
                root = uf.union(root, other)
            valid_quads += 1
            quads_by_root[uf.find(root)] += 1

    active_node_roots = [uf.find(idx) for idx in range(len(coord_rows)) if uf.find(idx) in quads_by_root]
    node_sizes = Counter(active_node_roots)
    top_components = []
    for root, node_count in node_sizes.most_common(20):
        top_components.append({
            "root": int(root),
            "nodes": int(node_count),
            "quads": int(quads_by_root.get(root, 0)),
            "node_fraction": float(node_count / max(1, sum(node_sizes.values()))),
            "quad_fraction": float(quads_by_root.get(root, 0) / max(1, valid_quads)),
        })

    return {
        "threshold": threshold,
        "coord_count": int(coords_cpu.shape[0]),
        "positive_edges": positive_edges,
        "positive_edge_fraction": float(positive_edges / max(1, logits_cpu.numel())),
        "valid_quads": int(valid_quads),
        "dropped_positive_edges_missing_neighbors": int(dropped_quads),
        "valid_quad_fraction_of_positive_edges": float(valid_quads / max(1, positive_edges)),
        "active_component_count": int(len(node_sizes)),
        "active_node_count": int(sum(node_sizes.values())),
        "largest_component_nodes": int(top_components[0]["nodes"]) if top_components else 0,
        "largest_component_quads": int(top_components[0]["quads"]) if top_components else 0,
        "largest_component_node_fraction": float(top_components[0]["node_fraction"]) if top_components else 0.0,
        "largest_component_quad_fraction": float(top_components[0]["quad_fraction"]) if top_components else 0.0,
        "top_components": top_components,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pre-mesh FDG edge/quad connectivity metrics from a Pixal3D decoder dump.")
    parser.add_argument("fdg_decoder_pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()

    data = torch.load(resolve(args.fdg_decoder_pt), map_location="cpu", weights_only=False)
    coords = data["coords"].to(torch.long)
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    logits = data["intersected_logits"].float()

    reports = [metrics_for_threshold(coords, logits, threshold) for threshold in args.thresholds]
    out = resolve(args.out)
    ensure_dir(out)
    write_json(out / "fdg_graph_metrics.json", reports)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
