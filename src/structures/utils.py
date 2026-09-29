from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def facet_to_string(facet: Sequence[int]) -> str:
    """Convert a Miller index such as (1, 1, 1) to '111'."""
    return "".join(str(int(v)) for v in facet)


def get_layer_indices(atoms: Atoms, tolerance: float = 0.35) -> list[list[int]]:
    """Return atom indices grouped into z-layers from bottom to top.

    This is intended for slabs whose surface normal is aligned with z.
    The grouping is based on Cartesian z coordinates after sorting.
    """
    if len(atoms) == 0:
        return []

    z = np.asarray(atoms.positions[:, 2], dtype=float)
    order = np.argsort(z)

    layers: list[list[int]] = []
    for idx in order:
        idx = int(idx)
        if not layers:
            layers.append([idx])
            continue

        current_mean = float(np.mean([z[i] for i in layers[-1]]))
        if abs(float(z[idx]) - current_mean) <= tolerance:
            layers[-1].append(idx)
        else:
            layers.append([idx])

    return layers


def get_surface_indices(atoms: Atoms, tolerance: float = 0.35) -> list[int]:
    """Return indices belonging to the top atomic layer."""
    layers = get_layer_indices(atoms, tolerance=tolerance)
    if not layers:
        return []
    return list(layers[-1])


def fixed_bottom_layer_indices(
    atoms: Atoms,
    n_fixed_layers: int,
    tolerance: float = 0.35,
) -> list[int]:
    """Return atom indices in the bottom ``n_fixed_layers`` layers."""
    layers = get_layer_indices(atoms, tolerance=tolerance)
    if n_fixed_layers < 0:
        raise ValueError("n_fixed_layers must be >= 0")
    if n_fixed_layers > len(layers):
        raise ValueError(
            f"Requested {n_fixed_layers} fixed layers but slab only has {len(layers)} layers."
        )

    return [idx for layer in layers[:n_fixed_layers] for idx in layer]


def apply_fixed_bottom_layers(
    atoms: Atoms,
    n_fixed_layers: int,
    tolerance: float = 0.35,
) -> Atoms:
    """Apply an ASE FixAtoms constraint to the bottom layers in-place."""
    fixed = fixed_bottom_layer_indices(
        atoms,
        n_fixed_layers=n_fixed_layers,
        tolerance=tolerance,
    )
    atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def make_slab_id(
    host: str,
    facet: Sequence[int],
    dopant: str | None = None,
    dopant_count: int = 0,
    configuration_id: int = 0,
) -> str:
    """Create a stable human-readable identifier for a slab."""
    base = f"{host}_{facet_to_string(facet)}"
    if dopant is None or dopant_count == 0:
        return f"{base}_clean"
    return f"{base}_{dopant}_n{dopant_count}_cfg{configuration_id:03d}"


def dopant_pair_distance_fingerprint(
    atoms: Atoms,
    indices: Iterable[int],
    decimals: int = 3,
) -> tuple[float, ...]:
    """Fingerprint a surface dopant arrangement by MIC pair distances.

    It is a lightweight way to remove many equivalent arrangements without
    introducing a full symmetry-enumeration dependency. It is not a formal
    crystallographic symmetry proof, but is suitable for initial screening.
    """
    indices = tuple(int(i) for i in indices)
    if len(indices) < 2:
        return ()

    distances: list[float] = []
    for pos, i in enumerate(indices[:-1]):
        for j in indices[pos + 1 :]:
            d = atoms.get_distance(i, j, mic=True)
            distances.append(round(float(d), decimals))

    return tuple(sorted(distances))
