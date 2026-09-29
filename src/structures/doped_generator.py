from __future__ import annotations

from itertools import combinations

import numpy as np

from .slab import Slab
from .utils import dopant_pair_distance_fingerprint, get_surface_indices, make_slab_id

class DopedSlabGenerator:
    """Generate representative top-layer substitutional dopant configurations."""

    def __init__(
        self,
        crystal_settings: dict,
        layer_tolerance: float = 0.35,
        fingerprint_decimals: int = 3,
    ):
        self.crystal_settings = crystal_settings
        self.layer_tolerance = float(layer_tolerance)
        self.fingerprint_decimals = int(fingerprint_decimals)

    def _unique_configurations(
        self,
        slab: Slab,
        dopant_count: int,
    ) -> list[tuple[int, ...]]:
        surface_indices = get_surface_indices(
            slab.atoms,
            tolerance=self.layer_tolerance,
        )

        if dopant_count < 1:
            return [tuple()]
        if dopant_count > len(surface_indices):
            raise ValueError(
                f"Cannot place {dopant_count} dopants on only {len(surface_indices)} surface sites."
            )

        if dopant_count == 1:
            # All sites are symmetry-equivalent on an ideal clean fcc surface.
            return [(surface_indices[0],)]

        unique: dict[tuple[float, ...], tuple[int, ...]] = {}
        for combo in combinations(surface_indices, dopant_count):
            fingerprint = dopant_pair_distance_fingerprint(
                slab.atoms,
                combo,
                decimals=self.fingerprint_decimals,
            )
            unique.setdefault(fingerprint, tuple(combo))

        configs = list(unique.values())
        configs.sort(key=lambda c: self._configuration_sort_key(slab, c))
        return configs

    @staticmethod
    def _configuration_sort_key(slab: Slab, config: tuple[int, ...]):
        if len(config) < 2:
            return (0.0, 0.0, config)

        distances = []
        for pos, i in enumerate(config[:-1]):
            for j in config[pos + 1 :]:
                distances.append(slab.atoms.get_distance(i, j, mic=True))

        return (float(np.mean(distances)), float(np.std(distances)), config)

    @staticmethod
    def _select_diverse(
        configs: list[tuple[int, ...]],
        max_configurations: int,
    ) -> list[tuple[int, ...]]:
        if len(configs) <= max_configurations:
            return configs
        if max_configurations <= 0:
            return []

        # Configurations are ordered from clustered to dispersed. Evenly spaced
        # picks retain both extremes plus representative intermediates.
        positions = np.linspace(0, len(configs) - 1, max_configurations)
        selected_indices = []
        for p in positions:
            idx = int(round(float(p)))
            if idx not in selected_indices:
                selected_indices.append(idx)

        # Rounding can very rarely reduce the count by one; fill deterministically.
        if len(selected_indices) < max_configurations:
            for idx in range(len(configs)):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) == max_configurations:
                    break

        return [configs[i] for i in selected_indices]

    def generate(
        self,
        host_slab: Slab,
        dopant: str,
        dopant_count: int,
        max_configurations: int | None = None,
    ) -> list[Slab]:
        host = str(host_slab.metadata["host"])
        facet = tuple(host_slab.metadata["facet"])

        if dopant == host:
            return []
        if dopant_count == 0:
            return [host_slab.copy()]

        if max_configurations is None:
            max_configurations = int(
                self.crystal_settings.get("max_configurations_per_composition", 5)
            )

        configs = self._unique_configurations(host_slab, dopant_count)
        configs = self._select_diverse(configs, int(max_configurations))

        surface_site_count = len(
            get_surface_indices(host_slab.atoms, tolerance=self.layer_tolerance)
        )

        slabs: list[Slab] = []
        for configuration_id, dopant_indices in enumerate(configs):
            atoms = host_slab.atoms.copy()
            symbols = atoms.get_chemical_symbols()
            for idx in dopant_indices:
                symbols[idx] = dopant
            atoms.set_chemical_symbols(symbols)

            slab_id = make_slab_id(
                host=host,
                facet=facet,
                dopant=dopant,
                dopant_count=dopant_count,
                configuration_id=configuration_id,
            )

            metadata = {
                **host_slab.metadata,
                "dopant": dopant,
                "dopant_count": int(dopant_count),
                "dopant_fraction_surface": float(dopant_count / surface_site_count),
                "dopant_indices": tuple(int(i) for i in dopant_indices),
                "configuration_id": int(configuration_id),
            }
            slabs.append(Slab(atoms=atoms, slab_id=slab_id, metadata=metadata))

        return slabs
