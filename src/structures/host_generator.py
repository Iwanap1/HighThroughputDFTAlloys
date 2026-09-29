from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc100, fcc110, fcc111
from ase.io import read

from .slab import Slab
from .utils import apply_fixed_bottom_layers, get_layer_indices, make_slab_id


class HostSlabGenerator:
    """Generate clean fcc host slabs from the design-space settings.

    If ``<bulk_dir>/<element>.cif`` exists, its cubic lattice constant is used.
    Otherwise ASE's built-in elemental fcc reference is used as a fallback.
    """

    def __init__(
        self,
        hosts: Iterable[str],
        facets: Iterable[Sequence[int]],
        crystal_settings: dict,
        bulk_dir: str | Path | None = None,
        layer_tolerance: float = 0.35,
    ):
        self.hosts = list(hosts)
        self.facets = [tuple(int(x) for x in facet) for facet in facets]
        self.crystal_settings = crystal_settings
        self.bulk_dir = Path(bulk_dir) if bulk_dir is not None else None
        self.layer_tolerance = float(layer_tolerance)

    def _get_lattice_constant(self, host: str) -> tuple[float, str]:
        if self.bulk_dir is not None:
            cif_path = self.bulk_dir / f"{host}.cif"
            if cif_path.exists():
                atoms = read(cif_path)
                lengths = np.asarray(atoms.cell.lengths(), dtype=float)
                angles = np.asarray(atoms.cell.angles(), dtype=float)

                if not np.allclose(lengths, lengths[0], rtol=0.02, atol=0.02):
                    raise ValueError(
                        f"{cif_path} is not approximately cubic; cell lengths are {lengths}. "
                        "Use a conventional cubic fcc CIF."
                    )
                if not np.allclose(angles, 90.0, atol=1.0):
                    raise ValueError(
                        f"{cif_path} is not approximately cubic; cell angles are {angles}. "
                        "Use a conventional cubic fcc CIF."
                    )

                return float(lengths.mean()), str(cif_path)

        reference = bulk(host, crystalstructure="fcc", cubic=True)
        return float(reference.cell.lengths()[0]), "ase_reference"

    @staticmethod
    def _build_fcc_surface(
        host: str,
        facet: tuple[int, int, int],
        size: tuple[int, int, int],
        lattice_constant: float,
        vacuum_A: float,
    ) -> Atoms:
        builders = {
            (1, 1, 1): fcc111,
            (1, 0, 0): fcc100,
            (1, 1, 0): fcc110,
        }
        if facet not in builders:
            raise NotImplementedError(
                f"Facet {facet} is not yet supported by HostSlabGenerator. "
                "Current fcc builders support (111), (100), and (110)."
            )

        atoms = builders[facet](
            host,
            size=size,
            a=lattice_constant,
            vacuum=vacuum_A,
        )
        # Plane-wave DFT and many periodic MLIPs expect a periodic z cell;
        # the vacuum separates periodic slab images.
        atoms.set_pbc((True, True, True))
        return atoms

    def generate(self) -> list[Slab]:
        settings = self.crystal_settings["slab"]
        nx, ny = tuple(settings["supercell"])
        n_layers = int(settings["n_layers"])
        n_fixed = int(settings["fixed_layers"])
        vacuum_A = float(settings["vacuum_A"])

        slabs: list[Slab] = []

        for host in self.hosts:
            lattice_constant, structure_source = self._get_lattice_constant(host)

            for facet in self.facets:
                atoms = self._build_fcc_surface(
                    host=host,
                    facet=facet,
                    size=(int(nx), int(ny), n_layers),
                    lattice_constant=lattice_constant,
                    vacuum_A=vacuum_A,
                )

                apply_fixed_bottom_layers(
                    atoms,
                    n_fixed_layers=n_fixed,
                    tolerance=self.layer_tolerance,
                )

                layers = get_layer_indices(atoms, tolerance=self.layer_tolerance)
                if len(layers) != n_layers:
                    raise RuntimeError(
                        f"Expected {n_layers} layers for {host}{facet}, found {len(layers)}."
                    )

                slab_id = make_slab_id(host=host, facet=facet)
                metadata = {
                    "host": host,
                    "facet": facet,
                    "dopant": None,
                    "dopant_count": 0,
                    "dopant_fraction_surface": 0.0,
                    "configuration_id": 0,
                    "lattice_constant_A": lattice_constant,
                    "structure_source": structure_source,
                    "supercell": (int(nx), int(ny)),
                    "n_layers": n_layers,
                    "fixed_layers": n_fixed,
                    "vacuum_A": vacuum_A,
                }
                slabs.append(Slab(atoms=atoms, slab_id=slab_id, metadata=metadata))

        return slabs
