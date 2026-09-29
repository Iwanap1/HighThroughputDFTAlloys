from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass
class RelaxationResult:
    structure_id: str
    atoms: Atoms
    energy_eV: float
    converged: bool
    n_steps: int
    max_force_eV_A: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Slab:
    """One specific clean or doped slab configuration."""

    atoms: Atoms
    slab_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Slab":
        return Slab(
            atoms=self.atoms.copy(),
            slab_id=self.slab_id,
            metadata=dict(self.metadata),
        )

    def write(self, path: str | Path) -> None:
        """Write the atomic structure using ASE's file format inference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        atoms = self.atoms.copy()
        atoms.info.update(self.metadata)
        atoms.info["slab_id"] = self.slab_id
        write(path, atoms)

    @classmethod
    def read(cls, path: str | Path) -> "Slab":
        """Load a slab previously written with :meth:`write`."""
        atoms = read(path)
        info = dict(atoms.info)
        slab_id = info.pop("slab_id", Path(path).stem)
        return cls(atoms=atoms, slab_id=slab_id, metadata=info)

    def relax(self, calculator, settings: dict[str, Any]) -> RelaxationResult:
        """Relax a copy of this slab with any ASE-compatible calculator.

        Supported settings are ``optimizer`` (BFGS/FIRE/LBFGS), ``fmax``,
        ``max_steps`` and ``logfile``. The original slab is not modified.
        """
        from ase.optimize import BFGS, FIRE, LBFGS

        optimizers = {
            "BFGS": BFGS,
            "FIRE": FIRE,
            "LBFGS": LBFGS,
        }

        optimizer_name = str(settings.get("optimizer", "BFGS")).upper()
        if optimizer_name not in optimizers:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. Choose from {sorted(optimizers)}."
            )

        atoms = self.atoms.copy()
        atoms.calc = calculator

        optimizer_cls = optimizers[optimizer_name]
        optimizer = optimizer_cls(atoms, logfile=settings.get("logfile", None))
        converged = optimizer.run(fmax=float(settings["fmax"]), steps=int(settings["max_steps"]))

        forces = np.asarray(atoms.get_forces(), dtype=float)
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0

        return RelaxationResult(
            structure_id=self.slab_id,
            atoms=atoms,
            energy_eV=float(atoms.get_potential_energy()),
            converged=bool(converged),
            n_steps=int(optimizer.nsteps),
            max_force_eV_A=max_force,
            metadata={
                **self.metadata,
                "optimizer": optimizer_name,
                "fmax": float(settings.get("fmax", 0.05)),
                "max_steps": int(settings.get("max_steps", 200)),
            },
        )
