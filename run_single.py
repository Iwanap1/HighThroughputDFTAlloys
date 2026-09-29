from pathlib import Path

from configs.calculators import get_mace_calculator
from configs.calculation_settings import RELAXATION_SETTINGS
from src.structures.slab import Slab


initial_path = Path(
    "data/slabs/initial/Pt_111_Cu_n4_cfg000.traj"
)

slab = Slab.read(initial_path)

calculator = get_mace_calculator(device="cpu")

result = slab.relax(
    calculator=calculator,
    settings=RELAXATION_SETTINGS,
)

print("Converged:", result.converged)
print("Steps:", result.n_steps)
print("Energy:", result.energy_eV, "eV")
print("Max force:", result.max_force_eV_A, "eV/A")

relaxed_slab = Slab(
    atoms=result.atoms,
    slab_id=slab.slab_id,
    metadata={
        **slab.metadata,
        "energy_eV": result.energy_eV,
        "converged": result.converged,
        "n_steps": result.n_steps,
        "max_force_eV_A": result.max_force_eV_A,
        "calculator": "MACE-MH-1",
        "calculator_head": "oc20_usemppbe",
    },
)

relaxed_path = Path(
    "data/slabs/relaxed"
) / f"{slab.slab_id}.traj"

relaxed_slab.write(relaxed_path)