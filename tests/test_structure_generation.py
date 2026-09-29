from configs.design_space import CRYSTAL_SETTINGS
from src.structures import DopedSlabGenerator, HostSlabGenerator
from src.structures.utils import get_layer_indices, get_surface_indices


def test_pt111_geometry():
    generator = HostSlabGenerator(
        hosts=["Pt"],
        facets=[(1, 1, 1)],
        crystal_settings=CRYSTAL_SETTINGS,
    )
    slab = generator.generate()[0]

    assert len(slab.atoms) == 64
    assert len(get_layer_indices(slab.atoms)) == 4
    assert len(get_surface_indices(slab.atoms)) == 16


def test_surface_doping():
    host = HostSlabGenerator(
        hosts=["Pt"],
        facets=[(1, 1, 1)],
        crystal_settings=CRYSTAL_SETTINGS,
    ).generate()[0]

    generator = DopedSlabGenerator(CRYSTAL_SETTINGS)
    slabs = generator.generate(host, dopant="Cu", dopant_count=2)

    assert 1 <= len(slabs) <= CRYSTAL_SETTINGS["max_configurations_per_composition"]
    for slab in slabs:
        assert slab.atoms.get_chemical_symbols().count("Cu") == 2
        assert slab.metadata["dopant_fraction_surface"] == 2 / 16
