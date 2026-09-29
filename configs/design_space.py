HOSTS = ["Pd", "Pt", "Ni", "Cu"]
DOPANTS = ["Pd", "Pt", "Ni", "Cu", "Ag", "Au", "Co", "Rh", "Ir", "Ru"]
FACETS = [(1, 1, 1)]
DOPANT_COUNTS = [0, 1, 2, 4]
CRYSTAL_SETTINGS = {
    "slab": {
        "supercell": (4, 4),
        "n_layers": 4,
        "fixed_layers": 2,
        "vacuum_A": 15.0 
    },
    "max_configurations_per_composition": 5
}

ADSORBATES = {
    "CO": ["ontop", "bridge", "fcc", "hcp"],
    "O": ["ontop", "bridge", "fcc", "hcp"]
}

SPIN_POLARIZED = True

MAGMOMS = {
    "Co": 2.0,
    "Ni": 1.0,

    "Pd": 0.0,
    "Pt": 0.0,
    "Cu": 0.0,
    "Ag": 0.0,
    "Au": 0.0,
    "Rh": 0.0,
    "Ir": 0.0,
    "Ru": 0.0,
}