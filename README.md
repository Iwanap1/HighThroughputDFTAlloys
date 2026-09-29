# DFT/ML surface screening starter

Initial calculator-independent structure-generation layer for clean and substitutionally doped fcc metal surfaces.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional: place Materials Project **conventional standard** CIFs at:

```text
data/bulk/Pd.cif
data/bulk/Pt.cif
data/bulk/Ni.cif
data/bulk/Cu.cif
```

If a CIF is absent, `HostSlabGenerator` falls back to ASE's elemental fcc reference lattice so the smoke-test notebook still runs.

## First test

Open:

```text
notebooks/01_test_structure_generation.ipynb
```

It generates all clean host slabs, checks layer/site counts and fixed layers, creates representative Cu-doped Pt(111) slabs for 1, 2, and 4 surface substitutions, plots top/side views, and tests `.traj` save/reload.

## Structure API

- `Slab`: one atomic slab plus metadata and a calculator-agnostic `relax()` method.
- `HostSlabGenerator`: clean fcc host slabs from `HOSTS`, `FACETS`, and `CRYSTAL_SETTINGS`.
- `DopedSlabGenerator`: representative substitutional configurations in the top layer.
- `utils.py`: layer detection, constraints, IDs, and configuration fingerprints.

The current dopant deduplication uses periodic pair-distance fingerprints. This is intentionally lightweight and can later be replaced by rigorous symmetry enumeration (e.g. `icet`) without changing the generator interface.
