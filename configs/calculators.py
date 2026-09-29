from mace.calculators import mace_mp

def get_mace_calculator(device="cpu"):
    return mace_mp(
        model="mh-1",
        head="oc20_usemppbe",
        device=device,
        default_dtype="float64",
    )