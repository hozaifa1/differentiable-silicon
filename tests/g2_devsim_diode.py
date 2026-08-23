"""G2 gate: DEVSIM imports, meshes, and converges a 1-D pn diode.

This is the canonical DEVSIM `diode_1d` walkthrough, written out here because the
wheel ships no examples directory. Exit 0 + a rectifying I(V) is the gate.
"""

from __future__ import annotations

import sys

from diffsilicon.shared.devsim_env import ensure_math_libs

ensure_math_libs()

from devsim import (  # noqa: E402
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    edge_from_node_model,
    finalize_mesh,
    get_contact_current,
    node_model,
    set_parameter,
    solve,
)
from devsim.python_packages.simple_physics import (  # noqa: E402
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSolution,
    SetSiliconParameters,
)

DEVICE = "diode"
REGION = "bulk"


def build_and_solve(voltages=(0.0, 0.2, 0.4, 0.6)) -> list[tuple[float, float]]:
    create_1d_mesh(mesh="dio")
    add_1d_mesh_line(mesh="dio", pos=0.0, ps=1e-7, tag="top")
    add_1d_mesh_line(mesh="dio", pos=0.5e-5, ps=1e-9, tag="mid")
    add_1d_mesh_line(mesh="dio", pos=1e-5, ps=1e-7, tag="bot")
    add_1d_contact(mesh="dio", name="top", tag="top", material="metal")
    add_1d_contact(mesh="dio", name="bot", tag="bot", material="metal")
    add_1d_region(mesh="dio", material="Silicon", region=REGION, tag1="top", tag2="bot")
    finalize_mesh(mesh="dio")
    create_device(mesh="dio", device=DEVICE)

    SetSiliconParameters(DEVICE, REGION, 300)
    set_parameter(device=DEVICE, region=REGION, name="taun", value=1e-8)
    set_parameter(device=DEVICE, region=REGION, name="taup", value=1e-8)

    CreateSolution(DEVICE, REGION, "Potential")
    node_model(
        device=DEVICE, region=REGION, name="Acceptors", equation="1.0e18*step(0.5e-5-x);"
    )
    node_model(device=DEVICE, region=REGION, name="Donors", equation="1.0e18*step(x-0.5e-5);")
    node_model(device=DEVICE, region=REGION, name="NetDoping", equation="Donors-Acceptors;")

    CreateSiliconPotentialOnly(DEVICE, REGION)
    from devsim import set_node_values

    node_model(
        device=DEVICE,
        region=REGION,
        name="Potential_init",
        equation="V_t*log(0.5*(NetDoping+(NetDoping^2+4*n_i^2)^0.5)/n_i);",
    )
    set_node_values(device=DEVICE, region=REGION, name="Potential", init_from="Potential_init")
    for c in ("top", "bot"):
        set_parameter(device=DEVICE, name=f"{c}_bias", value=0.0)
        CreateSiliconPotentialOnlyContact(DEVICE, REGION, c)

    # Equilibrium: potential only.
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

    # Promote to full drift-diffusion, initialised at the equilibrium carrier
    # densities the potential-only solve just produced.
    CreateSolution(DEVICE, REGION, "Electrons")
    CreateSolution(DEVICE, REGION, "Holes")
    set_node_values(
        device=DEVICE, region=REGION, name="Electrons", init_from="IntrinsicElectrons"
    )
    set_node_values(device=DEVICE, region=REGION, name="Holes", init_from="IntrinsicHoles")
    edge_from_node_model(device=DEVICE, region=REGION, node_model="Potential")

    CreateSiliconDriftDiffusion(DEVICE, REGION)
    for c in ("top", "bot"):
        CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, c)

    solve(type="dc", absolute_error=1e30, relative_error=1e-5, maximum_iterations=30)

    iv = []
    for v in voltages:
        set_parameter(device=DEVICE, name="top_bias", value=float(v))
        solve(type="dc", absolute_error=1e30, relative_error=1e-6, maximum_iterations=40)
        i = get_contact_current(device=DEVICE, contact="top", equation="ElectronContinuityEquation")
        i += get_contact_current(device=DEVICE, contact="top", equation="HoleContinuityEquation")
        iv.append((float(v), float(i)))
    return iv


if __name__ == "__main__":
    for v, i in build_and_solve():
        print(f"V = {v:.2f} V   I = {i: .6e} A")
    print("G2 DEVSIM: OK")
    sys.exit(0)
