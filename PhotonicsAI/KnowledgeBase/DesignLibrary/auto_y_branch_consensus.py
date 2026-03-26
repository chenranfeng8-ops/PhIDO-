
import gdsfactory as gf

@gf.cell
def auto_y_branch_consensus(
    length: float = 20.0,
    angle: float = 10.0,
    gap: float = 2.0,
    width: float = 0.5,
) -> gf.Component:
    """Auto-generated Y-Branch Splitter from: Consensus parameters from 0 papers
    Source: Default parameters (no papers retrieved)
    
    Y-branch splitter with smooth S-bend transition.
    Uses mmi1x2 as base for better performance.
    """
    c = gf.Component()
    
    # Use mmi1x2 for reliable Y-branch performance
    # Y-branch is essentially a 1x2 splitter
    ref = c << gf.components.mmi1x2(
        length_mmi=length * 0.3,  # MMI region
        width_mmi=gap * 2 + width,
        gap_mmi=gap,
        width_taper=width,
        length_taper=length * 0.35,  # Taper for smooth transition
    )
    c.add_ports(ref.ports)
    return c
