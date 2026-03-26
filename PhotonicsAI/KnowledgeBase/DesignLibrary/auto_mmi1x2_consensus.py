
import gdsfactory as gf

@gf.cell
def auto_mmi1x2_consensus(
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
) -> gf.Component:
    """Auto-generated 1x2 MMI from: Consensus parameters from 0 papers
    Source: Default parameters (no papers retrieved)
    """
    c = gf.Component()
    ref = c << gf.components.mmi1x2(
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        width_taper=width_taper,
        length_taper=length_taper,
    )
    c.add_ports(ref.ports)
    return c
