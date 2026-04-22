from .base import FusionModule
from .HCL import HCLFusion
from .SLIDE import SLIDEFusion
from .HNN import HNNFusion
from .ConVIRT import ConVIRTFusion
from .MMFL import MMFLFusion
from .MISA import MISAFusion
from .DLF import DLFFusion
from .TSD import TSDFusion
from .JIVE import JIVEFusion
from .sJIVE import sJIVEFusion
# Registry mapping fusion_type string -> class
_FUSION_REGISTRY = {
    "hcl"    : HCLFusion,
    "slide"  : SLIDEFusion,
    "hnn"    : HNNFusion,
    "convirt": ConVIRTFusion,
    "mmfl"   : MMFLFusion,
    "misa"   : MISAFusion,
    "dlf"    : DLFFusion,
    "tsd"    : TSDFusion,
    "jive"   : JIVEFusion,
    "sjive"  : sJIVEFusion,
}
def build_fusion(fusion_type: str, **kwargs) -> FusionModule:
    """
    Factory function: instantiate a FusionModule by name.

    For HCL, accepts either:
      - r_list: list of 7 ints (per-structure dims), OR
      - r: single int (broadcast to all 7 structures for backward compat)

    Only passes kwargs that are accepted by the target fusion class constructor.
    """
    import inspect

    fusion_type = fusion_type.lower()
    if fusion_type not in _FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion_type={fusion_type!r}. "
            f"Available options: {list(_FUSION_REGISTRY.keys())}"
        )

    cls = _FUSION_REGISTRY[fusion_type]

    # For HCL: convert single r to r_list if r_list not provided
    if fusion_type == "hcl":
        if "r_list" not in kwargs and "r" in kwargs:
            kwargs["r_list"] = [kwargs["r"]] * 7
        kwargs.pop("r", None)

    # Filter kwargs to only those accepted by the target class __init__
    valid_params = inspect.signature(cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**filtered_kwargs)