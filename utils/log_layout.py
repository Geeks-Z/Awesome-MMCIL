"""Canonical model and method names for benchmark log directories."""

from pathlib import Path


# Internal configuration identifiers are intentionally separate from the names
# used by the original papers.  Every alias for one method must resolve to the
# same directory so case or spelling differences cannot split its logs.
_CANONICAL_METHOD_DIRS = {
    "area": "AREA",
    "aera": "AREA",
    "bofa": "BOFA",
    "clg-cbm": "CLG-CBM",
    "clg_cbm": "CLG-CBM",
    "coda": "CODA-Prompt",
    "coda_prompt": "CODA-Prompt",
    "coop": "CoOp",
    "dualprompt": "DualPrompt",
    "dual_prompt": "DualPrompt",
    "ease": "EASE",
    "engine": "ENGINE",
    "finetune": "FineTune",
    "foster": "FOSTER",
    "l2p": "L2P",
    "l2p_without": "L2P-without",
    "memo": "MEMO",
    "mg_clip": "MG-CLIP",
    "mind": "MIND",
    "proof": "PROOF",
    "rapf": "RAPF",
    "sagla": "SAGLA",
    "sagla-c": "SAGLA-C",
    "simplecil": "SimpleCIL",
    "simple_cil": "SimpleCIL",
    "tuna": "TUNA",
    "zs_clip": "ZS-CLIP",
    "zs-clip": "ZS-CLIP",
    "aper": "APER",
    "aper_adapter": "APER-Adapter",
    "aper_finetune": "APER-FineTune",
    "aper_ssf": "APER-SSF",
    "aper_vpt": "APER-VPT",
}

_BACKBONE_LOG_ROOTS = {
    "openai_clip": "OpenAI_CLIP_ViTB16",
    "openai-clip": "OpenAI_CLIP_ViTB16",
    "clip": "OpenCLIP_LAION400M_ViTB16",
    "clip_laion2b": "OpenCLIP_LAION2B_ViTB16",
}


def canonical_method_name(model_name):
    """Return the paper-style directory name for a config method identifier."""
    identifier = str(model_name).strip().lower()
    try:
        return _CANONICAL_METHOD_DIRS[identifier]
    except KeyError as error:
        raise ValueError(f"Unsupported log method {model_name!r}") from error


def backbone_log_root(backbone_type):
    """Return the directory name for the exact CLIP model family in use."""
    identifier = str(backbone_type).strip().lower()
    if identifier.startswith("pretrained_"):
        identifier = identifier[len("pretrained_") :]
    try:
        return _BACKBONE_LOG_ROOTS[identifier]
    except KeyError as error:
        raise ValueError(f"Unsupported log backbone {backbone_type!r}") from error


def log_directory(model_name, backbone_type, base_dir="logs"):
    """Build ``<base>/<model family>/<paper method>`` for a run."""
    return Path(base_dir) / backbone_log_root(backbone_type) / canonical_method_name(
        model_name
    )
