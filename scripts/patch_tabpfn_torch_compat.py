"""One-time compatibility patch for tabpfn==0.1.11 against modern PyTorch.

tabpfn/layer.py (written in 2022, against an early-1.x-era PyTorch) does:

    from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor,
        Optional, MultiheadAttention, Linear, Dropout, LayerNorm

`Optional` was never actually defined in that torch module -- it was only
reachable there because torch's own transformer.py did a `from typing import
*`-style import at the time. Modern torch's transformer.py imports `typing`
names explicitly and no longer re-exports `Optional`, so this import now
raises ImportError. Every other name in that import list still resolves
correctly from torch.nn.modules.transformer (checked directly against the
installed torch version before writing this patch).

Fix: rewrite the one import line to pull `Optional` from `typing` directly,
which is both correct and what the name always was. Idempotent: skips
rewriting if already patched, so it's safe to run after every fresh
`pip install tabpfn==0.1.11` in a new environment.

Usage: python scripts/patch_tabpfn_torch_compat.py
"""

from __future__ import annotations

import site
import sysconfig
from pathlib import Path

BROKEN_IMPORT = (
    "from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, "
    "Optional, MultiheadAttention, Linear, Dropout, LayerNorm"
)
FIXED_IMPORT = (
    "from typing import Optional\n"
    "from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, "
    "MultiheadAttention, Linear, Dropout, LayerNorm"
)


def _find_layer_py() -> str:
    # Deliberately avoid `import tabpfn` / importlib.util.find_spec("tabpfn.layer")
    # here: both execute tabpfn/__init__.py to resolve the parent package,
    # which is exactly the broken import this script exists to fix. Locate
    # the file directly on disk instead.
    candidate_roots = [Path(p) for p in site.getsitepackages()]
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidate_roots.append(Path(purelib))

    for root in candidate_roots:
        candidate = root / "tabpfn" / "layer.py"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        f"could not find tabpfn/layer.py under any of {candidate_roots}; "
        "is tabpfn==0.1.11 installed in this environment?"
    )


def main() -> None:
    layer_path = _find_layer_py()

    text = open(layer_path).read()
    if FIXED_IMPORT in text:
        print(f"{layer_path} already patched, nothing to do.")
        return
    if BROKEN_IMPORT not in text:
        raise RuntimeError(
            f"expected broken import line not found in {layer_path} -- "
            "tabpfn version mismatch or torch compatibility has changed again; "
            "re-inspect before blindly patching."
        )

    patched = text.replace(BROKEN_IMPORT, FIXED_IMPORT)
    with open(layer_path, "w") as f:
        f.write(patched)
    print(f"patched {layer_path}")


if __name__ == "__main__":
    main()
