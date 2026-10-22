"""Create a CPU-only copy of the Triton model repository for local runs without a GPU.

Rewrites instance_group to KIND_CPU and drops the TensorRT accelerator block,
leaving everything else (dynamic batching, versions, warm-up) the same.

  python scripts/make_cpu_repo.py model_repository build/model_repository_cpu
"""

import re
import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.npz", "*.json"))
for cfg in dst.glob("*/config.pbtxt"):
    text = cfg.read_text()
    text = text.replace("KIND_GPU", "KIND_CPU")
    text = re.sub(r"\n  execution_accelerators \{.*?\n  \}\n", "\n", text, flags=re.S)
    cfg.write_text(text)
    print(f"wrote {cfg}")
