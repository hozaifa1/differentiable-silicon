"""Import shim shared by every tesseract_api.py in this repo.

Locally the package lives at <repo>/src/diffsilicon. Inside a built Tesseract,
`build_config.package_data` drops the same tree next to tesseract_api.py at
/tesseract/diffsilicon. One of the two is always right.
"""

import sys
from pathlib import Path


def bootstrap(api_file: str) -> None:
    here = Path(api_file).resolve().parent
    for cand in (here, here.parents[1] / "src"):
        if (cand / "diffsilicon").is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
