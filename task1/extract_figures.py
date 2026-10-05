"""Save every PNG figure embedded in an executed notebook as figures/<Output>-<n>.png (for review and the report)."""
import base64
import re
import sys
from pathlib import Path

import nbformat

nb_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path.parent / "figures_png"
out_dir.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(nb_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    label, count = None, 0
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            found = re.findall(r"Output (\d\.\d)", output.get("text", ""))
            if found:
                label, count = found[-1], 0
        png = output.get("data", {}).get("image/png") if "data" in output else None
        if png and label:
            count += 1
            path = out_dir / f"{label}-{count}.png"
            path.write_bytes(base64.b64decode(png))
            print("wrote", path.name)
