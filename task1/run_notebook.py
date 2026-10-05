"""Execute Assignment1.ipynb headlessly, optionally stopping before a given cell.

Usage: python run_notebook.py PRESET OUT.ipynb [STOP_CELL]
The kernel inherits PA1_PRESET (and any PA1_* variables already set in the environment).
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

preset, out = sys.argv[1], sys.argv[2]
stop = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.environ["PA1_PRESET"] = preset
here = os.path.dirname(os.path.abspath(__file__))

nb = nbformat.read(os.path.join(here, "Assignment1.ipynb"), as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": here}})
started = time.time()
with client.setup_kernel():
    for index, cell in enumerate(nb.cells):
        if stop is not None and index >= stop:
            break
        if cell.cell_type != "code":
            continue
        t0 = time.time()
        client.execute_cell(cell, index)
        texts = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                texts.append(output["text"].rstrip())
            elif output.get("output_type") == "error":
                texts.append(f"ERROR {output['ename']}: {output['evalue']}")
        print(f"--- cell {index} ({time.time() - t0:.1f}s)", flush=True)
        for text in texts:
            print("   " + text.replace("\n", "\n   "), flush=True)
nbformat.write(nb, os.path.join(here, out))
print(f"saved {out}; total {time.time() - started:.1f}s", flush=True)
