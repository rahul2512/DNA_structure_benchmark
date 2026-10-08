#!/usr/bin/env python3
"""
Patch script: replaces deprecated @render.download with @render.download_button
in an existing app.py. Run once from the tetramer_comparison folder:
    python3 patch_download.py
"""
import re, sys, pathlib

target = pathlib.Path(__file__).parent / "app.py"
if not target.exists():
    sys.exit(f"app.py not found at {target}")

src = target.read_text()
fixed = src.replace("@render.download(", "@render.download_button(")
n = src.count("@render.download(")
target.write_text(fixed)
print(f"Patched {n} occurrence(s) in {target}")
