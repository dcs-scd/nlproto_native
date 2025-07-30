#!/usr/bin/env python3
"""
YAML-to-NetLogo headless runner for NetLogo 6.4.0
"""

import yaml, subprocess, sys, pathlib, platform
import time, random
from pathlib import Path

MAX_RETRIES = 3

if len(sys.argv) != 2:
    sys.exit("Usage: python run_netlogo.py <config.yaml>")

cfg_path  = pathlib.Path(sys.argv[1])
with cfg_path.open() as fh:
    cfg = yaml.safe_load(fh)

model = pathlib.Path(cfg["modelpath"])
jar   = next(pathlib.Path(cfg["nlpath"]).rglob("netlogo-6.4*.jar"))
xml_file = cfg_path.with_suffix(".xml")
csv_main = cfg_path.with_suffix(".csv")
csv_rows = cfg_path.with_name(cfg_path.stem + "_rows.csv")

experiment = cfg["experiment"]

with open(xml_file, 'w', encoding='utf-8') as f:
    f.write(f'<experiments>\n<experiment name="{cfg["name"]}">\n')
    
    f.write(f'  <timeLimit steps="{experiment["runtime"]}"/>\n')
    
    for metric in experiment["metrics"]:
        f.write(f'  <metric>{metric}</metric>\n')

    for var, spec in experiment["variables"].items():
        mn, mx, st = spec["min"], spec["max"], spec["step"]
        f.write(f'  <enumeratedValueSet variable="{var}">\n')
        for v in range(mn, mx + 1, st):
            f.write(f'    <value value="{v}"/>\n')
        f.write('  </enumeratedValueSet>\n')

    f.write('</experiment>\n</experiments>')

system = platform.system().lower()
java_path_key = "linux" if system == "linux" else "macos" if system == "darwin" else "windows"
cmd = [str(pathlib.Path(cfg["java_home"][java_path_key]) / "bin" / "java"),
       "-Djava.awt.headless=true", "-Xmx1G",
       "-cp", str(jar),
       "org.nlogo.headless.Main",
       "--model", str(model),
       "--setup-file", str(xml_file),
       "--spreadsheet", str(csv_main),
       "--table", str(csv_rows)]

print("RUN:", " ".join(cmd))

for attempt in range(1, MAX_RETRIES+1):
    try:
        subprocess.run(cmd, check=True)
        # success
        break
    except subprocess.CalledProcessError as e:
        if attempt == MAX_RETRIES:
            sys.exit("[FATAL] Last retry failed – bailing out")
        jitter = random.uniform(0.1, 0.5)
        time.sleep(2 ** attempt + jitter)



print("Done →", csv_main)