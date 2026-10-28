
ls_path_cfg = [
    "v247.11_classes.2s-15frames.j",
    "v248.11_classes.2s-15frames.b",
    "v249.11_classes.2s-15frames.jm",
    "v250.11_classes.2s-15frames.bm",
]
num_models = 10
ls_devices = [0, 0, 0, 0]

import concurrent.futures
import subprocess
import sys

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for id_model in range(num_models):
        futures = []
        for name_cfg, device in zip(ls_path_cfg, ls_devices):
            futures.append(
                executor.submit(
                    subprocess.run,
                    args=f"{sys.executable} main.py --config config/fs26/{name_cfg}.yaml --device {device} --overwrite --id_model {id_model}",
                    # cwd="/home/laptq/laptq-fs26-shoplifting-detection/submodules/ProtoGCN",
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]
