
ls_path_cfg = [
    "v251.3_classes.2s-15frames.j",
    "v252.3_classes.2s-15frames.b",
    "v253.3_classes.2s-15frames.jm",
    "v254.3_classes.2s-15frames.bm",
]
num_models = 3
ls_devices = [3, 3, 3, 3]

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
