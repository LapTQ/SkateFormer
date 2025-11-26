
ls_path_cfg = [
    "v235--satudora_107cia--2s-15frames--cluster-skeleton-6--j",
    "v236--satudora_107cia--2s-15frames--cluster-skeleton-6--b",
    "v237--satudora_107cia--2s-15frames--cluster-skeleton-6--jm",
    "v238--satudora_107cia--2s-15frames--cluster-skeleton-6--bm",
]
num_models = 10
ls_devices = [3, 3, 3, 3]

import subprocess
import multiprocessing as mp
import concurrent.futures

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for id_model in range(num_models):
        futures = []
        for name_cfg, device in zip(ls_path_cfg, ls_devices):
            futures.append(
                executor.submit(
                    subprocess.run,
                    args=f"python main.py --config config/fs26/{name_cfg}.yaml --device {device} --overwrite --id_model {id_model}",
                    # cwd="/home/laptq/laptq-fs26-shoplifting-detection/submodules/ProtoGCN",
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]
