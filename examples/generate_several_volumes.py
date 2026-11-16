"""Generate several synthetic CT volumes in one task and unpack them.

Spends credits: one task, budget "personal", priced per volume. The result is one .zip with a
file per volume. Run: python generate_several_volumes.py
"""
import zipfile

import nalvera

task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-ct",                    # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "image-only",           # required. Options: "image-only" | "image-mask"
        "output_size": [256, 256, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256
        "spacing": [1.5, 1.5, 2.0],     # required. Options: [S, S, Sz] in mm, each 0.5 to 5.0;
                                        # X times S at least 256 mm
        "num_output_samples": 3,        # default: 1. Options: 1 to 64
        "seed": 100,                    # default: 0. Options: 0 to 2147483647
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    label="phantom-set",                # default: None (the file name starts with the task id)
)
finished = nalvera.wait([task])
for archive in nalvera.download(finished, out="generated"):  # default: "results"
    with zipfile.ZipFile(archive) as files:
        files.extractall("generated/phantom-set")
        print(files.namelist())
