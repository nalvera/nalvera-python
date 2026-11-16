"""Generate a synthetic CT volume with MAISI-v2 (no input scan), wait and save it.

Spends credits: one task, budget "personal". Every setting with its range:
nalvera.models() (each task's generation). Run: python generate_ct.py
"""
import nalvera

nalvera.login()                         # asks, or reads the environment
task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-ct",                    # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "image-only",           # required. Options: "image-only" | "image-mask"
        "output_size": [256, 256, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256
        "spacing": [1.5, 1.5, 2.0],     # required. Options: [S, S, Sz] in mm, each 0.5 to 5.0;
                                        # X times S at least 256 mm
        "num_output_samples": 1,        # default: 1. Options: 1 to 64
        "seed": 7,                      # default: 0. Options: 0 to 2147483647
        "num_inference_steps": 30,      # default: 30. Options: 1 to 1000
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    label="phantom-01",                 # default: None (the file name starts with the task id)
)
finished = nalvera.wait([task])
paths = nalvera.download(finished, out="generated")  # default: "results"
print(paths)                        # generated/phantom-01__nv_generate_ctmr__rflow-ct.nii.gz
