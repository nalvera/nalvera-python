"""Generate a synthetic brain MRI volume (T1) with MAISI-v2, wait and save it.

Spends credits: one task, budget "personal". Settings and ranges as the platform lists them:
nalvera.models() (each task's generation). Run: python generate_mr.py
"""
import nalvera

nalvera.login()                         # asks, or reads the environment
task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-mr-brain",              # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "image-only",           # required. Options: "image-only" | "cross-contrast"
        "contrasts": ["T1"],            # required: one for image-only.
                                        # Options: "T1" | "T2" | "FLAIR" | "SWI"
        "skull_stripped": False,        # default: False. Options: False | True
        "output_size": [256, 256, 256],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256; Z 256 needs X 256
        "spacing": [0.95, 0.95, 0.7],   # required. Options: [S, S, Sz] in mm, each 0.5 to 5.0
        "cfg_guidance_scale": 2.0,      # default: 2.0. Options: 0.0 to 20.0
        "num_output_samples": 1,        # default: 1. Options: 1 to 64
        "seed": 0,                      # default: 0. Options: 0 to 2147483647
        "num_inference_steps": 30,      # default: 30. Options: 1 to 1000
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    label="brain-t1-01",                # default: None (the file name starts with the task id)
)
finished = nalvera.wait([task])
paths = nalvera.download(finished, out="generated")  # default: "results"
print(paths)                        # generated/brain-t1-01__nv_generate_ctmr__rflow-mr-brain.nii.gz
