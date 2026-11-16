"""Download one file of a generation that made several: here the mask of an image-mask CT.

Spends credits: one task, budget "personal". task.parts names the files.
"""
import nalvera

task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-ct",                    # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "image-mask",           # required. Options: "image-only" | "image-mask"
        "body_region": ["pelvis"],      # required for image-mask, one region. Options: "head" |
                                        # "chest" | "thorax" | "abdomen" | "pelvis" | "lower"
        "output_size": [512, 512, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256
        "spacing": [0.781, 0.781, 2.981],  # required. Options: [S, S, Sz] in mm, each 0.5
                                           # to 5.0; X times S at least 256 mm
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
)
finished = nalvera.wait([task])
print("Parts:", finished[0].parts)      # ['image', 'mask']
mask = nalvera.download(
    finished,
    out="generated",                    # default: "results"
    part="mask",                        # default: None (every file, in one .zip).
                                        # Options: the task's parts, such as "image" | "mask"
)
print(mask)                             # generated/<task id>__nv_generate_ctmr__rflow-ct__mask
