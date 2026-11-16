"""Generate a synthetic CT with its anatomy label map (MAISI-v2, mode image-mask).

The result has two files, image and mask: download_one_part_of_a_generation.py fetches one.
Spends credits: one task, budget "personal". Run: python generate_ct_with_mask.py
"""
import nalvera

task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-ct",                    # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "image-mask",           # required. Options: "image-only" | "image-mask"
        "body_region": ["abdomen"],     # required for image-mask, one region. Options: "head" |
                                        # "chest" | "thorax" | "abdomen" | "pelvis" | "lower"
        "anatomy_list": [],             # default: [] (no tumor). Options: in the abdomen,
                                        # "hepatic tumor" | "pancreatic tumor" |
                                        # "colon cancer primaries" | "bone lesion"
        "output_size": [512, 512, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256
        "spacing": [0.781, 0.781, 2.981],  # required. Options: [S, S, Sz] in mm, each 0.5
                                           # to 5.0; X times S at least 256 mm
        "seed": 11,                     # default: 0. Options: 0 to 2147483647
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    label="abdomen-mask-01",            # default: None (the file name starts with the task id)
)
finished = nalvera.wait([task])
print(finished[0].parts)                # ['image', 'mask']
print(nalvera.download(finished, out="generated"))  # default: "results"
