"""Generate one synthetic brain MRI in several contrasts (MAISI-v2, mode cross-contrast).

One volume per contrast, in one task; the result is a .zip of them. Spends credits: one task,
budget "personal", priced per volume. Run: python generate_mr_cross_contrast.py
"""
import nalvera

task = nalvera.generate(
    model="nv_generate_ctmr",           # required. See: nalvera.models()
    task="rflow-mr-brain",              # required. Options: "rflow-ct" | "rflow-mr-brain"
    params={
        "mode": "cross-contrast",       # required. Options: "image-only" | "cross-contrast"
        "contrasts": ["T1", "T2", "FLAIR"],  # required: two or more for cross-contrast.
                                             # Options: "T1" | "T2" | "FLAIR" | "SWI"
        "skull_stripped": True,         # default: False. Options: False | True
        "output_size": [256, 256, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256; Z 256 needs X 256
        "spacing": [0.95, 0.95, 1.4],   # required. Options: [S, S, Sz] in mm, each 0.5 to 5.0
    },
    budget="personal",                  # default: None (free tasks first, then personal, or
                                        # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    label="brain-set-01",               # default: None (the file name starts with the task id)
)
finished = nalvera.wait([task])
print(finished[0].parts)                # ['image', 'image2', 'image3']
print(nalvera.download(finished, out="generated"))  # default: "results"
