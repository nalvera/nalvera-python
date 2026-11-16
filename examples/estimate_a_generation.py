"""Ask what a generation would cost, and whether a free task pays for it, before queueing it.

Needs a key; spends nothing. A setting the platform refuses raises InvalidRequest with its
explanation. Run: python estimate_a_generation.py
"""
import nalvera

estimate = nalvera.estimate(
    "nv_generate_ctmr",
    "rflow-mr-brain",
    params={
        "mode": "cross-contrast",       # required. Options: "image-only" | "cross-contrast"
        "contrasts": ["T1", "FLAIR"],   # required: two or more for cross-contrast.
                                        # Options: "T1" | "T2" | "FLAIR" | "SWI"
        "output_size": [256, 256, 128],  # required. Options: [X, X, Z], X from 128 | 256 |
                                         # 384 | 512, Z from 128 | 256; Z 256 needs X 256
        "spacing": [0.95, 0.95, 1.4],   # required. Options: [S, S, Sz] in mm, each 0.5 to 5.0
    },
)
print("Credits:", estimate["credits"])
print("A free task pays for it:", estimate["free"]["available"])
for option in estimate["budgets"]:
    print(option["budget"], option["available_credits"], "enough" if option["sufficient"] else "")
