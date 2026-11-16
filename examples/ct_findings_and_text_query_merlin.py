"""Ask Merlin about an abdominal CT: its findings as text, and how well a text query matches.

Spends credits: two tasks, budget "personal". The query task needs text= (needs_text).
Run: python ct_findings_and_text_query_merlin.py
"""
import numpy as np

import nalvera

SCAN = "ct_scan.nii.gz"

queued = []
for task, text in [("phenotype", None), ("contrastive", "pleural effusion")]:
    queued += nalvera.run(
        SCAN,
        model="merlin",            # required. See: nalvera.models()
        task=task,                 # default: None (the model's default task, phenotype).
                                   # Options: the tasks nalvera.models() lists
        budget="personal",         # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
        text=text,                 # default: None. Required by a task with needs_text
    )
for path in nalvera.download(nalvera.wait(queued), out="results"):  # default: "results"
    if path.suffix == ".txt":      # phenotype, report, five_year_prediction
        print(path.read_text(encoding="utf-8"))
    elif path.suffix == ".npy":    # contrastive, image_embedding
        print("similarity:", np.load(path))
