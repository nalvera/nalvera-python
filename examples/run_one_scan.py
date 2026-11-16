"""Segment one CT scan with TotalSegmentator, wait for the task and save the result.

Spends credits: one task, budget "personal" (free tasks are used first).
Sign in first: nalvera.login(email, api_key).
"""
import nalvera

SCAN = "ct_scan.nii.gz"

nalvera.login()                    # asks for the email and the key, or reads NALVERA_EMAIL and
                                   # NALVERA_API_KEY, or what login(..., remember=True) stored
tasks = nalvera.run(
    SCAN,                          # a .nii or .nii.gz file, a folder of them, a DICOM folder
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    task="total",                  # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
finished = nalvera.wait(tasks)     # shows queue position, time waited, time running
paths = nalvera.download(
    finished,
    out="results",                 # default: "results"
)
print(paths)                       # results/ct_scan__totalsegmentator_ct__total.nii.gz
