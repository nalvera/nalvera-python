"""Get structured chest and abdominal reports for a CT scan from NV-Reason-CT, as text.

Spends credits: two tasks, budget "personal". The tasks chest_reasoning and
abdomen_reasoning give reasoning walkthroughs. Run: python ct_report_nv_reason.py
"""
import nalvera

SCAN = "ct_scan.nii.gz"
TASKS = ["chest_report", "abdomen_report"]     # "chest_report" is the default task

queued = []
for task in TASKS:
    queued += nalvera.run(
        SCAN,
        model="nv_reason_ct",      # required. See: nalvera.models()
        task=task,                 # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
        budget="personal",         # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    )
for task in nalvera.wait(queued):
    if task.state == "completed" and task.output_kind == "text":
        for path in nalvera.download(task, out="results"):  # default: "results"
            print(f"-- {task.task} --")
            print(path.read_text(encoding="utf-8"))
