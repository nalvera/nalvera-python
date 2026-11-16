"""List the scans recorded on this computer: which code is which scan, its tasks and results.

Reads the record on this computer: no request. In a notebook, nalvera.scans() shows a table.
Run: python list_my_scans_and_tasks.py
"""
import nalvera

record = nalvera.scans()
print("Record:", record.path)
for row in record:
    print(row["code"], row["original_name"], row["series"], row["task_id"] or "-",
          row["model"], row["task"], row["result_path"])

codes = sorted({row["code"] for row in record if row["original_name"] == "ct_scan.nii.gz"})
print("ct_scan.nii.gz is", codes or "not recorded")
