"""List every task of the account, queued by scripts and on the website, page by page.

Spends nothing. Original names come from the record on this computer when it knows a task.
"""
import nalvera

offset = 0
while True:
    page = nalvera.tasks(
        state="all",               # default: "all". Options: "active" | "finished" | "all"
        limit=100,                 # default: 50. Options: 1 to 100
        offset=offset,             # default: 0
    )
    for task in page:
        print(task.id, task.source, task.model, task.task, task.state, task.display_name)
    if len(page) < 100:
        break
    offset += len(page)
