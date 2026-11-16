"""Cancel every task your scripts queued that is still waiting or running.

A running task stops at the worker's next check. Refunds follow the website's rules. One
task: nalvera.cancel("TASK_ID").
"""
import nalvera

for task in nalvera.tasks(state="active"):  # default: "all". Options: "active" | "finished" | "all"
    if task.source == "api":                # "web" for tasks queued on the website
        task = nalvera.cancel(task)
        print(task.display_name, "cancelling" if task.cancelling else task.state)
