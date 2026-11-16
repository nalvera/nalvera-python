"""Follow your waiting and running tasks: queue position, time waited, time running.

Spends nothing. Checks no faster than your key allows (min_poll_seconds). For tasks you
hold, nalvera.wait(tasks) does the same.
"""
import time

import nalvera

pause = nalvera.limits()["min_poll_seconds"]
tasks = nalvera.tasks(state="active")   # default: "all". Options: "active" | "finished" | "all"
while tasks:
    for task in tasks:
        print(f"{task.display_name}: {task.state}, position {task.queue_position},",
              f"waited {task.waited_seconds} s, running {task.running_seconds} s")
    time.sleep(pause)
    tasks = [task for task in map(nalvera.task, tasks) if not task.finished]
print("No task is waiting or running.")
