"""Download the results of your finished tasks. Run it again: nothing is downloaded twice.

Spends nothing. A result is named after its original scan when the record on this computer
knows it; a file is never replaced.
"""
import nalvera

finished = nalvera.tasks(
    state="finished",              # default: "all". Options: "active" | "finished" | "all"
    limit=100,                     # default: 50. Options: 1 to 100
)
done = [task for task in finished if task.state == "completed"]
paths = nalvera.download(
    done,
    out="results",                 # default: "results"
    overwrite=False,               # default: False (a result saved before is returned, not
                                   # fetched again). Options: True | False
)
print(len(paths), "files saved;", len(paths.skipped), "tasks had no result")
