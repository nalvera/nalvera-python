"""List the models this account may run, with each task, its price and what it needs.

Spends nothing.
Run: python list_models_and_tasks.py
"""
import nalvera

nalvera.login()                     # asks, or reads the environment
for model in nalvera.models():
    kind = "generation, no scan" if model["generative"] else "/".join(model["modality"])
    print(f"{model['id']}: {model['name']} {model['version']} ({kind}, {model['type']})")
    for task in model["tasks"]:
        mark = " (default)" if task["default"] else ""
        print(f"    task={task['id']!r}{mark}: {task['name']},",
              task["price_eur_per_mvoxel"], "EUR per million voxels")
        if task["needs_text"]:
            print("        needs a text query: run(..., text=...)")
        if task["generation"]:                         # the settings nalvera.generate accepts
            for name, allowed in task["generation"].items():
                print(f"        {name}: {allowed}")
