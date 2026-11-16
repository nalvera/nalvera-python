"""List the models and tasks this account may run, with their prices.

Spends nothing.
Run: python list_models_and_tasks.py
"""
import nalvera

nalvera.login()                     # asks, or reads the environment
print(nalvera.models())             # a table: one row per task, with its price
