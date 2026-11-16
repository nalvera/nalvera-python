"""Check the sign-in and the account: plan, key, permissions, end date, free tasks, budgets.

Needs a key (see set_api_key.py). Spends nothing.
Run: python sign_in_and_check_account.py
"""
import nalvera

nalvera.login()                         # an unknown, expired or revoked key raises here
me = nalvera.me()
key = me["key"]
expires, days_left = nalvera.key_expiry()
print("Account:", me["email"], "on plan", me["plan_label"])
print("Key:", key["prefix"], key["name"], "may:", ", ".join(key["scopes"]))
print("Key ends:", expires, "in", days_left, "days")
if "tasks:write" not in key["scopes"]:
    print("This key may not run tasks: create one that may, under Settings → API.")
if me["terms_reaccept_required"]:
    print("Accept the new Terms on nalvera.ai before queueing tasks.")

budgets = nalvera.budgets()
print("Free tasks left:", budgets["free"]["free_tasks_left"])
print("Personal credits:", budgets["personal"]["available_credits"])
print("Encrypted uploads:", nalvera.encryption())
