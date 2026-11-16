"""Show your free tasks, personal credits and each lab's balance, and the suggested budget.

Free tasks are always used first. Spends nothing.
Run: python check_budgets_and_free_tasks.py
"""
import nalvera

budgets = nalvera.budgets()
free = budgets["free"]
if free["period_active"]:
    print("A free period is on: tasks cost nothing for now.")
print("Free tasks left:", free["free_tasks_left"], '(budget="free" uses only these)')

personal = budgets["personal"]
print('budget="personal":', personal["available_credits"], "credits:",
      personal["monthly_left"], "of the monthly allowance and", personal["topup"], "topped up")
for lab in budgets["labs"]:
    print(f'budget="lab:{lab["id"]}": {lab["name"]},', lab["balance"], "credits,",
          lab["member_status"])
print("Suggested:", budgets["suggested"])     # the active lab with the most credits, else personal
