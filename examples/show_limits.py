"""Show the limits of your key. The package stays within them and waits when the platform asks.

Spends nothing.
Run: python show_limits.py
"""
import nalvera

for name, value in nalvera.limits().items():
    print(f"{name}: {value}")
