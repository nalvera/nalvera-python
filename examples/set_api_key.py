"""Sign in to Nalvera.AI from Python, then check the sign-in.

Get a key: sign in at https://nalvera.ai, menu (⋮) → Settings → API, switch on API access and
create a key; tick "Run tasks and spend credits" to run tasks. The key is shown once.
Spends nothing. Run: python set_api_key.py
"""
import os

import nalvera

# 1. The email and the key as arguments. A key typed into a notebook cell is saved in the
#    .ipynb file and in IPython's history file:
#        nalvera.login("you@university.org", "nvk_...")
# 2. No key in the code: asks for the email and the key (hidden) when the notebook or the
#    terminal can ask; nothing is saved in the notebook:
#        nalvera.login()
# 3. Asked for as in 2, then stored on this computer, so later sessions need neither (the
#    system keychain with "nalvera[keyring]" installed, else a file only you can read):
#        nalvera.login(remember=True)
# 4. A cluster or a shared script: NALVERA_EMAIL and NALVERA_API_KEY in the environment,
#    filled from a secret store; nalvera.login() and every function read them.
EMAIL = os.environ.get("NALVERA_EMAIL")
API_KEY = os.environ.get("NALVERA_API_KEY")
if EMAIL and API_KEY:
    nalvera.login(EMAIL, API_KEY)               # 1, with the values read at run time
else:
    nalvera.login()                             # 2, or what 3 stored

me = nalvera.me()
expires, days_left = nalvera.key_expiry()
print("Signed in as", me["email"], "with key", me["key"]["prefix"], "until", (expires or "-")[:10])

# Sign out; forget=True also removes what remember=True stored:
#     nalvera.logout(forget=True)
