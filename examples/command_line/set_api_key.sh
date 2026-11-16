#!/usr/bin/env bash
# Give the nalvera command your API key, then check the sign-in. Spends nothing.
# Get a key: sign in at https://nalvera.ai, menu (⋮) → Settings → API, switch on API access and
# create a key; tick "Run tasks and spend credits" to run tasks. The key is shown once.
# Run: bash set_api_key.sh
set -euo pipefail

# 1. Once per computer: asks for the email and the key (hidden input), checks them and stores
#    them (the system keychain with "nalvera[keyring]" installed, else a file only you can read).
# --email  default: none (asked for)
nalvera login

# 2. On a cluster or in CI, fill the environment from a secret store instead (never write the
#    key into this file). The environment comes before what nalvera login stored.
#    export NALVERA_EMAIL="you@university.org"
#    export NALVERA_API_KEY="$(your-secret-store read nalvera-api-key)"

# Check the sign-in: account, key prefix, permissions, end date.
nalvera whoami

# Remove the stored key from this computer:
#    nalvera logout
