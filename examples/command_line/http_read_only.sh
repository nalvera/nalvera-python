#!/usr/bin/env bash
# Read your account, models and tasks over HTTP with curl: read-only, spends nothing.
# Uploads go through the nalvera package, which anonymizes scans before they leave the computer.
# Needs NALVERA_EMAIL and NALVERA_API_KEY in the environment. Run: bash http_read_only.sh
set -euo pipefail

BASE="${NALVERA_BASE_URL:-https://nalvera.ai}"
# The key travels over HTTPS only (http:// only to this computer).
case "$BASE" in
  https://* | http://localhost | http://localhost:* | http://127.0.0.1 | http://127.0.0.1:*) ;;
  *) echo "NALVERA_BASE_URL must start with https://" >&2; exit 2 ;;
esac

# The key reaches curl on standard input, so it never shows in the list of processes.
get() {
  printf 'Authorization: Bearer %s\n' "$NALVERA_API_KEY" |
    curl --silent --show-error --fail-with-body -H @- \
      -H "X-Nalvera-Email: $NALVERA_EMAIL" "$BASE$1"
  echo
}

get /api/v1/me                              # account, key, limits
get /api/v1/models                          # the models and tasks you may run
get "/api/v1/tasks?state=active&limit=50"   # state: active | finished | all; limit: 1 to 100
