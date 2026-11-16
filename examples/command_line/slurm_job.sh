#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Run TotalSegmentator on a folder of scans from a Slurm cluster node with internet access.
# Sign in once on the cluster first: nalvera login stores the key readable by you only.
# A site secret store may export NALVERA_EMAIL and NALVERA_API_KEY instead; this script holds
# no key and writes none. Queues no task that would pass 50 credits (budget personal). Submit: sbatch slurm_job.sh
set -euo pipefail

# NALVERA_BUDGET  default: unset (free tasks first, then personal, or a question with a lab)
#   Options: personal | lab:<id> | free | ask
export NALVERA_BUDGET=personal
# NALVERA_MAPPING  default: unset (one file per account in the configuration folder)
export NALVERA_MAPPING="$HOME/nalvera/nalvera_scans.csv"

# --max-credits  default: none (NALVERA_MAX_CREDITS, else no cap)
# --out  default: results
nalvera run "$HOME/scans" --model totalsegmentator_ct --max-credits 50 --out "$HOME/results"
# Exit status: 0 done, 1 scans failed, 3 sign-in, 4 credits or budget, 5 limits or network.
