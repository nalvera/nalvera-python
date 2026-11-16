#!/usr/bin/env bash
# Write the anonymized copies of scans, as a run prepares them, into one folder.
# Look at what would leave this computer: nothing is sent and no key is needed. Needs
# ct_scan.nii.gz and a folder scans/ here. Run: bash anonymize_only.sh
set -euo pipefail

# Each copy is named by a new code; no file in anonymized/ is replaced.
# --out  default: none (required)
nalvera anonymize ct_scan.nii.gz scans --out anonymized
ls -l anonymized                # scan-XXXXXXXX.nii.gz copies, and nothing else
