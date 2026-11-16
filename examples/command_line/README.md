# The nalvera command: examples

The `nalvera` command runs the same tasks from a terminal or a shell script. Each file here uses commands only.

| You want to | File |
|---|---|
| Store your key with `nalvera login`, check it with `nalvera whoami`, remove it with `nalvera logout` | [set_api_key.sh](set_api_key.sh) |
| Use every command once, in the order a new user needs them | [cli_walkthrough.sh](cli_walkthrough.sh) |
| Write the anonymized copies of files and folders with `nalvera anonymize ct_scan.nii.gz scans --out anonymized` | [anonymize_only.sh](anonymize_only.sh) |
| Run on a Slurm cluster | [slurm_job.sh](slurm_job.sh) |
| Run from GitHub Actions, the key in a repository secret | [github_actions_workflow.yml](github_actions_workflow.yml) |
| Read your account, models and tasks over HTTP with curl | [http_read_only.sh](http_read_only.sh) |
| Look up any command or option, and the exit statuses | [cli_cheatsheet.md](cli_cheatsheet.md) |
