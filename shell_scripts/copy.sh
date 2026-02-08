#!/bin/bash

# Usage: ./copy.sh [source] [target]
# Case 1: ./copy.sh       - copies from 1 to 10
# Case 2: ./copy.sh 20    - copies from 1 to 20 (source defaults to 1)
# Case 3: ./copy.sh 1 10  - copies from 1 to 10

# Case 1: No arguments, default to 1 to 10
if [ -z "$1" ]; then
    source=1
    target=10
# Case 2: One argument, treat as target, source defaults to 1
elif [ -z "$2" ]; then
    source=1
    target=$1
# Case 3: Two arguments, both specified
else
    source=$1
    target=$2
fi

# Copy the files
cp shell_scripts/job${source}.sh shell_scripts/job${target}.sh
cp shell_scripts/job_${source}a.sh shell_scripts/job_${target}a.sh
cp shell_scripts/job_${source}b.sh shell_scripts/job_${target}b.sh
cp shell_scripts/stop${source}.txt shell_scripts/stop${target}.txt

echo "Copied files from ${source} to ${target}:"
echo "  job${source}.sh -> job${target}.sh"
echo "  job_${source}a.sh -> job_${target}a.sh"
echo "  job_${source}b.sh -> job_${target}b.sh"
echo "  stop${source}.txt -> stop${target}.txt"