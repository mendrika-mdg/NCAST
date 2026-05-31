#!/bin/bash

set -e

clean_dir () {
    dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        rm -rf "$dir"/*
        echo "Cleaned: $dir"
    else
        echo "Skipped (empty or missing): $dir"
    fi
}

clean_dir /home/users/mendrika/NCAST/Output/submission-logs/inference/output
clean_dir /home/users/mendrika/NCAST/Output/submission-logs/inference/error
