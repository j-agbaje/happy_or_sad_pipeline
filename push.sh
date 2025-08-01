#!/bin/bash

# Usage:
# ./gitpush.sh "Your commit message"

# Exit if no commit message is provided
if [ -z "$1" ]; then
  echo "❌ Please provide a commit message."
  echo "Usage: ./gitpush.sh \"Your commit message\""
  exit 1
fi

# Add all changes
git add .

# Commit with the provided message
git commit -m "$1"

# Push to the current branch
git push

echo "✅ Changes pushed to GitHub."
