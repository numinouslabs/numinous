#!/usr/bin/env bash
set -o errexit  # Exit immediately if a command exits with a non-zero status
set -o nounset  # Treat unset variables as an error and exit immediately
set -o pipefail

# Settings
PUBLIC_REPO="https://github.com/numinouslabs/numinous.git"
RELEASE_DATE=$(date +"%Y-%m-%d-%H-%M")

# The private content is whatever branch this workflow was dispatched from,
# already checked out by the workflow - this script runs from a subdirectory of it.
SOURCE_DIR="${GITHUB_WORKSPACE:-$(cd .. && pwd)}"
SOURCE_BRANCH="${GITHUB_REF_NAME:-$(git -C "$SOURCE_DIR" rev-parse --abbrev-ref HEAD)}"
NEW_BRANCH="sync-${SOURCE_BRANCH//\//-}-$RELEASE_DATE"
echo "Syncing from private branch: $SOURCE_BRANCH"
echo "Public release branch: $NEW_BRANCH"

trap 'echo "Error occurred in script at line: $LINENO"' ERR

# Ensure the token is set
if [[ -z "$PUBLIC_REPO_TOKEN" ]]; then
    echo "Error: PUBLIC_REPO_TOKEN is not set"
    exit 1
fi

# Inject the token into the repository URL
PUBLIC_REPO_AUTH="https://${PUBLIC_REPO_TOKEN}@${PUBLIC_REPO#https://}"

echo "Clone the public repo"
git clone --quiet --branch main "$PUBLIC_REPO_AUTH" public-repo
cd public-repo || exit 1
git checkout -b "$NEW_BRANCH"

echo "Remove old files from this new branch"
git rm -r --cached . > /dev/null
rm -rf ./* > /dev/null

echo "Copy the private content from the checked out branch"
cd ..
rsync -a --exclude='.git' --exclude='sync' "$SOURCE_DIR"/ ./public-repo/ > /dev/null

echo "Add and commit new private files in the public repo"
cd ./public-repo || exit 1
git add .
if git diff --cached --quiet; then
    echo "Nothing to sync - the public repo already matches $SOURCE_BRANCH"
    exit 0
fi
git commit -m "Release $RELEASE_DATE from $SOURCE_BRANCH" > /dev/null

echo "Push the new branch to the public repo"
git push "$PUBLIC_REPO_AUTH" "$NEW_BRANCH"

echo "Changes were pushed to branch $NEW_BRANCH in the public repo - please manually open a PR!"
