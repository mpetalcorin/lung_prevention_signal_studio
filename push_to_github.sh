#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/mpetalcorin/lung_prevention_signal_studio.git"
BRANCH="main"

printf "\n🧬 Lung Prevention Signal Studio → GitHub deploy helper\n"
printf "Repository: %s\n\n" "$REPO_URL"

if ! command -v git >/dev/null 2>&1; then
  echo "❌ git is not installed. Install git first."
  exit 1
fi

if [ ! -f "app.py" ]; then
  echo "❌ Please run this script from inside the lung_prevention_signal_studio folder."
  exit 1
fi

# Avoid committing local virtual environments and caches.
cat > .gitignore <<'GITIGNORE'
.venv/
venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
.env
.ipynb_checkpoints/
GITIGNORE

# Add a Streamlit config for a clean app launch.
mkdir -p .streamlit
cat > .streamlit/config.toml <<'CONFIG'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
CONFIG

if [ ! -d .git ]; then
  git init
fi

git branch -M "$BRANCH"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git add app.py README.md requirements.txt run_app.sh .gitignore .streamlit/config.toml push_to_github.sh

if git diff --cached --quiet; then
  echo "✅ No new changes to commit."
else
  git commit -m "Add lung prevention signal studio Streamlit app"
fi

echo "\n🚀 Pushing to GitHub..."
git push -u origin "$BRANCH"

echo "\n✅ Done. Open: https://github.com/mpetalcorin/lung_prevention_signal_studio"
