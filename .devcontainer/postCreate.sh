#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as errors
set -o pipefail  # Detect pipeline errors

# Update .gitignore
echo "Updating .gitignore..."
GITIGNORE_ENTRIES=(
    "*.aider"
    ".env.local"
    "node_modules"
)
for ENTRY in "${GITIGNORE_ENTRIES[@]}"; do
    if ! grep -qx "$ENTRY" .gitignore; then
        echo "$ENTRY" >> .gitignore
    fi
done

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Perform an editable install if applicable
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e .
fi

echo "Post-create script has been executed successfully!"