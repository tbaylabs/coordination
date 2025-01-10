#!/bin/bash

# Add *.aider, .aider.conf.yml, node_modules, and .env.local to .gitignore if not already present
if ! grep -qx "*.aider" .gitignore; then
    echo "*.aider" >> .gitignore
fi
if ! grep -qx ".env.local" .gitignore; then
    echo ".env.local" >> .gitignore
fi
if ! grep -qx "node_modules" .gitignore; then
    echo "node_modules" >> .gitignore
fi

# Install aider-chat using pip
pip install aider-chat
pip install --upgrade pip
pip install litellm
pip install jupyter
#6 December bug requires downgrade to httpx until aider and openai libraries are updated
pip install httpx==0.27.2
pip install pandas tabulate matplotlib
pip install pytest-dotenv


echo "Post-create script has been executed."
