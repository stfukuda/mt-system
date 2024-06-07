#!/bin/bash

# Install Vim, Git, Git-Flow
apt-get update
apt-get install -y vim git git-flow

# NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
# Nodejs
nvm install --lts
nvm use --lts

# Commitizen
npm i -g npm git-cz

# Poetry
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U --user pipx
python3 -m pipx ensurepath
export PATH=$PATH:/root/.local/bin
pipx install poetry
pipx inject poetry poetry-plugin-export
pipx inject poetry "poetry-dynamic-versioning[plugin]"

# Run initialization process immediately after creating devcontainer
if [ -d ".git" ]; then
    make sync
else
    make setup
fi
