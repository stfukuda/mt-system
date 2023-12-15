#!/bin/sh

# Git-Flow
apt-get update
apt-get install -y git-flow

apt-get update
apt-get install -y gnupg2 locales vim

echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen

# NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
# Nodejs
nvm install --lts
nvm use --lts

npm i -g npm git-cz

python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U --user pipx
python3 -m pipx ensurepath --force
python3 -m pipx completions
pipx install poetry
poetry self add poetry-plugin-export
poetry self add "poetry-dynamic-versioning[plugin]"
