#!/bin/sh

apt-get update
apt-get install -y gnupg2 locales vim

echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen

apt-get update
apt-get install -y ca-certificates curl gnupg
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

NODE_MAJOR=20
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list

apt-get update
apt-get install nodejs -y

npm i -g npm git-cz

python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U --user pipx
python3 -m pipx ensurepath
pipx install poetry
poetry self add poetry-plugin-export
poetry self add "poetry-dynamic-versioning[plugin]"
