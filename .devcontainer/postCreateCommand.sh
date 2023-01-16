#!/bin/sh

apt-get update
apt-get install -y gnupg2 locales vim

echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen

curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
# curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install nodejs

npm i -g npm git-cz

python3 -m pip install -U pip setuptools wheel
curl -sSL https://install.python-poetry.org | python3 -

poetry self add "poetry-dynamic-versioning[plugin]"

git init
