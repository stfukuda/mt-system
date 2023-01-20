.PHONY: setup, install, update, format, lint, test, html, build, clean

setup:
	@git init
	@git commit --allow-empty -m "Initial commit"
	@git checkout -b develop
	@git checkout -b setup
	@git add .
	@git commit -m "chore: 🤖 add template folder"
	@poetry install --with test,docs
	@git add poetry.lock
	@git commit -m "chore: 🤖 add poetry.lock"
	@pre-commit install
	@pre-commit autoupdate
	@git add .pre-commit-config.yaml
	@git commit -m "chore: 🤖 update pre-commit hooks revision or tag"
	@git checkout develop
	@git merge setup
	@git checkout -d setup

install:
	@poetry install --with test,docs
	@pre-commit install

update:
	@poetry update
	@pre-commit autoupdate

format:
	-@isort ./src ./tests
	-@black ./src ./tests

lint:
	-@flake8 ./src ./tests --color auto
	-@bandit -r ./src ./tests

test:
	@pytest --cov=src --cov-report=term-missing --cov-report=html -n 4

html:
	@sphinx-build -b html ./docs/source ./docs

build:
	@git checkout main
	@poetry build

clean:
	-@rm -rf .pytest_cache
	-@rm -rf dist
	-@rm -rf htmlcov
	-@rm -rf ./src/__pycache__
	-@rm -rf ./tests/__pycache__
	-@rm -rf .coverage
