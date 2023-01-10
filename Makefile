.PHONY: setup, install, update, format, lint, test, html, build, clean

setup:
	@git init
	@git commit --allow-empty -m "Initial commit"
	@git add .
	@git commit -m "Add template folder"
	@poetry install --with dev,cqa,test,docs
	@git add poetry.lock
	@git commit -m "Add poetry.lock"
	@git checkout -b develop
	@pre-commit install
	@pre-commit autoupdate
	@git add .pre-commit-config.yaml
	@git commit -m "Update hooks revision or tag"

install:
	@poetry install --with dev,cqa,test,docs
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
	@pytest --cov=src --cov-report=term-missing --cov-report=html

html:
	@sphinx-build -b html ./docs/source ./docs

build:
	@git checkout main
	@poetry build

clean:
	-@rm -rf .pytest_cache
	-@rm -rf dist
	-@rm -rf htmlcov
	-@find ./ -name "__pycache__" -exec rm -rf {} \;
	-@rm -rf .coverage
