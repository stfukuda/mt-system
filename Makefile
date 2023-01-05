.PHONY: install, update, format, lint, test, html, build, publish-test, publish, clean

install:
	@git init
	@git commit --allow-empty -m "Initial commit"
	@git add .
	@git commit -m "Add template folder"
	@poetry install
	@git add poetry.lock
	@git commit -m "Add poetry.lock"
	@git checkout -b develop
	@pre-commit install
	@pre-commit autoupdate
	@git add .pre-commit-config.yaml
	@git commit -m "Update hooks revision or tag"

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
	@poetry build

publish-test:
	@poetry publish -r test-pypi

publish:
	@poetry publish

clean:
	-@pre-commit clean
	-@rm -rf dist
	-@rm -rf htmlcov
	-@rm -rf .coverage
