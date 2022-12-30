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
	@poetry run pre-commit install
	@poetry run pre-commit autoupdate
	@git add .pre-commit-config.yaml
	@git commit -m "Update hooks revision or tag"

update:
	@poetry update
	@poetry run pre-commit autoupdate

format:
	-@poetry run isort ./src ./tests
	-@poetry run black ./src ./tests

lint:
	-@poetry run flake8 ./src ./tests --color auto
	-@poetry run bandit -r ./src ./tests

test:
	@poetry run tox

html:
	@poetry run sphinx-build -b html ./docs/source ./docs

build:
	@poetry build

publish-test:
	@poetry publish -r test-pypi

publish:
	@poetry publish

clean:
	-@poetry run pre-commit clean
	-@rm -rf .tox
	-@rm -rf dist
	-@rm -rf htmlcov
	-@rm -rf .coverage