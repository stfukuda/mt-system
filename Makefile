.PHONY: setup, update, format, lint, test, html, build, publish-test, publish, clean

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
	-@rm -rf .pytest_cache
	-@rm -rf dist
	-@rm -rf htmlcov
	-@find ./ -name "__pycache__" -exec rm -rf {} \;
	-@rm -rf .coverage
