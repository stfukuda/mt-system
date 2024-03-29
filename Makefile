default: help

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup   Setup git and the project dependencies"
	@echo "  sync    Synchronize development environments"
	@echo "  update  Update the project dependencies"
	@echo "  check   Run formatters and linters to the source code"
	@echo "  docs    Build the project documentation"
	@echo "  build   Build the project to python package and upload to pypi"
	@echo "  clean   Clean up generated files"

.PHONY: setup
setup:
	@if [ -d ./.git ]; then \
		echo "Setup is already done."; \
	else \
		if command -v git &> /dev/null; then \
			git init; \
			git commit --allow-empty -m "initial commit"; \
			git checkout -b develop; \
			git add .; \
			git commit -m "add template folder"; \
			poetry install --with dev,test,docs; \
			poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls; \
			poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --without-urls --with dev,test,docs; \
			git add poetry.lock requirements.txt requirements-dev.txt; \
			git commit -m "add poetry.lock"; \
			poetry run pre-commit install; \
			poetry run pre-commit autoupdate; \
			git add .pre-commit-config.yaml; \
			git commit -m "update pre-commit hooks revision or tag"; \
		else \
			poetry install --with dev,test,docs; \
			poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls; \
			poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --without-urls --with dev,test,docs; \
		fi; \
		echo "Setup is complete."; \
	fi

.PHONY: sync
sync:
	@poetry install --with dev,test,docs

.PHONY: update
update:
	@poetry update --with dev,test,docs
	@poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls
	@poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --without-urls --with dev,test,docs
	@poetry run pre-commit autoupdate

.PHONY: check
check:
	-@poetry run ruff format ./src ./tests
	-@poetry run ruff check ./src ./tests --fix
	-@poetry run bandit -c pyproject.toml -r ./src ./tests

.PHONY: docs
docs:
	@poetry run sphinx-build -b html ./docs/source ./docs

.PHONY: build
build:
	@branch_name=$$(git rev-parse --abbrev-ref HEAD); \
	git checkout main; \
	poetry build; \
	git checkout $$branch_name

.PHONY: clean
clean:
	-@rm -rf .pytest_cache
	-@rm -rf dist
	-@rm -rf htmlcov
	-@find ./ -name "__pycache__" -exec rm -rf {} \;
	-@rm -rf .coverage
