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
	@echo "  test    Tests and coverage measurements"
	@echo "  build   Build the project to python package"
	@echo "  docs    Build the project to python package"
	@echo "  serve   Build the project to python package"
	@echo "  clean   Clean up generated files"

.PHONY: setup
setup:
	@if [ -d .git ]; then \
		echo "Setup is already done."; \
	else \
		if command -v git &> /dev/null; then \
			git init; \
			git commit --allow-empty -m "initial commit"; \
			git checkout -b develop; \
			git add .; \
			git commit -m "add template folder"; \
			git tag v0.1.0;\
			uv sync --group lint --group test --group docs --no-cache; \
			git add uv.lock _version.py; \
			git commit -m "run 'uv sync' to initialize project's synchronization"; \
			uv run pre-commit install; \
			uv run pre-commit autoupdate; \
			git add .pre-commit-config.yaml; \
			git commit --no-verify -m "update pre-commit hooks revision or tag"; \
		else \
			uv sync --group lint --group test --group docs --no-cache; \
		fi; \
		echo "Setup is complete."; \
	fi

.PHONY: sync
sync:
	@uv sync --group lint --group test --group docs --no-cache
	@uv run pre-commit install

.PHONY: update
update:
	@uv lock --no-cache
	@uv run pre-commit autoupdate

.PHONY: check
check:
	-@uv run ruff format ./src ./tests
	-@uv run ruff check --fix ./src ./tests
	-@uv run bandit -c pyproject.toml -r ./src ./tests

.PHONY: test
test:
	@uv run pytest --cov=src --cov-report=term-missing -n 1 ./tests/

.PHONY: build
build:
	@branch_name=$$(git rev-parse --abbrev-ref HEAD); \
	git checkout main; \
	uv build; \
	git checkout $$branch_name

.PHONY: docs
docs:
	@uv run mkdocs build

.PHONY: serve
serve:
	@uv run mkdocs serve

.PHONY: clean
clean:
	-@find ./ -type f -name "*.py[co]" -delete
	-@find ./ -type d -name "__pycache__" -delete
	-@find ./ -type d -name "dist" -delete
	-@find ./ -type d -name "htmlcov" -delete
	-@find ./ -type f -name ".coverage" -delete
