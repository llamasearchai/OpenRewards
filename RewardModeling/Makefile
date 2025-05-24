# Makefile for Reward Modeling Engineering Platform (RMEP)
# Production-ready commands for development, testing, and deployment

.PHONY: help install install-dev test test-coverage lint format clean build deploy docs run-api run-frontend docker-build docker-run setup-env

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Project variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := reward-modeling-platform
PYTHON_DIR := python
TAURI_DIR := tauri
RUST_DIR := rust

# Docker variables
DOCKER_IMAGE_NAME := rmep
DOCKER_TAG := latest
DOCKER_REGISTRY := your-registry.com

help: ## Show this help message
	@echo "$(BLUE)Reward Modeling Engineering Platform (RMEP)$(RESET)"
	@echo "$(BLUE)===========================================$(RESET)"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# Installation commands
install: ## Install the package and dependencies
	@echo "$(BLUE)Installing RMEP...$(RESET)"
	cd $(PYTHON_DIR) && $(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	cd $(PYTHON_DIR) && $(PIP) install -e ".[dev,all]"
	pre-commit install
	@echo "$(GREEN)Development setup complete!$(RESET)"

install-full: ## Install all optional dependencies
	@echo "$(BLUE)Installing with all optional dependencies...$(RESET)"
	cd $(PYTHON_DIR) && $(PIP) install -e ".[all]"
	@echo "$(GREEN)Full installation complete!$(RESET)"

# Environment setup
setup-env: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv venv
	@echo "$(YELLOW)Activate virtual environment with: source venv/bin/activate$(RESET)"
	@echo "$(YELLOW)Then run: make install-dev$(RESET)"

# Testing commands
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ --cov=reward_modeling --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (exclude slow integration tests)
	@echo "$(BLUE)Running fast tests...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ -v -m "integration"

# Code quality commands
lint: ## Run linting checks
	@echo "$(BLUE)Running linting...$(RESET)"
	cd $(PYTHON_DIR) && flake8 reward_modeling tests
	cd $(PYTHON_DIR) && mypy reward_modeling

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	cd $(PYTHON_DIR) && black reward_modeling tests
	cd $(PYTHON_DIR) && isort reward_modeling tests

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	cd $(PYTHON_DIR) && black --check reward_modeling tests
	cd $(PYTHON_DIR) && isort --check-only reward_modeling tests

# Development server commands
run-api: ## Run the FastAPI development server
	@echo "$(BLUE)Starting FastAPI server...$(RESET)"
	cd $(PYTHON_DIR) && uvicorn reward_modeling.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Run the Tauri frontend development server
	@echo "$(BLUE)Starting Tauri frontend...$(RESET)"
	cd $(TAURI_DIR) && npm run tauri dev

run-monitoring: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.utils.monitoring --port 3000

# Training and evaluation commands
train-example: ## Run example training
	@echo "$(BLUE)Running example training...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.scripts.train \
		--config configs/training/reward_model_example.yaml \
		--output-dir ./outputs/example_training

evaluate-example: ## Run example evaluation
	@echo "$(BLUE)Running example evaluation...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.scripts.evaluate \
		--model-path ./outputs/example_training/final_model \
		--dataset-path ./data/eval_dataset.jsonl \
		--output-dir ./outputs/example_evaluation

# Data processing commands
process-data: ## Process and validate dataset
	@echo "$(BLUE)Processing dataset...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.scripts.data_processing \
		--input-path ./data/raw_data.jsonl \
		--output-path ./data/processed_data.jsonl \
		--validate

create-synthetic-data: ## Create synthetic dataset for testing
	@echo "$(BLUE)Creating synthetic dataset...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -c "from reward_modeling.data.dataset import create_synthetic_preference_data; dataset = create_synthetic_preference_data(n_samples=1000); dataset.save('./data/synthetic_data.jsonl')"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -p 8000:8000 -p 3000:3000 $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-run-dev: ## Run Docker container in development mode
	@echo "$(BLUE)Running Docker container in development mode...$(RESET)"
	docker run -it -v $(PWD):/workspace -p 8000:8000 -p 3000:3000 $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) bash

docker-push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image to registry...$(RESET)"
	docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# Docker Compose commands
compose-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting services with Docker Compose...$(RESET)"
	docker-compose up -d

compose-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(RESET)"
	docker-compose down

compose-logs: ## View logs from all services
	@echo "$(BLUE)Viewing logs...$(RESET)"
	docker-compose logs -f

compose-rebuild: ## Rebuild and restart services
	@echo "$(BLUE)Rebuilding services...$(RESET)"
	docker-compose up -d --build

# Documentation commands
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd docs && make html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	cd docs && make clean

# Build and packaging commands
build: ## Build the package
	@echo "$(BLUE)Building package...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m build

build-rust: ## Build Rust components
	@echo "$(BLUE)Building Rust components...$(RESET)"
	cd $(RUST_DIR) && cargo build --release

build-tauri: ## Build Tauri application
	@echo "$(BLUE)Building Tauri application...$(RESET)"
	cd $(TAURI_DIR) && npm run tauri build

build-all: build build-rust build-tauri ## Build all components

# Deployment commands
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	./scripts/deploy.sh staging

deploy-production: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	./scripts/deploy.sh production

# Database commands
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	cd $(PYTHON_DIR) && alembic upgrade head

db-create-migration: ## Create new database migration
	@echo "$(BLUE)Creating new migration...$(RESET)"
	@read -p "Migration name: " name; \
	cd $(PYTHON_DIR) && alembic revision --autogenerate -m "$$name"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all database data!$(RESET)"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		cd $(PYTHON_DIR) && alembic downgrade base && alembic upgrade head; \
		echo "$(GREEN)Database reset complete$(RESET)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(RESET)"; \
	fi

# Monitoring and maintenance commands
health-check: ## Run health checks
	@echo "$(BLUE)Running health checks...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.scripts.health_check

logs: ## View application logs
	@echo "$(BLUE)Viewing logs...$(RESET)"
	tail -f logs/reward_modeling.log

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m reward_modeling.scripts.benchmark

# Security commands
security-check: ## Run security vulnerability checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	cd $(PYTHON_DIR) && safety check
	cd $(PYTHON_DIR) && bandit -r reward_modeling

# Cleanup commands
clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	cd $(PYTHON_DIR) && rm -rf build/ dist/
	cd $(RUST_DIR) && cargo clean
	@echo "$(GREEN)Cleanup complete!$(RESET)"

clean-all: clean ## Clean everything including virtual environments
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	rm -rf venv/
	rm -rf node_modules/
	cd $(TAURI_DIR) && rm -rf node_modules/
	@echo "$(GREEN)Deep cleanup complete!$(RESET)"

# Release commands
version-bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	cd $(PYTHON_DIR) && bump2version patch

version-bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	cd $(PYTHON_DIR) && bump2version minor

version-bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(RESET)"
	cd $(PYTHON_DIR) && bump2version major

release: ## Create a new release
	@echo "$(BLUE)Creating release...$(RESET)"
	./scripts/release.sh

# CI/CD helpers
ci-setup: ## Setup CI environment
	@echo "$(BLUE)Setting up CI environment...$(RESET)"
	$(PIP) install -e ".[dev,all]"

ci-test: ## Run tests in CI environment
	@echo "$(BLUE)Running CI tests...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ --cov=reward_modeling --cov-report=xml

ci-lint: ## Run linting in CI environment
	@echo "$(BLUE)Running CI linting...$(RESET)"
	cd $(PYTHON_DIR) && flake8 reward_modeling tests
	cd $(PYTHON_DIR) && mypy reward_modeling
	cd $(PYTHON_DIR) && black --check reward_modeling tests
	cd $(PYTHON_DIR) && isort --check-only reward_modeling tests

# Development utilities
shell: ## Start an interactive Python shell with project context
	@echo "$(BLUE)Starting interactive shell...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -c "import reward_modeling; print('RMEP modules loaded. Access via reward_modeling.*'); import IPython; IPython.start_ipython()"

jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(RESET)"
	cd $(PYTHON_DIR) && jupyter notebook --ip=0.0.0.0 --port=8888

profile: ## Run profiling on example training
	@echo "$(BLUE)Running profiling...$(RESET)"
	cd $(PYTHON_DIR) && $(PYTHON) -m cProfile -o profile_output.prof -m reward_modeling.scripts.train --config configs/training/profile_config.yaml

# Quick development workflow
dev-setup: install-dev create-synthetic-data ## Quick development setup
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "$(YELLOW)You can now run:$(RESET)"
	@echo "  make run-api       (start API server)"
	@echo "  make test          (run tests)"
	@echo "  make train-example (run example training)"

# Production workflow
prod-check: ci-lint ci-test security-check ## Run all production readiness checks
	@echo "$(GREEN)Production readiness checks complete!$(RESET)

# Show environment info
info: ## Show environment information
	@echo "$(BLUE)Environment Information$(RESET)"
	@echo "======================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Docker not installed')"
	@echo "Docker Compose version: $$(docker-compose --version 2>/dev/null || echo 'Docker Compose not installed')" 