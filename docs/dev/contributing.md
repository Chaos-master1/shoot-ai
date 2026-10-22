# Contributing

## Setup

```bash
git clone https://github.com/wizard404error/shoot-ai
cd kawkab-ai
pip install -e ".[dev,tactical,graph,test,cloud]"
```

## Code Style

- Ruff for linting and formatting
- Line length: 100
- Type hints required for new code
- No comments in source code unless explaining a non-obvious design decision

## Testing

```bash
# Run all unit tests
python -m pytest tests/unit/ -q

# Run specific test file
python -m pytest tests/unit/test_xg_model.py -v

# Run with coverage
python -m pytest --cov=src/kawkab
```

## Branch Strategy

- `main` — stable, release-ready
- `develop` — integration branch
- Feature branches: `feat/description`

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/unit/ -q`)
- [ ] Ruff lint passes (`ruff check src/ tests/`)
- [ ] Type checks pass (`mypy src/kawkab/`)
- [ ] New code has tests
- [ ] No hardcoded secrets or debug code
