# Building the Documentation

## Prerequisites

Install tilus with dev dependencies:

```bash
pip install -e ".[dev]"
```

No GPU is required to build the documentation.

## Build

```bash
cd docs
make html
```

The output will be in `docs/build/html/`. Open `docs/build/html/index.html` in a browser to view.

## Live preview

```bash
cd docs
sphinx-autobuild source build/html
```

Opens a local server at `http://localhost:8000` that auto-rebuilds and refreshes the browser when you edit RST files.

## Clean build

```bash
cd docs
make clean
make html
```

## Versioned deployment

Docs are deployed to GitHub Pages via the `gh-pages` branch. The CI handles this automatically:

- **On merge to `main`**: docs are built and deployed to the `latest/` directory.
- **On tag push** (`v*`): docs are built and deployed to a versioned directory (e.g., `v0.1.0/`).

The root URL redirects to `latest/`.
