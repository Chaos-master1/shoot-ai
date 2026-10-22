# Getting Started

## Installation

```bash
git clone https://github.com/wizard404error/shoot-ai
cd kawkab-ai
pip install -e ".[dev,tactical,graph]"
```

For GPU support:

```bash
pip install -e ".[gpu,dev,tactical,graph]"
```

## Running the Desktop App

```bash
python -m kawkab
```

## Starting the Cloud Server

```bash
export KAWKAB_JWT_SECRET="your-strong-secret-here"
python -m uvicorn kawkab.cloud.server:app --host 0.0.0.0 --port 8741
```

## Using the REST API

```python
from kawkab.api.client import KawkabClient
client = KawkabClient(base_url="http://localhost:8741")
matches = await client.list_matches()
```

## OAuth Setup

Set environment variables for providers:

```bash
export KAWKAB_GOOGLE_CLIENT_ID="your-google-client-id"
export KAWKAB_GOOGLE_CLIENT_SECRET="your-google-client-secret"
export KAWKAB_GITHUB_CLIENT_ID="your-github-client-id"
export KAWKAB_GITHUB_CLIENT_SECRET="your-github-client-secret"
```
