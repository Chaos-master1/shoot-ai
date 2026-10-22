# Kawkab AI

Professional football analytics platform — 100% private, offline, free.

## Quick Start

```bash
pip install -e ".[dev,tactical,graph]"
python -m kawkab
```

## Key Capabilities

- **Player & ball tracking** via YOLO + BoT-SORT
- **50+ analytical metrics** (xG, VAEP, xT, pitch control, pressing, formations)
- **Cloud sync** with JWT auth, teams, and collaboration
- **OAuth sign-in** with Google, GitHub, Apple
- **Real-time pipeline** with dual async producer/consumer
- **REST API** (FastAPI) with 25+ endpoints
- **Webhook system** with HMAC signing and retry
- **RBAC** with 5 roles and 20+ permissions
- **PostgreSQL adapter** with asyncpg pool + SQLite fallback
- **Model monitoring** with drift detection and auto-retrain
- **Medical tracking** (injury DB, rehab plans, concussion protocol)
- **Docker deployment** with PostgreSQL + nginx

## Guides

- **Analyst workflow** (import → data quality → validation → report): [`analyst-guide.md`](analyst-guide.md)
- **Elite deployment for club IT** (Postgres, SAML SSO, RBAC, audit): [`elite-deployment.md`](elite-deployment.md)
- Validation reports: [`validation/`](validation/)
- Model cards: [`model-cards.md`](model-cards.md)
- Deployment basics: [`deploy/`](deploy/)
