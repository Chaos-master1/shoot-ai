# Elite Deployment Guide

For the club's IT department. Covers running Kawkab as a managed, multi-user,
auditable deployment: real Postgres, SSO, RBAC, and the audit trail.
For the analyst's daily workflow see `docs/analyst-guide.md`.

---

## 1. Architecture in one paragraph

Kawkab runs as a desktop application (PySide6 shell + web UI) backed by a
local SQLite database by default. For club deployments the same code runs
against **PostgreSQL** via the asyncpg adapter, behind **nginx**, with
**JWT** sessions, **SAML SSO** for analyst identities, **RBAC** for roles,
and an append-only **audit trail** for every sensitive data action.

## 2. Postgres (the club database)

### Quick start (docker)

```bash
docker run -d --name kawkab-pg \
  -e POSTGRES_USER=kawkab -e POSTGRES_PASSWORD=<secret> -e POSTGRES_DB=kawkab \
  -v kawkab-pgdata:/var/lib/postgresql/data \
  -p 5432:5432 postgres:16
```

Set `KAWKAB_DB_URL=postgresql://kawkab:<secret>@localhost:5432/kawkab` for
the app. On first connect the adapter applies `src/kawkab/migrations/pg_schema.sql`
plus the numbered migration chain — **idempotently**, so restarts and
re-deploys are safe. For production, prefer a managed Postgres or the
compose file in `docs/deploy/docker.md`.

The Postgres schema is kept behaviorally in sync with the SQLite schema by
an integration test suite (`tests/unit/test_postgres_storage.py`) that runs
against a real server:

```bash
docker run -d --rm --name kawkab-pg-test \
  -e POSTGRES_USER=kawkab -e POSTGRES_PASSWORD=kawkab -e POSTGRES_DB=kawkab \
  -p 5433:5432 postgres:16

KAWKAB_DB_URL="postgresql://kawkab:kawkab@127.0.0.1:5433/kawkab" \
KAWKAB_PG_TESTS=1 \
  pytest tests/unit/test_postgres_storage.py -v
```

(37 tests: schema application, upsert/dedup round-trips, tracking-import
save/get, match-context updates, soft-delete parity.) The suite is skipped
unless both variables are set, so default CI stays hermetic.

## 3. SSO (SAML)

Kawkab integrates with the club's identity provider over **SAML 2.0**
(OneLogin SDK; configured via API, stored per-environment). Environment
variables:

| Variable | Meaning |
|---|---|
| `KAWKAB_SAML_IDP_ENTITY_ID` | IdP entity/issuer URI |
| `KAWKAB_SAML_IDP_SSO_URL` | IdP SSO redirect URL |
| `KAWKAB_SAML_IDP_CERT` | IdP x509 certificate (PEM) |

Endpoints (all under the cloud API):

- `GET /auth/saml/status` — whether the SDK is available and which IdPs are configured
- `POST /auth/saml/configure` — register/replace an IdP (admin only)
- `POST /auth/saml/callback` — consume the IdP's SAMLResponse, resolve or create the local user, issue a JWT

A club Okta/Entra/Azure AD SAML app is configured by pointing it at the
callback and registering its metadata via `/auth/saml/configure`. Google,
GitHub, and Apple **OAuth** sign-in are also available for clubs that don't
run SAML.

## 4. RBAC

Five roles ship out of the box (admin, analyst, coach, medical, viewer)
with 20+ fine-grained permissions; API endpoints enforce role gates and
the desktop UI mirrors them. Assign roles from the admin panel after SSO
provisioning — SSO authenticates, RBAC authorizes.

## 5. Audit trail

Every sensitive data action (imports, exports, deletions, model runs,
permission changes) is written to an append-only audit log with actor,
action, timestamp, and provenance (vendor, file, and external IDs for
imported data). The API exposes it read-only:

- `GET /api/v1/audit/events` — filter by actor, action, date range

Data provenance is stored per match (vendor, external IDs, import run,
**data-quality report**) — when someone asks "where did this number come
from?", the answer is a query, not folklore.

## 6. Data quality at import

Tracking imports produce a data-quality report (frame completeness,
duplicate timestamps, player-count anomalies, out-of-bounds coordinates)
stored with the import provenance and surfaced in the desktop UI. Club
policy can gate on it: a match whose quality report shows sustained
anomalies should be re-imported before its analytics are trusted.

## 7. Backups and data residency

- All data stays on club infrastructure — Kawkab is offline-first; no
  telemetry, no cloud dependency for the analytics path.
- Backup = the Postgres volume (`docs/deploy/production.md` covers
  scheduled dumps); the audit log ships with it.
- The SQLite fallback remains available for single-analyst laptops;
  role-gated cloud sync is opt-in per team.

## 8. Health and monitoring

`kawkab benchmark --module <name>` measures analytical throughput on the
deployment's own hardware — run it after install and after upgrades to
detect regressions. Model drift monitoring and auto-retrain hooks are
built in (see `docs/model-cards.md` for each model's monitoring story).
