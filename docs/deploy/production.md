# Production Deployment Guide — Kawkab AI

> **Version**: 0.12.0  
> **Updated**: 2026-07-07  
> **Audience**: Platform / SRE engineers

---

## Architecture Overview

```
                          ┌─────────────────────────┐
                          │     Desktop Client       │
                          │  (PySide6 + QML + OMP)   │
                          └──────────┬──────────────┘
                                     │ HTTPS / WebSocket
                          ┌──────────▼──────────────┐
                          │     Nginx Ingress        │
                          │  (TLS, rate-limit, CORS) │
                          │  max_body_size: 500MB    │
                          └──────────┬──────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │   FastAPI Server × N     │
                          │  (uvicorn, asyncpg)      │
                          │  /health, /metrics       │
                          │  OAuth: Google / GitHub  │
                          └──────┬──────────────┬────┘
                                 │              │
                  ┌──────────────▼──┐   ┌──────▼──────────────┐
                  │   PostgreSQL 16 │   │   Celery Worker *   │
                  │  (asyncpg, SQL) │   │  (video, export,    │
                  │  migration_mgr  │   │   ML inference)     │
                  └─────────────────┘   └─────────────────────┘
                                              │
                                  ┌───────────▼───────────────┐
                                  │   Object Storage (S3/GCS) │
                                  │   (match video, exports)  │
                                  └───────────────────────────┘
```

> \* Celery worker is optional. The server handles sync requests inline; run workers only for heavyweight async tasks (video transcoding, batch export, ML model inference).

### Component Responsibilities

| Component | Stack | Role |
|-----------|-------|------|
| **Server** | FastAPI + uvicorn + asyncpg | REST API, WebSocket (real-time collab), auth (JWT + OAuth), sync, health checks |
| **Worker** | Celery + Redis/RabbitMQ broker | Async task queue for video processing, export generation, batch analytics (optional) |
| **PostgreSQL** | 16.x + pgvector (optional) | Primary store — users, teams, projects, events, sync log, contracts, shortlists |
| **Ingress** | nginx-ingress-controller | TLS termination, rate limiting, CORS, 500MB body limit for video uploads |
| **Object Storage** | S3-compatible (MinIO, AWS S3, GCS) | Raw match video files, exported reports, model weights |

---

## Prerequisites

### Cluster

- **Kubernetes** v1.28+ (tested on EKS, AKS, GKE, k3s)
- **Helm** 3.12+ (for ingress controller, Prometheus stack)
- **kubectl** v1.28+ with cluster context configured
- **Cert-Manager** (for automatic TLS certificate provisioning)
- **nginx-ingress-controller** (Helm: `ingress-nginx/ingress-nginx`)

### External Services

| Service | Version | Notes |
|---------|---------|-------|
| PostgreSQL | 16+ | Required. Use managed (RDS, Cloud SQL, Azure DB) or self-hosted via Helm |
| Object Storage | S3-compatible | Required for video uploads. MinIO works for self-hosted |
| Redis (opt.) | 7+ | Only needed if Celery worker is deployed |
| Container Registry | any | Docker Hub, ECR, GCR, GHCR — where the image is pushed |

### CLI Tools

- `kubectl` — cluster management
- `docker` — image build and push
- `envsubst` — template variable substitution
- `openssl` — secret generation
- `helm` — chart deployments (optional)

---

## Step-by-Step Deployment

### 1. Clone and Prepare

```bash
git clone https://github.com/yourorg/kawkab-ai.git
cd kawkab-ai
```

### 2. Configure Secrets

Generate strong secrets:

```bash
export KAWKAB_JWT_SECRET=$(openssl rand -hex 64)
export KAWKAB_DB_PASSWORD=$(openssl rand -hex 32)
export KAWKAB_COOKIE_SECRET=$(openssl rand -hex 32)
```

Create a `.env` file or export these in your CI/CD environment:

```bash
# Required
KAWKAB_JWT_SECRET=<generated>
# Plain postgresql:// -- cloud/database.py passes this straight to
# asyncpg.connect(), which does not understand the SQLAlchemy
# "postgresql+asyncpg://" dialect prefix.
KAWKAB_DB_URL=postgresql://kawkab:<password>@<host>:5432/kawkab
KAWKAB_DB_PASSWORD=<generated>

# Optional — OAuth providers
KAWKAB_GOOGLE_CLIENT_ID=
KAWKAB_GOOGLE_CLIENT_SECRET=
KAWKAB_GITHUB_CLIENT_ID=
KAWKAB_GITHUB_CLIENT_SECRET=

# Optional — data APIs
FOOTBALL_DATA_API_KEY=
APIFOOTBALL_API_KEY=

# Optional — LLM
OPENAI_API_KEY=
```

### 3. Set Up PostgreSQL

Deploy PostgreSQL 16 (example using Helm):

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm upgrade --install kawkab-db bitnami/postgresql \
  --namespace kawkab --create-namespace \
  --set postgresqlDatabase=kawkab \
  --set postgresqlUsername=kawkab \
  --set postgresqlPassword="$KAWKAB_DB_PASSWORD" \
  --set persistence.size=50Gi
```

Or point `KAWKAB_DB_URL` to your managed PostgreSQL instance.

### 4. Set Up Object Storage

Create an S3 bucket `kawkab-video-<env>` and configure access keys. The server expects these environment variables:

```bash
KAWKAB_STORAGE_BUCKET=kawkab-video-production
KAWKAB_STORAGE_ENDPOINT=https://s3.amazonaws.com
KAWKAB_STORAGE_ACCESS_KEY=
KAWKAB_STORAGE_SECRET_KEY=
```

### 5. Build and Push Docker Image

```bash
docker build --target server -t registry.example.com/kawkab/server:0.12.0 .
docker push registry.example.com/kawkab/server:0.12.0
```

### 6. Deploy via Script

```bash
# Deploy with default namespace "kawkab"
./scripts/deploy.sh

# Deploy to custom namespace, skip build
KAWKAB_JWT_SECRET=$KAWKAB_JWT_SECRET \
KAWKAB_DB_PASSWORD=$KAWKAB_DB_PASSWORD \
./scripts/deploy.sh --skip-build --namespace production

# Dry run (validate prerequisites only)
./scripts/deploy.sh --dry-run
```

### 7. Verify Deployment

```bash
# Check pods
kubectl -n kawkab get pods

# Health endpoint
kubectl -n kawkab port-forward svc/kawkab-service 8741:8741
curl http://localhost:8741/health

# Check HPA
kubectl -n kawkab get hpa

# View logs
kubectl -n kawkab logs -l app=kawkab,tier=server --tail=100
```

---

## Scaling Considerations

### Horizontal Scaling

| Trigger | Metric | Target | Action |
|---------|--------|--------|--------|
| CPU utilization | Resource | 70% | HPA adds replicas (2→10) |
| Memory utilization | Resource | 80% | HPA adds replicas |
| Concurrent connections | Custom | — | Requires KEDA or Prometheus adapter |

The HPA is pre-configured with `minReplicas: 2`, `maxReplicas: 10`, targeting 70% CPU. For high-traffic deployments (1000+ concurrent users), consider:

- Increasing `minReplicas` to 4
- Adding a memory metric target
- Enabling PDB (PodDisruptionBudget) for HA:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: kawkab-server-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: kawkab
      tier: server
```

### Vertical Scaling

- **Server pods**: 1 CPU / 2 GB RAM per replica handles ~200 concurrent connections
- **PostgreSQL**: Tune `shared_buffers` (25% of RAM), `work_mem` (32MB), `max_connections` (200)
- **Worker pods**: CPU-intensive (video transcoding, ML). 2 CPU / 4 GB RAM recommended

### Database Connection Pooling

The server uses asyncpg with connection pooling. Default pool size is 20. Adjust via:

```bash
KAWKAB_DB_POOL_MIN=5
KAWKAB_DB_POOL_MAX=40
```

For production, deploy PgBouncer as a sidecar or connection pooler:

```bash
helm upgrade --install pgbouncer eddycharly/pgbouncer \
  --namespace kawkab \
  --set postgresql.host=kawkab-db-postgresql \
  --set postgresql.port=5432 \
  --set postgresql.user=kawkab \
  --set postgresql.password="$KAWKAB_DB_PASSWORD"
```

---

## Monitoring Setup

### Prometheus + Grafana

1. **Install Prometheus Stack** (includes Grafana, AlertManager):

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install kawkab-monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

2. **Enable scraping** — the server exposes `/metrics` (disabled by design in `/infrastructure/nginx.conf` / ingress config). Ensure Prometheus scrapes pods directly via pod annotations:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8741"
  prometheus.io/path: "/metrics"
```

3. **Import Grafana dashboards**:

| Dashboard | UID | Source |
|-----------|-----|--------|
| Kawkab AI — Server | `kawkab-server` | `ops/grafana/kawkab-server.json` |
| PostgreSQL | `postgresql` | Grafana.com #9628 |
| Node Exporter | `node-exporter` | Grafana.com #1860 |

4. **Key alerts** (AlertManager config):

| Alert | Condition | Severity |
|-------|-----------|----------|
| ServerDown | `up{job="kawkab-server"} == 0` for 1m | critical |
| HighErrorRate | `sum(rate(http_requests_total{status=~"5.."}[5m])) > 0.05` | warning |
| HighLatency | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2` | warning |
| MigrationFailed | `migration_success == 0` | critical |
| DiskSpace | `node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} < 0.1` | critical |

### Structured Logging

All server logs are JSON-formatted via loguru. Collect with:

```yaml
# DaemonSet to ship logs (e.g., fluent-bit, vector, or promtail)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
spec:
  ...
```

For **Loki + Grafana**:

```bash
helm upgrade --install loki grafana/loki-stack \
  --namespace monitoring \
  --set grafana.enabled=false \
  --set prometheus.enabled=false
```

Then configure Promtail to scrape `/var/log/containers/*.log` with the `json` parser.

---

## Backup and Restore

### PostgreSQL

**Automated backups** via `pg_dump` with CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: kawkab-db-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: pg_dump
              image: postgres:16-alpine
              command:
                - sh
                - -c
                - |
                  pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
                    --format=custom --compress=9 \
                    | aws s3 cp - s3://kawkab-backups/db/$(date +%Y-%m-%d).dump
              envFrom:
                - secretRef:
                    name: kawkab-secrets
          restartPolicy: OnFailure
```

**Restore**:

```bash
aws s3 cp s3://kawkab-backups/db/2026-07-07.dump ./backup.dump
pg_restore --host=<host> --port=5432 \
  --username=kawkab --dbname=kawkab \
  --clean --if-exists \
  ./backup.dump
```

### Persistent Volume (Application Data)

The PVC `kawkab-data` stores local application state. Backup via Velero:

```bash
velero backup create kawkab-backup-$(date +%Y%m%d) \
  --include-namespaces kawkab \
  --include-resources persistentvolumeclaims,pods,deployments
```

Or snapshot directly with your cloud provider's CSI driver.

### Retention Policy

| Backup Type | Schedule | Retention |
|-------------|----------|-----------|
| DB (pg_dump) | Daily at 02:00 | 30 days |
| PVC snapshot | Daily | 7 days |
| Object Storage (video) | — | Client-managed (no automatic deletion) |
| Event sync log | — | 90 days (set `SYNC_LOG_RETENTION_DAYS=90`) |

---

## Security Checklist

### Pre-Deployment

- [ ] **JWT secret**: Minimum 64 bytes of cryptographically random data (`openssl rand -hex 64`)
- [ ] **DB password**: Minimum 32 bytes (`openssl rand -hex 32`)
- [ ] **Cookie secret**: Separate from JWT secret, minimum 32 bytes
- [ ] **API keys**: External service keys stored in Kubernetes Secrets, never in images or ConfigMaps
- [ ] **TLS certificate**: Provisioned via cert-manager or uploaded as `kawkab-tls` Secret
- [ ] **Network policies**: Default deny ingress, allow only from ingress controller

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: kawkab-deny-all
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: kawkab-allow-server
spec:
  podSelector:
    matchLabels:
      app: kawkab
      tier: server
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - port: 8741
  policyTypes:
    - Ingress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: kawkab-allow-postgres
spec:
  podSelector:
    matchLabels:
      tier: db
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: kawkab
      ports:
        - port: 5432
  policyTypes:
    - Ingress
```

### Runtime Protections

- [ ] **Pod Security Standards**: Restricted (enforced via namespace label `pod-security.kubernetes.io/enforce=restricted`)
- [ ] **Seccomp & AppArmor**: Default runtime profile; consider custom profiles for video processing containers
- [ ] **Read-only root filesystem**: Server containers run with `readOnlyRootFilesystem: false` (needs `/tmp` for model weights); isolate with emptyDir mount
- [ ] **Resource limits**: Enforced at deployment level; consider LimitRange for namespace-wide defaults
- [ ] **OAuth state validation**: Anti-CSRF via per-request state token (implemented in `server.py`)
- [ ] **Rate limiting**: `100 req/s` at ingress; per-endpoint rate limits in `security.py` (analysis=5/min, export=10/min, search=30/min)

### Data Security

- [ ] **Column sanitization**: SQL column names validated via `_sanitize_column_name` regex in `storage_service.py`
- [ ] **Input validation**: `CoordinateValidator` clamps event coordinates with warnings
- [ ] **Path traversal**: Blocked in `data_export_service.py`
- [ ] **XSS prevention**: HTML-escaped output via `sanitizeString` in frontend bridge
- [ ] **Encryption at rest**: Enable disk encryption on PVC (provider-specific) and RDS encryption
- [ ] **Encryption in transit**: TLS everywhere (ingress → server, server → DB via SSL if supported)

### Incident Response

1. **Detect**: Prometheus alert (`kawkab:HighErrorRate`) → PagerDuty/OpsGenie
2. **Triage**: `kubectl -n kawkab logs -l tier=server --tail=200`
3. **Mitigate**: Scale up, roll back, or fail over to replica
4. **Recover**: Restore from latest DB + PVC backup (see [Backup and Restore](#backup-and-restore))

---

## Maintenance

### Upgrading

```bash
# 1. Build new image
docker build --target server -t registry.example.com/kawkab/server:0.13.0 .
docker push registry.example.com/kawkab/server:0.13.0

# 2. Update kustomization
cd k8s && kustomize edit set image kawkab/server:0.13.0

# 3. Deploy
../scripts/deploy.sh --skip-build

# 4. Verify
kubectl -n kawkab rollout status deployment/kawkab-server
```

### Running Ad-Hoc Migration

```bash
kubectl -n kawkab delete job kawkab-migration-job --ignore-not-found
kubectl -n kawkab apply -f k8s/deployment.yaml
# Wait for completion
kubectl -n kawkab wait --for=condition=complete job/kawkab-migration-job --timeout=120s
```

### Tearing Down

```bash
kubectl delete namespace kawkab
# Remove PVCs separately if needed:
kubectl delete pvc -n kawkab kawkab-data
```

---

## References

| Resource | Location |
|----------|----------|
| Dockerfile | `Dockerfile` (multi-stage: `server`, `api`, `full`) |
| Docker Compose | `docker-compose.yml` (local dev with Postgres + Nginx) |
| Kubernetes manifests | `k8s/deployment.yaml`, `k8s/ingress.yaml`, `k8s/kustomization.yaml` |
| Deployment script | `scripts/deploy.sh` |
| Nginx config | `infrastructure/nginx.conf` |
| Environment template | `.env.example` |
| REST API docs | `docs/API.md` |
| Architecture overview | `docs/architecture.md` |
