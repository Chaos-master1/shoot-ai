#!/usr/bin/env bash
# =============================================================================
# Kawkab AI — Production Deployment Script
# Usage:  ./scripts/deploy.sh [--skip-build] [--dry-run] [--namespace <ns>]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-kawkab}"
IMAGE_REPO="${IMAGE_REPO:-kawkab/server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN=false
SKIP_BUILD=false

# ─── Color helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ─── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --dry-run)    DRY_RUN=true;    shift ;;
    --namespace)  NAMESPACE="$2";  shift 2 ;;
    *)            fail "Unknown arg: $1" ;;
  esac
done

# ─── Prerequisites ────────────────────────────────────────────────────────────
info "Checking prerequisites..."
for cmd in kubectl docker envsubst; do
  command -v "$cmd" &>/dev/null || fail "Missing prerequisite: $cmd"
done
ok "All prerequisites present."

# Required env vars (fail fast if missing)
: "${KAWKAB_JWT_SECRET:?Must set KAWKAB_JWT_SECRET}"
: "${KAWKAB_DB_PASSWORD:?Must set KAWKAB_DB_PASSWORD}"

# ─── Build ─────────────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
  info "Building Docker image ${IMAGE_REPO}:${IMAGE_TAG} ..."
  docker build \
    --target server \
    -t "${IMAGE_REPO}:${IMAGE_TAG}" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$SCRIPT_DIR"
  ok "Image built: ${IMAGE_REPO}:${IMAGE_TAG}"
fi

if [[ "$DRY_RUN" == true ]]; then
  info "DRY RUN — skipping deployment steps."
  exit 0
fi

# ─── Push to registry (customize for your registry) ────────────────────────────
# Uncomment and configure for your container registry:
# info "Pushing image to registry..."
# docker push "${IMAGE_REPO}:${IMAGE_TAG}"
# ok "Image pushed."

# ─── Create namespace if needed ────────────────────────────────────────────────
kubectl get namespace "$NAMESPACE" &>/dev/null || {
  kubectl create namespace "$NAMESPACE"
  ok "Created namespace: $NAMESPACE"
}

# ─── Apply Kubernetes manifests via Kustomize ─────────────────────────────────
info "Applying Kubernetes manifests to namespace ${NAMESPACE}..."

# Patch image tag into kustomization
cd "$SCRIPT_DIR/k8s"
kubectl -n "$NAMESPACE" apply -k .

# Ensure secrets exist with real values
# (KAWKAB_DB_URL uses plain postgresql:// -- cloud/database.py passes it
# straight to asyncpg.connect(), which rejects the SQLAlchemy
# "postgresql+asyncpg://" dialect suffix this used to have.)
kubectl -n "$NAMESPACE" create secret generic kawkab-secrets \
  --from-literal=KAWKAB_JWT_SECRET="$KAWKAB_JWT_SECRET" \
  --from-literal=KAWKAB_DB_URL="postgresql://kawkab:${KAWKAB_DB_PASSWORD}@postgres:5432/kawkab" \
  --from-literal=KAWKAB_DB_PASSWORD="$KAWKAB_DB_PASSWORD" \
  --dry-run=client -o yaml | kubectl -n "$NAMESPACE" apply -f -
ok "Manifests applied."

# ─── Run migration job ─────────────────────────────────────────────────────────
info "Running database migration job..."
kubectl -n "$NAMESPACE" delete job kawkab-migration-job --ignore-not-found=true
kubectl -n "$NAMESPACE" apply -f "$SCRIPT_DIR/k8s/deployment.yaml" 2>/dev/null || true
kubectl -n "$NAMESPACE" wait --for=condition=complete --timeout=120s job/kawkab-migration-job || {
  warn "Migration job did not complete in time. Checking logs..."
  pod=$(kubectl -n "$NAMESPACE" get pods -l tier=migration --field-selector=status.phase!=Running -o name 2>/dev/null | head -1)
  if [ -n "$pod" ]; then
    kubectl -n "$NAMESPACE" logs "$pod"
  fi
}
ok "Migration complete."

# ─── Watch rollout ─────────────────────────────────────────────────────────────
info "Watching server rollout..."
kubectl -n "$NAMESPACE" rollout status deployment/kawkab-server --timeout=180s || {
  fail "Server rollout failed. Rolling back..."
  kubectl -n "$NAMESPACE" rollout undo deployment/kawkab-server
  exit 1
}
ok "Server rollout successful."

info "Watching worker rollout (if present)..."
kubectl -n "$NAMESPACE" rollout status deployment/kawkab-celery-worker --timeout=120s || {
  warn "Worker rollout did not complete — check celery configuration."
}

# ─── Verify ────────────────────────────────────────────────────────────────────
info "Verifying deployment..."
sleep 5
HEALTH_URL="http://$(kubectl -n "$NAMESPACE" get svc kawkab-service -o jsonpath='{.spec.clusterIP}'):8741/health"
HEALTH_STATUS=$(kubectl -n "$NAMESPACE" run curl-test --image=curlimages/curl --restart=Never --rm -- -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
if [[ "$HEALTH_STATUS" == "200" ]]; then
  ok "Service health check passed (HTTP ${HEALTH_STATUS})."
else
  warn "Health check returned HTTP ${HEALTH_STATUS}. Check pod logs."
fi

# ─── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Kawkab AI deployed successfully!            ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║${NC}  Namespace:  ${CYAN}$NAMESPACE${NC}"
echo -e "${GREEN}║${NC}  Image:      ${CYAN}${IMAGE_REPO}:${IMAGE_TAG}${NC}"
echo -e "${GREEN}║${NC}  Server:     ${CYAN}kawkab-service:8741${NC}"
echo -e "${GREEN}║${NC}  Replicas:   ${CYAN}$(kubectl -n "$NAMESPACE" get deploy kawkab-server -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo '?')${NC}"
echo -e "${GREEN}║${NC}  Ingress:    ${CYAN}https://kawkab.example.com${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
