#!/usr/bin/env bash
# Zero-downtime blue/green deploy for Triton.
#
#   1. Find the live colour from the `triton` Service selector.
#   2. Render and apply the idle colour with the new image / model repository.
#   3. Wait for the rollout: every pod must pass its readiness probe, which with
#      --strict-readiness means every model loaded and warmed up.
#   4. Smoke-test the idle colour directly through its own Service.
#   5. Only then patch the live Service selector to the new colour. Traffic shifts
#      in one step; nothing is ever sent to a pod that is not ready.
#   6. Keep the old colour running (scaled down to MIN_OLD) for instant rollback:
#        ./scripts/blue_green.sh rollback
#
# Env: MODEL_REPOSITORY (required), TRITON_VERSION (default 24.08), REPLICAS (default 2),
#      NAMESPACE (default inference), MIN_OLD (default 1), TIMEOUT (default 900s)
set -euo pipefail

NS=${NAMESPACE:-inference}
TIMEOUT=${TIMEOUT:-900s}
live() { kubectl -n "$NS" get svc triton -o jsonpath='{.spec.selector.color}'; }
other() { [ "$1" = blue ] && echo green || echo blue; }

switch_to() {
  kubectl -n "$NS" patch svc triton --type merge -p "{\"spec\":{\"selector\":{\"app\":\"triton\",\"color\":\"$1\"}}}"
  echo "live colour is now $1"
}

if [ "${1:-}" = "rollback" ]; then
  current=$(live); previous=$(other "$current")
  kubectl -n "$NS" patch hpa "triton-$previous" --type merge -p "{\"spec\":{\"minReplicas\":${REPLICAS:-2}}}" || true
  kubectl -n "$NS" scale deploy "triton-$previous" --replicas="${REPLICAS:-2}"
  kubectl -n "$NS" rollout status deploy "triton-$previous" --timeout="$TIMEOUT"
  switch_to "$previous"
  exit 0
fi

: "${MODEL_REPOSITORY:?set MODEL_REPOSITORY, e.g. s3://bucket/model_repository}"
export MODEL_REPOSITORY TRITON_VERSION=${TRITON_VERSION:-24.08} REPLICAS=${REPLICAS:-2}

current=$(live 2>/dev/null || echo "")
if kubectl -n "$NS" get deploy "triton-${current:-none}" >/dev/null 2>&1; then
  target=$(other "$current")
else
  target=blue   # first deployment
fi
export COLOR=$target
echo "live: ${current:-<none>}  deploying: $target"

envsubst < k8s/triton/deployment.tmpl.yaml | kubectl apply -f -

echo "waiting for every $target pod to pass readiness (models loaded + warmed up)..."
if ! kubectl -n "$NS" rollout status deploy "triton-$target" --timeout="$TIMEOUT"; then
  echo "rollout failed; live traffic untouched on ${current:-<none>}" >&2
  exit 1
fi

echo "smoke test against triton-$target"
kubectl -n "$NS" run "smoke-$target-$RANDOM" --rm -i --restart=Never --image=curlimages/curl:8.8.0 --quiet -- \
  sh -c "set -e
    curl -sf http://triton-$target:8000/v2/health/ready
    for m in fraud_mlp digits_mlp risk_mlp; do curl -sf http://triton-$target:8000/v2/models/\$m/ready; done" \
  || { echo "smoke test failed; live traffic untouched" >&2; exit 1; }

kubectl -n "$NS" patch hpa "triton-$target" --type merge -p "{\"spec\":{\"minReplicas\":$REPLICAS}}" || true
switch_to "$target"

if [ -n "$current" ] && [ "$current" != "$target" ]; then
  # lower the HPA floor first, otherwise it would scale the idle colour straight back up
  kubectl -n "$NS" patch hpa "triton-$current" --type merge -p "{\"spec\":{\"minReplicas\":${MIN_OLD:-1}}}" || true
  kubectl -n "$NS" scale deploy "triton-$current" --replicas="${MIN_OLD:-1}"
  echo "kept triton-$current at ${MIN_OLD:-1} replica(s) for rollback"
fi
