# EKS setup

```bash
eksctl create cluster -f deploy/eks/cluster.yaml

# NVIDIA device plugin so pods can request nvidia.com/gpu
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm upgrade --install nvdp nvdp/nvidia-device-plugin -n kube-system \
  --set-json 'tolerations=[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}]'

# Monitoring + custom metrics for the HPAs
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter -n monitoring \
  -f k8s/monitoring/prometheus-adapter.yaml

# AWS Load Balancer Controller (for the gateway Ingress) and Cluster Autoscaler: follow the AWS docs for your account.

# Model repository in S3
aws s3 sync model_repository s3://YOUR_BUCKET/model_repository --exclude "*.npz" --exclude "*.json"

# Deploy
kubectl apply -f k8s/base/
kubectl apply -f k8s/triton/service.yaml -f k8s/triton/hpa.yaml
MODEL_REPOSITORY=s3://YOUR_BUCKET/model_repository ./scripts/blue_green.sh   # first deploy goes to blue
kubectl apply -f k8s/vllm/ -f k8s/gateway/ -f k8s/monitoring/servicemonitors.yaml -f k8s/monitoring/alerts.yaml
```

Cost note: two g5.xlarge on demand are roughly $2/hour. Scale the GPU node group to zero when you are not testing:
`eksctl scale nodegroup --cluster inference --name gpu-a10g --nodes 0 --nodes-min 0`.
