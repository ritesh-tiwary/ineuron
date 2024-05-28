# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
#!/bin/bash

# Configuration: Replace these with your actual values
KUBE_CONTEXT="your-kube-context"
KUBE_NAMESPACE="your-namespace"
AUTH_URL="https://your-auth-server.com/token"
CLIENT_ID="your-client-id"
CLIENT_SECRET="your-client-secret"
REFRESH_TOKEN="your-refresh-token"
KUBECONFIG_PATH="$HOME/.kube/config"

# Function to get new token
refresh_token() {
  new_token=$(curl -s -X POST "$AUTH_URL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=refresh_token" \
    -d "client_id=$CLIENT_ID" \
    -d "client_secret=$CLIENT_SECRET" \
    -d "refresh_token=$REFRESH_TOKEN" | jq -r '.access_token')

  if [[ -z "$new_token" ]]; then
    echo "Failed to refresh token"
    exit 1
  fi

  echo "Token refreshed successfully"
}

# Function to update kubeconfig with the new token
update_kubeconfig() {
  kubectl config set-credentials "$KUBE_CONTEXT" --token="$new_token"
}

# Main script
refresh_token
update_kubeconfig

echo "Kubeconfig updated successfully with new token"

# You can also optionally switch context or namespace if needed
kubectl config use-context "$KUBE_CONTEXT"
kubectl config set-context --current --namespace="$KUBE_NAMESPACE"

echo "Context set to $KUBE_CONTEXT and namespace set to $KUBE_NAMESPACE"


```
