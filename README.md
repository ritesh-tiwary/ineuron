# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
refresh_token() {
  response=$(curl -s -X POST "$AUTH_URL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=refresh_token" \
    -d "client_id=$CLIENT_ID" \
    -d "client_secret=$CLIENT_SECRET" \
    -d "refresh_token=$REFRESH_TOKEN")

  new_token=$(echo "$response" | grep -oP '"access_token":\s*"\K[^"]+')

  if [[ -z "$new_token" ]]; then
    echo "Failed to refresh token"
    exit 1
  fi

  echo "Token refreshed successfully"
}
```
