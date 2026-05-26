#!/bin/bash

# --- CONFIGURATION ---
SERVER_USER="boygioi15"
SERVER_IP="172.16.1.30"              # Your old laptop's IP
SERVER_DIR="~/ai_server"  # Where the code lives on the server
# ---------------------

echo "🚀 1. Syncing selected code to server (app/ and requirements.txt)..."
rsync -avz --delete \
  --include='app/***' \
  --include='requirements.txt' \
  --include='makefile' \
  --exclude='*' \
  ./ $SERVER_USER@$SERVER_IP:$SERVER_DIR

echo "🛑 2. Killing old server instance..."
# pkill finds the process running your app and kills it. '|| true' prevents the script from crashing if it's already stopped.
ssh $SERVER_USER@$SERVER_IP "pkill -9 -f 'uvicorn' || true"

echo "🏃 3. Starting the server in the background..."
ssh $SERVER_USER@$SERVER_IP "(cd $SERVER_DIR && nohup make start > server.log 2>&1) > /dev/null 2>&1 &"
echo "✅ Deployment complete!"