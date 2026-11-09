#!/usr/bin/env bash
# 用法：./scripts/run.sh    （读取 .env 里的 HOST/PORT）
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec uvicorn app.main:get_app --factory --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
