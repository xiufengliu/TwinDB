#!/usr/bin/env bash
# TwinDB 演示：一键起库 / 起后端 / 起前端 / 后台全开 / 停止
# 用法见末尾 demo.sh help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

API_PID_FILE="${TMPDIR:-/tmp}/twindb-demo-api.pid"
WEB_PID_FILE="${TMPDIR:-/tmp}/twindb-demo-web.pid"
API_LOG="${TMPDIR:-/tmp}/twindb-demo-api.log"
WEB_LOG="${TMPDIR:-/tmp}/twindb-demo-web.log"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  # Prefer the python that actually has FastAPI installed.
  # (On some systems `python3` and `python` point to different environments.)
  if command -v python >/dev/null 2>&1; then
    if python -c "import fastapi" >/dev/null 2>&1; then
      PY=python
    fi
  fi
  if [[ -z "$PY" ]] && command -v python3 >/dev/null 2>&1; then
    if python3 -c "import fastapi" >/dev/null 2>&1; then
      PY=python3
    fi
  fi
  if [[ -z "$PY" ]]; then
    echo "需要 python/ python3 环境且包含 fastapi。请先跑: pip install -r requirements.txt"
    exit 1
  fi
fi

wait_for_docker() {
  local max="${1:-90}"
  local i=0
  echo "等待 Docker 守护进程…"
  while ! docker info >/dev/null 2>&1; do
    i=$((i + 1))
    if [[ "$i" -ge "$max" ]]; then
      echo "超时：请先手动打开 Docker Desktop，再重试。"
      exit 1
    fi
    sleep 2
  done
}

cmd_db() {
  if ! docker info >/dev/null 2>&1; then
    echo "正在启动 Docker Desktop…"
    open -a Docker 2>/dev/null || true
    wait_for_docker 120
  fi
  docker compose up -d
  echo "等待 PostgreSQL (5432)…"
  local i=0
  while ! pg_isready -h 127.0.0.1 -p 5432 -U twindb -d twindb >/dev/null 2>&1; do
    i=$((i + 1))
    if [[ "$i" -ge 60 ]]; then
      echo "数据库未就绪，请查看: docker compose logs"
      exit 1
    fi
    sleep 1
  done
  echo "数据库已就绪: postgresql://twindb:***@127.0.0.1:5432/twindb"
}

cmd_load() {
  echo "加载示例数据 (data/loader.py)…"
  "$PY" "$ROOT/data/loader.py"
  echo "完成。"
}

cmd_api() {
  echo "启动 API: http://0.0.0.0:${TWINDB_API_PORT:-8000}  (Ctrl+C 结束)"
  cd "$ROOT/core"
  exec "$PY" server.py
}

cmd_ui() {
  if [[ ! -d "$ROOT/web/node_modules" ]]; then
    echo "首次运行：正在 npm install…"
    (cd "$ROOT/web" && npm install)
  fi
  # Default to Mock mode to avoid TwinQL backend limitations during demos.
  # Use --real / env TWINDB_DEMO_REAL_API=true for real DB persistence.
  export VITE_USE_REAL_API="${VITE_USE_REAL_API:-${TWINDB_DEMO_REAL_API:-false}}"
  export VITE_API_URL="${VITE_API_URL:-http://localhost:8000}"
  echo "启动前端: ${VITE_API_URL} → API  (Ctrl+C 结束)"
  cd "$ROOT/web"
  exec npm run dev
}

cmd_all() {
  local load_data=0
  local real_api=0
  for a in "$@"; do
    [[ "$a" == "--load" ]] && load_data=1
    [[ "$a" == "--real" ]] && real_api=1
    [[ "$a" == "--mock" ]] && real_api=0
  done
  # Allow env override (useful for quick switches).
  if [[ "${TWINDB_DEMO_REAL_API:-}" == "true" ]]; then
    real_api=1
  fi

  cmd_db
  if [[ "$load_data" -eq 1 ]]; then
    cmd_load
  fi

  # 若已在跑，先停
  cmd_stop_quiet || true

  echo "后台启动 API → $API_LOG"
  nohup bash -c "cd \"$ROOT/core\" && exec \"$PY\" server.py" >>"$API_LOG" 2>&1 &
  echo $! >"$API_PID_FILE"

  sleep 2
  if ! kill -0 "$(cat "$API_PID_FILE")" 2>/dev/null; then
    echo "API 启动失败，见: $API_LOG"
    exit 1
  fi

  if [[ ! -d "$ROOT/web/node_modules" ]]; then
    (cd "$ROOT/web" && npm install)
  fi
  echo "后台启动 Web → $WEB_LOG"
  nohup env VITE_USE_REAL_API="$real_api" VITE_API_URL="${VITE_API_URL:-http://localhost:8000}" \
    bash -c "cd \"$ROOT/web\" && exec npm run dev" >>"$WEB_LOG" 2>&1 &
  echo $! >"$WEB_PID_FILE"

  echo ""
  echo "======== TwinDB 演示已后台运行 ========"
  echo "  API:    http://localhost:${TWINDB_API_PORT:-8000}  (健康检查 /api/health)"
  echo "  页面:   http://localhost:5173"
  echo "  API 日志: $API_LOG"
  echo "  Web 日志: $WEB_LOG"
  echo "  停止:    $ROOT/scripts/twindb-demo.sh stop"
  echo "========================================"
}

cmd_stop_quiet() {
  [[ -f "$API_PID_FILE" ]] && kill "$(cat "$API_PID_FILE")" 2>/dev/null || true
  [[ -f "$WEB_PID_FILE" ]] && kill "$(cat "$WEB_PID_FILE")" 2>/dev/null || true
  rm -f "$API_PID_FILE" "$WEB_PID_FILE"
}

cmd_stop() {
  echo "停止演示进程 (API + Web)…"
  cmd_stop_quiet
  read -r -p "是否同时关闭 Docker 数据库? [y/N] " ans
  if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
    docker compose down
    echo "已 docker compose down"
  fi
  echo "已停止。"
}

cmd_status() {
  echo "--- Docker ---"
  docker info >/dev/null 2>&1 && echo "Docker: 运行中" || echo "Docker: 未连接"
  docker compose ps 2>/dev/null || true
  echo "--- 端口 ---"
  pg_isready -h 127.0.0.1 -p 5432 -U twindb -d twindb 2>&1 || true
  curl -sf "http://127.0.0.1:${TWINDB_API_PORT:-8000}/api/health" && echo "" || echo "API: 无响应"
  echo "--- 后台 PID 文件 ---"
  [[ -f "$API_PID_FILE" ]] && echo "API pid $(cat "$API_PID_FILE")" || echo "API: 无"
  [[ -f "$WEB_PID_FILE" ]] && echo "Web pid $(cat "$WEB_PID_FILE")" || echo "Web: 无"
}

cmd_help() {
  cat <<'EOF'
TwinDB 演示脚本

  scripts/twindb-demo.sh db              # 仅启动 Docker 数据库 (会自动尝试打开 Docker Desktop)
  scripts/twindb-demo.sh load           # 灌示例数据 (需库已就绪)
  scripts/twindb-demo.sh api            # 前台运行 FastAPI (需另开终端跑 ui)
  scripts/twindb-demo.sh ui             # 前台运行 Vite 前端 (默认连真实 API)
  scripts/twindb-demo.sh all            # 后台同时起 db + api + web
  scripts/twindb-demo.sh all --load     # 同上，并执行 load
  scripts/twindb-demo.sh all --mock     # 后台起 db + api + mock 前端（适合跑 Scenario 1-4）
  scripts/twindb-demo.sh all --real --load  # 真实后端 + DB 写入（适合 Scenario 5）
  scripts/twindb-demo.sh stop           # 停后台 api/web；可选 down 数据库
  scripts/twindb-demo.sh status         # 快速检查

依赖: Docker Desktop, Python3+requirements.txt, Node.js+npm (前端)

首次请先:  pip install -r requirements.txt
EOF
}

main() {
  local sub="${1:-help}"
  shift || true
  case "$sub" in
    db)     cmd_db ;;
    load)   cmd_db; cmd_load ;;  # load 前确保库起来
    api)    cmd_api ;;
    ui)     cmd_ui ;;
    all)    cmd_all "$@" ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    help|-h|--help) cmd_help ;;
    *) echo "未知命令: $sub"; cmd_help; exit 1 ;;
  esac
}

main "$@"
