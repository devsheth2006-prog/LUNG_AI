#!/bin/sh
set -e
case "$1" in
  ui)  exec streamlit run app/Home.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 ;;
  api) exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 ;;
  test) exec python -m pytest tests -q ;;
  *)   exec "$@" ;;
esac
