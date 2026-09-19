#!/bin/bash
# NIPS — Network Intrusion Prevention System deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    command -v docker >/dev/null 2>&1 || { log_error "Docker required"; exit 1; }
    command -v docker compose >/dev/null 2>&1 || { log_error "docker compose plugin required"; exit 1; }
}

# Build image
build() {
    log_info "Building Docker image..."
    docker compose build api
}

# Start services
start() {
    log_info "Starting services..."
    
    # Create empty rules.json if missing (bind-mount source must exist)
    [ -f rules.json ] || touch rules.json
    
    docker compose up -d
    log_info "API available at http://localhost:8000"
}

# Stop services
stop() {
    log_info "Stopping services..."
    docker compose down
}

# View logs
logs() {
    docker compose logs -f api
}

# Health check
health() {
    # python3, not python: server hosts commonly ship python3 only, and a
    # missing alias here made the health check look broken rather than absent.
    curl -s http://localhost:8000/health | python3 -m json.tool
}

# Run verification tests
test() {
    # python3, not a pinned python3.13: the image is python:3.12-slim, which
    # ships python3/python only — the pinned name made every one of these
    # commands fail with "command not found" while looking like a passing suite.
    # verify_kernel_rules is intentionally absent: the container is unprivileged,
    # so it has no NET_ADMIN and its iptables assertions cannot run there.
    log_info "Running verification tests..."
    docker compose exec api python3 /app/scripts/verify_engine_module.py
    docker compose exec api python3 /app/scripts/verify_interception_module.py
    docker compose exec api python3 /app/scripts/verify_block_lifecycle.py
    docker compose exec api python3 /app/scripts/verify_live_exposed_bugs.py
    docker compose exec api python3 /app/scripts/verify_fpr_regression.py
    docker compose exec api python3 /app/scripts/verify_features_module.py
    docker compose exec api python3 /app/scripts/verify_data_module.py
    docker compose exec api python3 /app/scripts/verify_management_plane.py
}

# Help
usage() {
    echo "Usage: $0 {build|start|stop|restart|logs|health|test}"
    echo "  build   - Build Docker image"
    echo "  start   - Start services"
    echo "  stop    - Stop services"
    echo "  restart - Restart services"
    echo "  logs    - View logs"
    echo "  health  - Health check"
    echo "  test    - Run verification tests"
}

# Main
case "$1" in
    build) check_dependencies; build ;;
    start) check_dependencies; start ;;
    stop) stop ;;
    restart) stop; start ;;
    logs) logs ;;
    health) health ;;
    test) test ;;
    *) usage; exit 1 ;;
esac
