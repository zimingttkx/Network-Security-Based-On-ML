#!/bin/bash
# Network Security ML - 部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    command -v docker >/dev/null 2>&1 || { log_error "需要安装Docker"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log_error "需要安装docker-compose"; exit 1; }
}

# 构建镜像
build() {
    log_info "构建Docker镜像..."
    docker build -t network-security-ml:latest .
}

# 启动服务
start() {
    log_info "启动服务..."
    docker-compose up -d
    log_info "服务已启动: http://localhost:8000"
}

# 停止服务
stop() {
    log_info "停止服务..."
    docker-compose down
}

# 查看日志
logs() {
    docker-compose logs -f app
}

# 健康检查
health() {
    curl -s http://localhost:8000/health | python -m json.tool
}

# 运行测试
test() {
    log_info "运行测试..."
    docker-compose exec app python -m pytest tests/ -v
}

# 帮助信息
usage() {
    echo "用法: $0 {build|start|stop|restart|logs|health|test}"
    echo "  build   - 构建Docker镜像"
    echo "  start   - 启动服务"
    echo "  stop    - 停止服务"
    echo "  restart - 重启服务"
    echo "  logs    - 查看日志"
    echo "  health  - 健康检查"
    echo "  test    - 运行测试"
}

# 主函数
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
