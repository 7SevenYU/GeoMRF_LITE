import logging
from pathlib import Path


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


LOG_DIR = _get_project_root() / "kg_construction" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """
    设置模块专用日志器

    Args:
        name: 日志器名称
        log_file: 日志文件名

    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 文件handler
    file_handler = logging.FileHandler(
        LOG_DIR / log_file,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
