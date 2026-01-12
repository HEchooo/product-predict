"""
日志配置模块。

支持按天和按大小切割日志。
日志文件格式: app_2026-01-12.log, app_2026-01-12_01.log, app_2026-01-12_02.log
"""

import logging
import os
import glob
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path


class DailyRotatingFileHandler(RotatingFileHandler):
    """
    按天 + 按大小切割的日志处理器。
    
    - 每天使用新的日志文件: app_2026-01-12.log
    - 当文件超过 maxBytes 时，切换到: app_2026-01-12_01.log, app_2026-01-12_02.log
    - 自动清理超过 backupDays 天的历史日志
    """
    
    def __init__(
        self,
        log_dir: str,
        base_name: str = "app",
        max_bytes: int = 50 * 1024 * 1024,  # 50MB
        backup_count: int = 10,  # 每天最多10个文件
        backup_days: int = 30,  # 保留30天
        encoding: str = "utf-8",
    ):
        self.log_dir = Path(log_dir)
        self.base_name = base_name
        self.backup_days = backup_days
        self._current_date = None
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取当前日期的日志文件
        self._update_filename()
        
        super().__init__(
            filename=str(self._current_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        
        # 清理旧日志
        self._cleanup_old_logs()
    
    def _update_filename(self):
        """更新当前日志文件名"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date != today:
            self._current_date = today
            self._current_file = self.log_dir / f"{self.base_name}_{today}.log"
    
    def shouldRollover(self, record):
        """检查是否需要切换文件"""
        # 检查日期是否变化
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date != today:
            return True
        return super().shouldRollover(record)
    
    def doRollover(self):
        """执行日志切割"""
        # 检查日期是否变化
        today = datetime.now().strftime("%Y-%m-%d")
        if self._current_date != today:
            # 日期变化，切换到新文件
            if self.stream:
                self.stream.close()
                self.stream = None
            self._update_filename()
            self.baseFilename = str(self._current_file)
            self.stream = self._open()
        else:
            # 大小超限，使用默认的轮转逻辑
            super().doRollover()
    
    def rotation_filename(self, default_name):
        """生成切割后的文件名: app_2026-01-12_01.log"""
        # 默认名字是 app_2026-01-12.log.1
        # 转换为 app_2026-01-12_01.log
        base = default_name.rsplit(".log.", 1)[0]
        if ".log." in default_name:
            num = default_name.rsplit(".log.", 1)[1]
            return f"{base}_{num.zfill(2)}.log"
        return default_name
    
    def _cleanup_old_logs(self):
        """清理超过 backup_days 天的日志"""
        cutoff_date = datetime.now() - timedelta(days=self.backup_days)
        pattern = str(self.log_dir / f"{self.base_name}_*.log")
        
        for log_file in glob.glob(pattern):
            try:
                # 从文件名提取日期
                filename = os.path.basename(log_file)
                # app_2026-01-12.log or app_2026-01-12_01.log
                date_str = filename.replace(f"{self.base_name}_", "").split(".log")[0].split("_")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    os.remove(log_file)
                    logging.debug(f"已删除旧日志: {log_file}")
            except (ValueError, OSError):
                pass


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 10,  # 每天最多10个文件
    backup_days: int = 30,  # 保留30天
) -> None:
    """
    配置日志系统。
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        max_bytes: 单个日志文件最大大小 (字节)
        backup_count: 每天最多保留的日志文件数
        backup_days: 保留历史日志的天数
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 清除已有的处理器 (避免重复)
    root_logger.handlers.clear()
    
    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # 2. 应用日志 - 按天+按大小切割
    app_handler = DailyRotatingFileHandler(
        log_dir=log_dir,
        base_name="app",
        max_bytes=max_bytes,
        backup_count=backup_count,
        backup_days=backup_days,
    )
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(app_handler)
    
    # 3. 错误日志 - 单独存放
    error_handler = DailyRotatingFileHandler(
        log_dir=log_dir,
        base_name="error",
        max_bytes=max_bytes,
        backup_count=backup_count,
        backup_days=backup_days,
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)
    
    # 设置第三方库日志级别 (减少噪音)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    logging.info(f"日志系统初始化完成, 目录: {log_path.absolute()}, 单文件最大: {max_bytes // 1024 // 1024}MB")
