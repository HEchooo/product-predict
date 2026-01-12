"""
SQLAlchemy model for translation tasks.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, Integer, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TaskStatus:
    """Task status constants."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TranslateTask(Base):
    """
    Translation task model.
    
    Stores async translation task information for persistence and recovery.
    """
    __tablename__ = "translate_tasks"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Task identifier (UUID, exposed to clients)
    task_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    
    # Task status: PENDING, PROCESSING, SUCCESS, FAILED
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=TaskStatus.PENDING, index=True)
    
    # Input parameters
    img_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_language: Mapped[str] = mapped_column(String(10), nullable=False, default="zh")
    target_language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    
    # Result (populated on success)
    result_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Translation metadata
    translate_source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # aliyun / online
    translate_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Error info (populated on failure)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Callback configuration
    enable_callback: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)  # 使用Integer兼容MySQL
    callback_env: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, default="test")  # test 或 prod
    
    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_status_created", "status", "created_at"),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "taskId": self.task_id,
            "status": self.status,
            "imgUrl": self.img_url,
            "sourceLanguage": self.source_language,
            "targetLanguage": self.target_language,
            "resultUrl": self.result_url,
            "translateSource": self.translate_source,
            "translateTimeMs": self.translate_time_ms,
            "errorMessage": self.error_message,
            "retryCount": self.retry_count,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def __repr__(self) -> str:
        return f"<TranslateTask(task_id={self.task_id}, status={self.status})>"
