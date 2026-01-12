"""
Task service for managing translation tasks.
Provides CRUD operations for TranslateTask.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.models.task import TranslateTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing translation tasks."""
    
    @staticmethod
    async def create_task(
        img_url: str,
        source_language: str = "zh",
        target_language: str = "en",
        enable_callback: bool = False,
        callback_env: Optional[str] = "test",
    ) -> TranslateTask:
        """
        Create a new translation task.
        
        Args:
            img_url: URL of the image to translate
            source_language: Source language code
            target_language: Target language code
            enable_callback: Whether to call callback URL when task completes
            callback_env: Callback environment (test or prod)
        
        Returns:
            Created TranslateTask
        """
        task_id = str(uuid.uuid4())
        
        async with get_async_session() as session:
            task = TranslateTask(
                task_id=task_id,
                status=TaskStatus.PENDING,
                img_url=img_url,
                source_language=source_language,
                target_language=target_language,
                enable_callback=enable_callback,
                callback_env=callback_env,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            
            logger.info(f"Created task: {task_id}, callback={enable_callback}, env={callback_env}")
            return task
    
    @staticmethod
    async def get_task(task_id: str) -> Optional[TranslateTask]:
        """
        Get task by task_id.
        
        Args:
            task_id: UUID of the task
        
        Returns:
            TranslateTask or None if not found
        """
        async with get_async_session() as session:
            result = await session.execute(
                select(TranslateTask).where(TranslateTask.task_id == task_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_pending_tasks(limit: int = 10) -> List[TranslateTask]:
        """
        Get pending tasks ordered by creation time.
        
        Args:
            limit: Maximum number of tasks to return
        
        Returns:
            List of pending TranslateTask
        """
        async with get_async_session() as session:
            result = await session.execute(
                select(TranslateTask)
                .where(TranslateTask.status == TaskStatus.PENDING)
                .order_by(TranslateTask.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def claim_task(task_id: str) -> bool:
        """
        Claim a task for processing (atomically set status to PROCESSING).
        
        Args:
            task_id: UUID of the task
        
        Returns:
            True if claimed successfully, False if already claimed
        """
        async with get_async_session() as session:
            result = await session.execute(
                update(TranslateTask)
                .where(
                    TranslateTask.task_id == task_id,
                    TranslateTask.status == TaskStatus.PENDING
                )
                .values(
                    status=TaskStatus.PROCESSING,
                    started_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Claimed task: {task_id}")
                return True
            return False
    
    @staticmethod
    async def complete_task(
        task_id: str,
        result_url: str,
        translate_source: str,
        translate_time_ms: int,
    ) -> None:
        """
        Mark task as completed successfully.
        
        Args:
            task_id: UUID of the task
            result_url: URL of the translated image
            translate_source: Translation source (aliyun/online)
            translate_time_ms: Translation time in milliseconds
        """
        async with get_async_session() as session:
            await session.execute(
                update(TranslateTask)
                .where(TranslateTask.task_id == task_id)
                .values(
                    status=TaskStatus.SUCCESS,
                    result_url=result_url,
                    translate_source=translate_source,
                    translate_time_ms=translate_time_ms,
                    completed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
            logger.info(f"Task completed: {task_id}")
    
    @staticmethod
    async def fail_task(
        task_id: str,
        error_message: str,
        translate_source: Optional[str] = None,
    ) -> None:
        """
        Mark task as failed.
        
        Args:
            task_id: UUID of the task
            error_message: Error description
            translate_source: Translation source that was attempted
        """
        async with get_async_session() as session:
            await session.execute(
                update(TranslateTask)
                .where(TranslateTask.task_id == task_id)
                .values(
                    status=TaskStatus.FAILED,
                    error_message=error_message,
                    translate_source=translate_source,
                    completed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
            logger.error(f"Task failed: {task_id} - {error_message}")
    
    @staticmethod
    async def reset_processing_tasks() -> int:
        """
        Reset PROCESSING tasks to PENDING (for recovery after restart).
        
        Returns:
            Number of tasks reset
        """
        async with get_async_session() as session:
            result = await session.execute(
                update(TranslateTask)
                .where(TranslateTask.status == TaskStatus.PROCESSING)
                .values(
                    status=TaskStatus.PENDING,
                    started_at=None,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
            
            count = result.rowcount
            if count > 0:
                logger.info(f"Reset {count} processing tasks to pending")
            return count
    
    @staticmethod
    async def increment_retry(task_id: str) -> None:
        """
        Increment retry count for a task.
        
        Args:
            task_id: UUID of the task
        """
        async with get_async_session() as session:
            # Get current task
            result = await session.execute(
                select(TranslateTask).where(TranslateTask.task_id == task_id)
            )
            task = result.scalar_one_or_none()
            
            if task:
                await session.execute(
                    update(TranslateTask)
                    .where(TranslateTask.task_id == task_id)
                    .values(
                        retry_count=task.retry_count + 1,
                        updated_at=datetime.utcnow(),
                    )
                )
                await session.commit()
