"""
Background worker for processing translation tasks.
Runs in the same process as the FastAPI application.
"""

import asyncio
import logging
from typing import Optional, Set

import httpx

from app.config import get_settings
from app.models.task import TaskStatus
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)


async def send_callback(
    callback_url: str,
    task_id: str,
    status: str,
    result_url: Optional[str] = None,
    translate_source: Optional[str] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    发送回调通知。
    
    Args:
        callback_url: 回调URL
        task_id: 任务ID
        status: 任务状态 (SUCCESS/FAILED)
        result_url: 翻译结果URL
        translate_source: 翻译来源
        error_message: 错误信息
    
    Returns:
        回调是否成功
    """
    payload = {
        "taskId": task_id,
        "taskResult": result_url or "",
        "translateSource": translate_source or "",
        "errorMessage": error_message or "",
        "status": status,
    }
    
    # 打印请求地址和参数
    logger.info(f"[任务 {task_id}] 发送回调请求: {callback_url}")
    logger.info(f"[任务 {task_id}] 回调参数: {payload}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
            
        # 打印响应
        logger.info(f"[任务 {task_id}] 回调成功, 状态码: {response.status_code}")
        logger.info(f"[任务 {task_id}] 回调响应: {response.text}")
        return True
    except Exception as e:
        logger.error(f"[任务 {task_id}] 回调失败: {e}")
        return False


class BackgroundWorker:
    """
    Background worker that processes translation tasks.
    
    Polls the database for pending tasks and processes them concurrently.
    """
    
    def __init__(self, concurrency: int = 3, poll_interval: float = 2.0):
        """
        Initialize the background worker.
        
        Args:
            concurrency: Maximum number of concurrent tasks
            poll_interval: Seconds between polling for new tasks
        """
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._poll_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            logger.warning("Worker already running")
            return
        
        self._running = True
        self._semaphore = asyncio.Semaphore(self.concurrency)
        
        # Reset any tasks that were processing when server stopped
        reset_count = await TaskService.reset_processing_tasks()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} interrupted tasks")
        
        # Start the polling loop
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"Background worker started (concurrency={self.concurrency})")
    
    async def stop(self) -> None:
        """Stop the background worker gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel the polling task
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        # Wait for all processing tasks to complete
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} tasks to complete...")
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Background worker stopped")
    
    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                # Get pending tasks
                pending_tasks = await TaskService.get_pending_tasks(limit=self.concurrency)
                
                for task in pending_tasks:
                    if not self._running:
                        break
                    
                    # Try to claim the task
                    claimed = await TaskService.claim_task(task.task_id)
                    if claimed:
                        # Process in background
                        worker_task = asyncio.create_task(
                            self._process_task_wrapper(
                                task.task_id,
                                task.img_url,
                                task.source_language,
                                task.target_language,
                                task.enable_callback,
                                task.callback_env,
                            )
                        )
                        self._tasks.add(worker_task)
                        worker_task.add_done_callback(self._tasks.discard)
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _process_task_wrapper(
        self,
        task_id: str,
        img_url: str,
        source_language: str,
        target_language: str,
        enable_callback: bool = False,
        callback_env: Optional[str] = "test",
    ) -> None:
        """Wrapper to handle task processing with semaphore."""
        async with self._semaphore:
            await self._process_task(
                task_id, img_url, source_language, target_language,
                enable_callback, callback_env
            )
    
    async def _process_task(
        self,
        task_id: str,
        img_url: str,
        source_language: str,
        target_language: str,
        enable_callback: bool = False,
        callback_env: Optional[str] = "test",
    ) -> None:
        """
        Process a single translation task.
        
        This reuses the translation logic from translate.py.
        """
        import base64
        import time
        
        from app.services.aliyun_translate_service import get_aliyun_translate_service
        from app.services.online_translate_service import get_online_translate_service
        from app.api.translate import upload_image_to_gcp, is_rate_limit_error
        
        # 根据环境选择回调URL
        settings = get_settings()
        if callback_env == "prod":
            callback_url = settings.CALLBACK_URL_PROD
        else:
            callback_url = settings.CALLBACK_URL_TEST
        
        logger.info(f"[任务 {task_id}] 开始处理: {img_url}, callback={enable_callback}, env={callback_env}")
        
        # 下载请求头 (模拟浏览器)
        download_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.taobao.com/",
        }
        
        # Step 1: 下载图片 (带重试)
        image_bytes = None
        download_max_retries = 3
        last_download_error = ""
        
        for attempt in range(1, download_max_retries + 1):
            try:
                download_start = time.time()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(img_url, headers=download_headers)
                    response.raise_for_status()
                    image_bytes = response.content
                
                download_time = int((time.time() - download_start) * 1000)
                
                # 检测图片格式
                content_type = response.headers.get("content-type", "")
                if "jpeg" in content_type or "jpg" in content_type:
                    mime_type = "image/jpeg"
                elif "png" in content_type:
                    mime_type = "image/png"
                elif "gif" in content_type:
                    mime_type = "image/gif"
                elif "webp" in content_type:
                    mime_type = "image/webp"
                elif "bmp" in content_type:
                    mime_type = "image/bmp"
                else:
                    # 通过文件头检测
                    if image_bytes[:3] == b'\xff\xd8\xff':
                        mime_type = "image/jpeg"
                    elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                        mime_type = "image/png"
                    elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
                        mime_type = "image/gif"
                    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"  # 默认 jpeg
                
                # 生成 base64 (纯字符串，用于在线翻译 API)
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                logger.info(f"[任务 {task_id}] 下载图片成功: {len(image_bytes)} bytes, {download_time}ms, 格式: {mime_type}")
                break
                
            except Exception as e:
                last_download_error = str(e)
                logger.warning(f"[任务 {task_id}] 下载图片失败 (尝试 {attempt}/{download_max_retries}): {e}")
                if attempt < download_max_retries:
                    await asyncio.sleep(1 * attempt)  # 递增等待
        
        # 检查下载是否成功
        if image_bytes is None:
            error_msg = f"下载图片失败 (重试{download_max_retries}次): {last_download_error}"
            logger.error(f"[任务 {task_id}] {error_msg}")
            await TaskService.fail_task(task_id, error_msg)
            # 发送回调
            if enable_callback and callback_url:
                await send_callback(
                    callback_url=callback_url,
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=error_msg,
                )
            return
        
        # Step 2: Try Aliyun translation
        translate_source = None
        translate_time_ms = None
        use_online_fallback = False
        aliyun_error = ""
        
        try:
            aliyun_service = get_aliyun_translate_service()
            translate_start = time.time()
            
            # 使用 URL 方式调用阿里云 (避免 base64 扩展名识别问题)
            aliyun_result = aliyun_service.translate_image_url(
                image_url=img_url,
                source_language=source_language,
                target_language=target_language,
                field="e-commerce",
            )
            
            if aliyun_result.success:
                translate_time_ms = int((time.time() - translate_start) * 1000)
                translate_source = "aliyun"
                logger.info(f"[Task {task_id}] Aliyun success: {aliyun_result.final_image_url}, {translate_time_ms}ms")
                
                await TaskService.complete_task(
                    task_id=task_id,
                    result_url=aliyun_result.final_image_url,
                    translate_source=translate_source,
                    translate_time_ms=translate_time_ms,
                )
                
                # 发送回调
                if enable_callback and callback_url:
                    await send_callback(
                        callback_url=callback_url,
                        task_id=task_id,
                        status=TaskStatus.SUCCESS,
                        result_url=aliyun_result.final_image_url,
                        translate_source=translate_source,
                    )
                return
            else:
                aliyun_error = aliyun_result.message
                if is_rate_limit_error(aliyun_result.code, aliyun_result.message):
                    logger.warning(f"[Task {task_id}] Aliyun rate limited, falling back")
                else:
                    logger.warning(f"[Task {task_id}] Aliyun failed: {aliyun_error}")
                use_online_fallback = True
                
        except ValueError as e:
            logger.warning(f"[Task {task_id}] Aliyun not configured: {e}")
            aliyun_error = str(e)
            use_online_fallback = True
        except Exception as e:
            logger.error(f"[Task {task_id}] Aliyun error: {e}")
            aliyun_error = str(e)
            use_online_fallback = True
        
        # Step 3: Fallback to online translation with retry
        if use_online_fallback:
            online_service = get_online_translate_service()
            max_retries = 3
            last_error = ""
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"[Task {task_id}] Online attempt {attempt}/{max_retries}")
                    online_start = time.time()
                    online_result = await online_service.translate_image_base64_async(image_base64)
                    
                    if online_result.success:
                        translate_time_ms = int((time.time() - online_start) * 1000)
                        translate_source = "online"
                        
                        # 检查是否真的有翻译结果
                        if not online_result.image_base64:
                            # 没有翻译结果 (可能图片没有中文，translated=False)
                            logger.info(f"[任务 {task_id}] 在线翻译返回成功但无翻译图片, 返回原图")
                            # 返回原图 URL 作为结果
                            await TaskService.complete_task(
                                task_id=task_id,
                                result_url=img_url,  # 返回原图
                                translate_source="none",  # 标记为未翻译
                                translate_time_ms=translate_time_ms,
                            )
                            # 发送回调
                            if enable_callback and callback_url:
                                await send_callback(
                                    callback_url=callback_url,
                                    task_id=task_id,
                                    status=TaskStatus.SUCCESS,
                                    result_url=img_url,
                                    translate_source="none",
                                )
                            return
                        
                        logger.info(f"[任务 {task_id}] 在线翻译成功, {translate_time_ms}ms")
                        
                        # Upload to GCP
                        result_url = await upload_image_to_gcp(
                            online_result.image_base64,
                            f"translated_{task_id}.jpg"
                        )
                        
                        if result_url:
                            await TaskService.complete_task(
                                task_id=task_id,
                                result_url=result_url,
                                translate_source=translate_source,
                                translate_time_ms=translate_time_ms,
                            )
                            # 发送回调
                            if enable_callback and callback_url:
                                await send_callback(
                                    callback_url=callback_url,
                                    task_id=task_id,
                                    status=TaskStatus.SUCCESS,
                                    result_url=result_url,
                                    translate_source=translate_source,
                                )
                            return
                        else:
                            # Upload failed but translation succeeded
                            error_msg = "翻译成功但图片上传GCP失败"
                            await TaskService.fail_task(
                                task_id=task_id,
                                error_message=error_msg,
                                translate_source=translate_source,
                            )
                            # 发送回调
                            if enable_callback and callback_url:
                                await send_callback(
                                    callback_url=callback_url,
                                    task_id=task_id,
                                    status=TaskStatus.FAILED,
                                    translate_source=translate_source,
                                    error_message=error_msg,
                                )
                            return
                    else:
                        last_error = online_result.message
                        logger.warning(f"[任务 {task_id}] 在线翻译失败: {last_error}")
                        if attempt < max_retries:
                            await asyncio.sleep(1)
                            
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[Task {task_id}] Online error: {last_error}")
                    if attempt < max_retries:
                        await asyncio.sleep(1)
            
            # All retries failed
            error_msg = f"在线翻译重试{max_retries}次失败: {last_error} (阿里云: {aliyun_error})"
            await TaskService.fail_task(
                task_id=task_id,
                error_message=error_msg,
                translate_source="online",
            )
            # 发送回调
            if enable_callback and callback_url:
                await send_callback(
                    callback_url=callback_url,
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    translate_source="online",
                    error_message=error_msg,
                )


# Global worker instance
_worker: Optional[BackgroundWorker] = None


def get_worker() -> BackgroundWorker:
    """Get or create the global worker instance."""
    global _worker
    if _worker is None:
        settings = get_settings()
        _worker = BackgroundWorker(
            concurrency=settings.WORKER_CONCURRENCY,
            poll_interval=settings.WORKER_POLL_INTERVAL,
        )
    return _worker


async def start_worker() -> None:
    """Start the global background worker."""
    worker = get_worker()
    await worker.start()


async def stop_worker() -> None:
    """Stop the global background worker."""
    global _worker
    if _worker:
        await _worker.stop()
        _worker = None
