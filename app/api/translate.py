"""
Translation API endpoints.

使用阿里云翻译服务作为主要翻译方案，当阿里云限流时降级到在线翻译服务。
"""

import asyncio
import base64
import logging
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.aliyun_translate_service import (
    AliyunTranslateService,
    get_aliyun_translate_service,
)
from app.services.online_translate_service import (
    OnlineTranslateService,
    get_online_translate_service,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================
# 请求/响应模型
# ============================================

class TranslateRequest(BaseModel):
    """翻译请求"""
    imgUrl: str = Field(..., description="图片URL")
    sourceLanguage: str = Field(default="zh", description="源语言")
    targetLanguage: str = Field(default="en", description="目标语言")


class TranslateResponse(BaseModel):
    """翻译响应"""
    success: bool = Field(description="是否成功")
    translated: bool = Field(default=False, description="是否进行了翻译")
    # 使用的翻译服务
    service: Optional[str] = Field(default=None, description="使用的翻译服务 (aliyun/online)")
    # 翻译后图片URL
    imageUrl: Optional[str] = Field(default=None, description="翻译后图片URL")
    # 翻译API调用耗时(毫秒)
    timeMs: Optional[int] = Field(default=None, description="翻译API调用耗时(毫秒)")
    # 错误信息
    message: Optional[str] = Field(default=None, description="状态消息或错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok")
    version: str = Field(default="0.2.0")


# ============================================
# 阿里云限流错误码
# ============================================

# 阿里云常见限流/配额错误码
ALIYUN_RATE_LIMIT_CODES = {
    10001,  # 限流错误
    10002,  # 配额超限
    429,    # Too Many Requests
}

# 需要降级到在线翻译的错误关键词
RATE_LIMIT_KEYWORDS = [
    "限流",
    "rate limit",
    "throttl",
    "quota",
    "too many",
    "frequency",
]


def is_rate_limit_error(code: int, message: str) -> bool:
    """判断是否为限流错误"""
    if code in ALIYUN_RATE_LIMIT_CODES:
        return True
    
    message_lower = message.lower()
    for keyword in RATE_LIMIT_KEYWORDS:
        if keyword.lower() in message_lower:
            return True
    
    return False


# 图片上传接口配置 (从环境变量读取)


async def upload_image_to_gcp(image_base64: str, filename: str = "translated.jpg") -> Optional[str]:
    """
    将Base64图片上传到GCP，返回URL
    
    Args:
        image_base64: 图片的Base64编码
        filename: 文件名
    
    Returns:
        上传成功返回URL，失败返回None
    """
    try:
        # Base64解码为字节
        image_bytes = base64.b64decode(image_base64)
        
        # 构建multipart表单
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }
        
        # 从配置获取上传URL
        settings = get_settings()
        upload_url = settings.UPLOAD_IMAGE_URL
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(upload_url, files=files)
            response.raise_for_status()
        
        data = response.json()
        
        if data.get("code") == 0 and data.get("data", {}).get("url"):
            url = data["data"]["url"]
            logger.info(f"图片上传成功: {url}")
            return url
        else:
            logger.error(f"图片上传失败: {data}")
            return None
            
    except Exception as e:
        logger.error(f"图片上传异常: {e}")
        return None


# ============================================
# API 端点
# ============================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    健康检查端点。
    返回服务状态和版本。
    """
    return HealthResponse(status="ok", version="0.2.0")


@router.post(
    "/api/v1/translate",
    response_model=TranslateResponse,
    tags=["Translation"],
    summary="翻译图片",
    description="提供图片URL，翻译图片中的文字。优先使用阿里云翻译，限流时降级到在线翻译。"
)
async def translate_image(request: TranslateRequest) -> TranslateResponse:
    """
    翻译图片中的文字。
    
    处理流程:
    1. 从URL下载图片并转换为Base64
    2. 尝试使用阿里云翻译服务
    3. 如果阿里云限流，降级到在线翻译服务
    
    阿里云返回翻译后的图片URL，在线翻译返回Base64。
    """
    img_url = request.imgUrl
    source_lang = request.sourceLanguage
    target_lang = request.targetLanguage
    
    # Step 1: 下载图片并转换为Base64
    try:
        logger.info(f"下载图片: {img_url}")
        download_start_time = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(img_url)
            response.raise_for_status()
            image_bytes = response.content
        
        download_time_ms = int((time.time() - download_start_time) * 1000)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        logger.info(f"图片下载成功，大小: {len(image_bytes)} bytes, 下载耗时: {download_time_ms}ms")
        
    except httpx.TimeoutException:
        logger.error(f"下载图片超时: {img_url}")
        return TranslateResponse(
            success=False,
            message=f"下载图片超时: {img_url}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"下载图片HTTP错误: {e.response.status_code}")
        return TranslateResponse(
            success=False,
            message=f"下载图片失败: HTTP {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"下载图片失败: {e}")
        return TranslateResponse(
            success=False,
            message=f"下载图片失败: {str(e)}"
        )
    
    # Step 2: 尝试阿里云翻译
    use_online_fallback = False
    aliyun_error_message = ""
    translate_start_time = time.time()
    
    try:
        aliyun_service = get_aliyun_translate_service()
        
        logger.info(f"调用阿里云翻译: {source_lang} -> {target_lang}")
        aliyun_result = aliyun_service.translate_image_base64(
            image_base64=image_base64,
            source_language=source_lang,
            target_language=target_lang,
            field="e-commerce",  # 使用电商领域
        )
        
        if aliyun_result.success:
            # 阿里云翻译成功
            translate_time_ms = int((time.time() - translate_start_time) * 1000)
            logger.info(f"阿里云翻译成功: {aliyun_result.final_image_url}, 耗时: {translate_time_ms}ms")
            return TranslateResponse(
                success=True,
                translated=True,
                service="aliyun",
                imageUrl=aliyun_result.final_image_url,
                timeMs=translate_time_ms,
                message="Translation completed via Aliyun"
            )
        else:
            # 阿里云翻译失败，降级到在线翻译
            if is_rate_limit_error(aliyun_result.code, aliyun_result.message):
                logger.warning(f"阿里云限流，准备降级: code={aliyun_result.code}, message={aliyun_result.message}")
            else:
                logger.warning(f"阿里云翻译失败，准备降级: code={aliyun_result.code}, message={aliyun_result.message}")
            use_online_fallback = True
            aliyun_error_message = aliyun_result.message
    
    except ValueError as e:
        # 阿里云配置错误（如未设置AccessKey），直接使用在线翻译
        logger.warning(f"阿里云服务未配置: {e}")
        use_online_fallback = True
        aliyun_error_message = str(e)
    except Exception as e:
        # 其他异常，尝试降级
        logger.error(f"阿里云翻译异常: {e}")
        use_online_fallback = True
        aliyun_error_message = str(e)
    
    # Step 3: 降级到在线翻译 (带重试机制)
    if use_online_fallback:
        online_service = get_online_translate_service()
        max_retries = 3
        last_error = ""
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"调用在线翻译服务 (尝试 {attempt}/{max_retries})")
                # 重新计时，只记录online翻译的时间
                online_start_time = time.time()
                online_result = await online_service.translate_image_base64_async(image_base64)
                
                if online_result.success:
                    translate_time_ms = int((time.time() - online_start_time) * 1000)
                    logger.info(f"在线翻译成功: translated={online_result.translated}, 翻译耗时: {translate_time_ms}ms")
                    
                    # 将Base64图片上传到GCP
                    image_url = None
                    if online_result.image_base64:
                        upload_start_time = time.time()
                        image_url = await upload_image_to_gcp(
                            online_result.image_base64,
                            f"translated_{int(time.time())}.jpg"
                        )
                        upload_time_ms = int((time.time() - upload_start_time) * 1000)
                        if image_url:
                            logger.info(f"图片上传成功, 上传耗时: {upload_time_ms}ms")
                        else:
                            logger.warning(f"图片上传失败，但翻译已成功, 上传耗时: {upload_time_ms}ms")
                    
                    return TranslateResponse(
                        success=True,
                        translated=online_result.translated,
                        service="online",
                        imageUrl=image_url,
                        timeMs=translate_time_ms,
                        message=f"Translation completed via online service (fallback from: {aliyun_error_message})"
                    )
                else:
                    # 翻译返回失败，记录错误并重试
                    last_error = online_result.message
                    logger.warning(f"在线翻译失败 (尝试 {attempt}/{max_retries}): {last_error}")
                    if attempt < max_retries:
                        # 等待一小段时间后重试
                        await asyncio.sleep(1)
                        continue
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"在线翻译异常 (尝试 {attempt}/{max_retries}): {last_error}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
        
        # 所有重试都失败
        logger.error(f"在线翻译服务重试{max_retries}次后仍失败")
        return TranslateResponse(
            success=False,
            service="online",
            message=f"在线翻译服务重试{max_retries}次后失败: {last_error} (阿里云错误: {aliyun_error_message})"
        )
    
    # 不应该到达这里
    return TranslateResponse(
        success=False,
        message="未知错误"
    )
