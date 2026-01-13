"""
图片处理工具函数
"""

import base64
import io
import logging
from typing import Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# 注册 AVIF/HEIF 格式支持
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    logger.info("AVIF/HEIF 格式支持已启用")
except ImportError:
    logger.warning("pillow-heif 未安装，AVIF/HEIF 格式不可用")


def convert_to_jpeg(image_bytes: bytes) -> Tuple[bytes, str]:
    """
    将图片转换为 JPEG 格式。
    
    支持的输入格式: JPEG, PNG, GIF, BMP, WEBP, AVIF, HEIF
    
    Args:
        image_bytes: 原始图片字节数据
    
    Returns:
        Tuple[bytes, str]: (转换后的 JPEG 字节数据, 原始格式)
    """
    try:
        # 打开图片
        img = Image.open(io.BytesIO(image_bytes))
        original_format = img.format or "UNKNOWN"
        
        logger.info(f"图片格式: {original_format}, 尺寸: {img.size}, 模式: {img.mode}")
        
        # 如果已经是 JPEG，直接返回
        if original_format.upper() == "JPEG":
            return image_bytes, original_format
        
        # 转换为 RGB 模式 (处理 RGBA, P 等模式)
        if img.mode in ("RGBA", "LA", "P"):
            # 创建白色背景
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        # 保存为 JPEG
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        jpeg_bytes = output.getvalue()
        
        logger.info(f"图片已从 {original_format} 转换为 JPEG: {len(image_bytes)} -> {len(jpeg_bytes)} bytes")
        return jpeg_bytes, original_format
        
    except Exception as e:
        logger.error(f"图片格式转换失败: {e}")
        # 转换失败时返回原始数据
        return image_bytes, "UNKNOWN"


def convert_to_jpeg_base64(image_bytes: bytes) -> Tuple[str, str]:
    """
    将图片转换为 JPEG 格式并编码为 base64。
    
    Args:
        image_bytes: 原始图片字节数据
    
    Returns:
        Tuple[str, str]: (base64 编码的 JPEG 数据, 原始格式)
    """
    jpeg_bytes, original_format = convert_to_jpeg(image_bytes)
    base64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    return base64_str, original_format
