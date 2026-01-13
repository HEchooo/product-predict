#!/usr/bin/env python3
"""
测试从 CDN 下载图片后使用在线翻译服务
"""

import asyncio
import base64
import json
import logging
import httpx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from app.services.online_translate_service import get_online_translate_service


async def main():
    # CDN 图片地址
    img_url = "https://img.alicdn.com/imgextra/i4/2524999298/O1CN01BXHv2I2IYXi0cfQoC_!!2524999298.jpg"
    
    # 下载请求头 (模拟浏览器)
    download_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.taobao.com/",
    }
    
    # 下载图片
    print(f"下载图片: {img_url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(img_url, headers=download_headers)
        response.raise_for_status()
        image_bytes = response.content
        content_type = response.headers.get("content-type", "")
    
    print(f"图片大小: {len(image_bytes)} bytes")
    print(f"Content-Type: {content_type}")
    
    # 保存下载的图片
    with open("downloaded_image.jpg", "wb") as f:
        f.write(image_bytes)
    print("已保存下载的图片: downloaded_image.jpg")
    
    # 如果是 AVIF/HEIF/WEBP，转换为 JPEG
    from app.utils.image_utils import convert_to_jpeg_base64
    
    if "avif" in content_type or "heif" in content_type or "webp" in content_type:
        print(f"\n检测到 {content_type} 格式，转换为 JPEG...")
        image_base64, original_format = convert_to_jpeg_base64(image_bytes)
        print(f"转换完成: {original_format} -> JPEG")
    else:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    print(f"Base64 长度: {len(image_base64)}")
    
    # 调用在线翻译服务
    print("\n调用在线翻译服务...")
    service = get_online_translate_service()
    result = await service.translate_image_base64_async(image_base64)
    
    # 打印结果
    print("\n========== 响应结果 ==========")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"translated: {result.translated}")
    print(f"has_product: {result.has_product}")
    print(f"image_base64 长度: {len(result.image_base64) if result.image_base64 else 0}")
    
    if result.raw_response:
        print(f"\n原始响应 (不含 image):")
        raw = {k: v for k, v in result.raw_response.items() if k != "image"}
        print(json.dumps(raw, ensure_ascii=False, indent=2))
    
    # 保存翻译后的图片
    if result.image_base64:
        output_path = "downloaded_image_translated.jpg"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result.image_base64))
        print(f"\n翻译后图片已保存: {output_path}")
    else:
        print("\n⚠️ 没有返回翻译后的图片 (translated=False)")


if __name__ == "__main__":
    asyncio.run(main())
