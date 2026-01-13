#!/usr/bin/env python3
"""
测试在线翻译服务
"""

import asyncio
import base64
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from app.services.online_translate_service import get_online_translate_service


async def main():
    # 读取图片
    image_path = "test_06.jpg"
    
    print(f"读取图片: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    print(f"图片大小: {len(image_bytes)} bytes")
    
    # 编码为 base64
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
    print(f"image_base64: {result.image_base64[:100] if result.image_base64 else None}...")
    
    if result.raw_response:
        print(f"\n原始响应:")
        # 不打印 image 字段 (太长)
        raw = {k: v for k, v in result.raw_response.items() if k != "image"}
        print(json.dumps(raw, ensure_ascii=False, indent=2))
    
    # 保存翻译后的图片
    if result.image_base64:
        output_path = "test_06_translated.jpg"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result.image_base64))
        print(f"\n翻译后图片已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
