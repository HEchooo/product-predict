"""
Online Translation Service - 在线图片翻译API服务

使用外部API服务进行图片翻译。
API端点: https://image-api.alvinclub.com/image-optimizer/no-auth/online/translate

该服务接收Base64编码的图片，返回翻译后的图片。
"""

import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# 默认API配置
DEFAULT_API_URL = "https://image-api.alvinclub.com/image-optimizer/no-auth/online/translate"
DEFAULT_TIMEOUT = 120.0  # 图片翻译可能耗时较长


@dataclass
class OnlineTranslateResult:
    """在线翻译结果"""
    success: bool
    message: str = ""
    translated: bool = False
    has_product: bool = False
    image_base64: Optional[str] = None
    raw_response: Optional[dict] = None


class OnlineTranslateService:
    """
    在线图片翻译服务
    
    调用外部API翻译图片中的文字。
    
    使用示例:
        service = OnlineTranslateService()
        
        # 方式1: 使用Base64
        result = service.translate_image_base64("iVBORw0KGgo...")
        
        # 方式2: 使用本地文件
        result = service.translate_image_file("/path/to/image.jpg")
        
        # 方式3: 使用字节数据
        result = service.translate_image_bytes(image_bytes)
        
        if result.success and result.translated:
            # 保存翻译后的图片
            with open("output.jpg", "wb") as f:
                f.write(base64.b64decode(result.image_base64))
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        初始化在线翻译服务
        
        Args:
            api_url: API端点URL (默认使用内置URL)
            timeout: 请求超时时间(秒)
        """
        self.api_url = api_url or DEFAULT_API_URL
        self.timeout = timeout
    
    def translate_image_base64(self, image_base64: str) -> OnlineTranslateResult:
        """
        通过Base64翻译图片
        
        Args:
            image_base64: 图片的Base64编码
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        try:
            logger.info(f"调用在线翻译API: {self.api_url}")
            
            # 发送请求
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    json={"image": image_base64},
                    headers={"Content-Type": "application/json"},
                )
            
            # 检查HTTP状态
            if response.status_code != 200:
                logger.error(f"API请求失败: HTTP {response.status_code}")
                return OnlineTranslateResult(
                    success=False,
                    message=f"HTTP错误: {response.status_code}",
                    raw_response={"status_code": response.status_code, "text": response.text},
                )
            
            # 解析响应
            data = response.json()
            
            # 检查响应状态
            status = data.get("status", "")
            if status != "success":
                logger.warning(f"API返回非成功状态: {status}")
                return OnlineTranslateResult(
                    success=False,
                    message=data.get("message", f"状态: {status}"),
                    raw_response=data,
                )
            
            # 构建成功结果
            result = OnlineTranslateResult(
                success=True,
                message=data.get("message", "Translation completed"),
                translated=data.get("translated", False),
                has_product=data.get("has_product", False),
                image_base64=data.get("image"),
                raw_response=data,
            )
            
            logger.info(
                f"在线翻译完成: translated={result.translated}, "
                f"has_product={result.has_product}"
            )
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"API请求超时: {e}")
            return OnlineTranslateResult(
                success=False,
                message=f"请求超时: {self.timeout}秒",
            )
        except httpx.RequestError as e:
            logger.error(f"API请求错误: {e}")
            return OnlineTranslateResult(
                success=False,
                message=f"请求错误: {str(e)}",
            )
        except Exception as e:
            logger.error(f"翻译过程发生异常: {e}")
            return OnlineTranslateResult(
                success=False,
                message=str(e),
            )
    
    def translate_image_bytes(self, image_bytes: bytes) -> OnlineTranslateResult:
        """
        翻译图片字节数据
        
        Args:
            image_bytes: 图片字节数据
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return self.translate_image_base64(image_base64)
    
    def translate_image_file(self, file_path: str) -> OnlineTranslateResult:
        """
        翻译本地图片文件
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        path = Path(file_path)
        if not path.exists():
            return OnlineTranslateResult(
                success=False,
                message=f"文件不存在: {file_path}",
            )
        
        # 检查文件大小 (限制10MB)
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            return OnlineTranslateResult(
                success=False,
                message=f"文件大小超过10MB限制: {file_size / 1024 / 1024:.2f}MB",
            )
        
        # 读取文件
        with open(path, "rb") as f:
            image_bytes = f.read()
        
        return self.translate_image_bytes(image_bytes)
    
    async def translate_image_base64_async(self, image_base64: str) -> OnlineTranslateResult:
        """
        异步翻译 - 通过Base64
        
        Args:
            image_base64: 图片的Base64编码
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        try:
            logger.info(f"调用在线翻译API(异步): {self.api_url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json={"image": image_base64},
                    headers={"Content-Type": "application/json"},
                )
            
            if response.status_code != 200:
                logger.error(f"API请求失败: HTTP {response.status_code}")
                return OnlineTranslateResult(
                    success=False,
                    message=f"HTTP错误: {response.status_code}",
                    raw_response={"status_code": response.status_code, "text": response.text},
                )
            
            data = response.json()
            
            status = data.get("status", "")
            if status != "success":
                logger.warning(f"API返回非成功状态: {status}")
                return OnlineTranslateResult(
                    success=False,
                    message=data.get("message", f"状态: {status}"),
                    raw_response=data,
                )
            
            result = OnlineTranslateResult(
                success=True,
                message=data.get("message", "Translation completed"),
                translated=data.get("translated", False),
                has_product=data.get("has_product", False),
                image_base64=data.get("image"),
                raw_response=data,
            )
            
            logger.info(
                f"在线翻译完成: translated={result.translated}, "
                f"has_product={result.has_product}"
            )
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"API请求超时: {e}")
            return OnlineTranslateResult(
                success=False,
                message=f"请求超时: {self.timeout}秒",
            )
        except httpx.RequestError as e:
            logger.error(f"API请求错误: {e}")
            return OnlineTranslateResult(
                success=False,
                message=f"请求错误: {str(e)}",
            )
        except Exception as e:
            logger.error(f"翻译过程发生异常: {e}")
            return OnlineTranslateResult(
                success=False,
                message=str(e),
            )
    
    async def translate_image_bytes_async(self, image_bytes: bytes) -> OnlineTranslateResult:
        """
        异步翻译 - 通过字节数据
        
        Args:
            image_bytes: 图片字节数据
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return await self.translate_image_base64_async(image_base64)
    
    async def translate_image_file_async(self, file_path: str) -> OnlineTranslateResult:
        """
        异步翻译 - 通过本地文件
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            OnlineTranslateResult: 翻译结果
        """
        path = Path(file_path)
        if not path.exists():
            return OnlineTranslateResult(
                success=False,
                message=f"文件不存在: {file_path}",
            )
        
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            return OnlineTranslateResult(
                success=False,
                message=f"文件大小超过10MB限制: {file_size / 1024 / 1024:.2f}MB",
            )
        
        with open(path, "rb") as f:
            image_bytes = f.read()
        
        return await self.translate_image_bytes_async(image_bytes)


# 单例模式
_online_translate_service: Optional[OnlineTranslateService] = None


def get_online_translate_service() -> OnlineTranslateService:
    """获取在线翻译服务单例"""
    global _online_translate_service
    if _online_translate_service is None:
        _online_translate_service = OnlineTranslateService()
    return _online_translate_service


# ============================================
# 命令行使用示例
# ============================================
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="在线图片翻译工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译本地文件
  python online_translate_service.py --file input.jpg --output output.jpg
  
  # 使用自定义API地址
  python online_translate_service.py --file input.jpg --api-url "https://custom-api.com/translate"
        """
    )
    
    parser.add_argument("--file", "-f", required=True, help="输入图片文件路径")
    parser.add_argument("--output", "-o", help="输出图片文件路径")
    parser.add_argument("--api-url", help="自定义API端点")
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时时间(秒)")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # 创建服务
        service = OnlineTranslateService(
            api_url=args.api_url,
            timeout=args.timeout,
        )
        
        # 调用翻译
        result = service.translate_image_file(args.file)
        
        if result.success:
            print(f"✅ 翻译完成!")
            print(f"   translated: {result.translated}")
            print(f"   has_product: {result.has_product}")
            print(f"   message: {result.message}")
            
            # 保存输出图片
            if args.output and result.image_base64:
                with open(args.output, "wb") as f:
                    f.write(base64.b64decode(result.image_base64))
                print(f"   已保存到: {args.output}")
        else:
            print(f"❌ 翻译失败: {result.message}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)
