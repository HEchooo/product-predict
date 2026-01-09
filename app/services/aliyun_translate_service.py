"""
Aliyun Machine Translation Service - 图片翻译
API文档: https://help.aliyun.com/zh/machine-translation/developer-reference/api-alimt-2018-10-12-translateimage

使用阿里云机器翻译API翻译图片中的文字。
"""

import json
import os
import base64
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from alibabacloud_alimt20181012.client import Client as AlimtClient
from alibabacloud_alimt20181012 import models as alimt_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

logger = logging.getLogger(__name__)


@dataclass
class AliyunTranslateResult:
    """阿里云图片翻译结果"""
    success: bool
    request_id: str = ""
    code: int = 0
    message: str = ""
    final_image_url: str = ""
    inpainting_url: str = ""
    template_json: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


class AliyunTranslateService:
    """
    阿里云机器翻译服务 - 图片翻译
    
    使用前需要配置以下环境变量:
    - ALIYUN_ACCESS_KEY_ID: 阿里云 Access Key ID
    - ALIYUN_ACCESS_KEY_SECRET: 阿里云 Access Key Secret
    - ALIYUN_REGION_ID: 区域ID (默认: cn-hangzhou)
    
    使用示例:
        service = AliyunTranslateService()
        
        # 方式1: 使用URL
        result = service.translate_image_url(
            image_url="https://example.com/image.jpg",
            source_language="zh",
            target_language="en"
        )
        
        # 方式2: 使用本地文件
        result = service.translate_image_file(
            file_path="/path/to/image.jpg",
            source_language="zh",
            target_language="en"
        )
        
        # 方式3: 使用Base64
        result = service.translate_image_base64(
            image_base64="...",
            source_language="zh",
            target_language="en"
        )
    """
    
    # 支持的语言代码
    SUPPORTED_LANGUAGES = {
        "zh": "中文",
        "en": "英语",
        "ja": "日语",
        "ko": "韩语",
        "fr": "法语",
        "es": "西班牙语",
        "de": "德语",
        "it": "意大利语",
        "pt": "葡萄牙语",
        "ru": "俄语",
        "ar": "阿拉伯语",
        "th": "泰语",
        "vi": "越南语",
        "id": "印尼语",
        "ms": "马来语",
        "tr": "土耳其语",
        "pl": "波兰语",
        "nl": "荷兰语",
    }
    
    # 支持的翻译领域
    FIELD_GENERAL = "general"  # 通用图片翻译
    FIELD_ECOMMERCE = "e-commerce"  # 电商领域图片翻译
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        region_id: str = "cn-hangzhou",
        endpoint: str = "mt.cn-hangzhou.aliyuncs.com",
    ):
        """
        初始化阿里云翻译服务
        
        Args:
            access_key_id: 阿里云 Access Key ID (默认从环境变量读取)
            access_key_secret: 阿里云 Access Key Secret (默认从环境变量读取)
            region_id: 区域ID
            endpoint: API端点
        """
        self.access_key_id = access_key_id or os.getenv("ALIYUN_ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.getenv("ALIYUN_ACCESS_KEY_SECRET")
        self.region_id = os.getenv("ALIYUN_REGION_ID", region_id)
        self.endpoint = os.getenv("ALIYUN_MT_ENDPOINT", endpoint)
        
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError(
                "阿里云 Access Key 未配置。请设置环境变量 "
                "ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET"
            )
        
        self._client: Optional[AlimtClient] = None
    
    @property
    def client(self) -> AlimtClient:
        """获取或创建阿里云客户端"""
        if self._client is None:
            config = open_api_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                region_id=self.region_id,
                endpoint=self.endpoint,
            )
            self._client = AlimtClient(config)
        return self._client
    
    def translate_image_url(
        self,
        image_url: str,
        source_language: str = "zh",
        target_language: str = "en",
        field: str = "general",
        need_editor_data: bool = False,
        ignore_entity_recognize: bool = False,
    ) -> AliyunTranslateResult:
        """
        通过URL翻译图片
        
        Args:
            image_url: 图片URL
            source_language: 源语言 (默认: zh)
            target_language: 目标语言 (默认: en)
            field: 翻译领域 (general/e-commerce)
            need_editor_data: 是否需要译后编辑器数据
            ignore_entity_recognize: 是否忽略商品主体识别 (仅电商领域有效)
        
        Returns:
            AliyunTranslateResult: 翻译结果
        """
        return self._translate_image(
            image_url=image_url,
            image_base64=None,
            source_language=source_language,
            target_language=target_language,
            field=field,
            need_editor_data=need_editor_data,
            ignore_entity_recognize=ignore_entity_recognize,
        )
    
    def translate_image_base64(
        self,
        image_base64: str,
        source_language: str = "zh",
        target_language: str = "en",
        field: str = "general",
        need_editor_data: bool = False,
        ignore_entity_recognize: bool = False,
    ) -> AliyunTranslateResult:
        """
        通过Base64翻译图片
        
        Args:
            image_base64: 图片的Base64编码
            source_language: 源语言 (默认: zh)
            target_language: 目标语言 (默认: en)
            field: 翻译领域 (general/e-commerce)
            need_editor_data: 是否需要译后编辑器数据
            ignore_entity_recognize: 是否忽略商品主体识别 (仅电商领域有效)
        
        Returns:
            AliyunTranslateResult: 翻译结果
        """
        return self._translate_image(
            image_url=None,
            image_base64=image_base64,
            source_language=source_language,
            target_language=target_language,
            field=field,
            need_editor_data=need_editor_data,
            ignore_entity_recognize=ignore_entity_recognize,
        )
    
    def translate_image_file(
        self,
        file_path: str,
        source_language: str = "zh",
        target_language: str = "en",
        field: str = "general",
        need_editor_data: bool = False,
        ignore_entity_recognize: bool = False,
    ) -> AliyunTranslateResult:
        """
        翻译本地图片文件
        
        Args:
            file_path: 图片文件路径
            source_language: 源语言 (默认: zh)
            target_language: 目标语言 (默认: en)
            field: 翻译领域 (general/e-commerce)
            need_editor_data: 是否需要译后编辑器数据
            ignore_entity_recognize: 是否忽略商品主体识别 (仅电商领域有效)
        
        Returns:
            AliyunTranslateResult: 翻译结果
        """
        path = Path(file_path)
        if not path.exists():
            return AliyunTranslateResult(
                success=False,
                message=f"文件不存在: {file_path}"
            )
        
        # 检查文件大小 (限制10MB)
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            return AliyunTranslateResult(
                success=False,
                message=f"文件大小超过10MB限制: {file_size / 1024 / 1024:.2f}MB"
            )
        
        # 读取并编码为Base64
        with open(path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        return self.translate_image_base64(
            image_base64=image_base64,
            source_language=source_language,
            target_language=target_language,
            field=field,
            need_editor_data=need_editor_data,
            ignore_entity_recognize=ignore_entity_recognize,
        )
    
    def translate_image_bytes(
        self,
        image_bytes: bytes,
        source_language: str = "zh",
        target_language: str = "en",
        field: str = "general",
        need_editor_data: bool = False,
        ignore_entity_recognize: bool = False,
    ) -> AliyunTranslateResult:
        """
        翻译图片字节数据
        
        Args:
            image_bytes: 图片字节数据
            source_language: 源语言 (默认: zh)
            target_language: 目标语言 (默认: en)
            field: 翻译领域 (general/e-commerce)
            need_editor_data: 是否需要译后编辑器数据
            ignore_entity_recognize: 是否忽略商品主体识别 (仅电商领域有效)
        
        Returns:
            AliyunTranslateResult: 翻译结果
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return self.translate_image_base64(
            image_base64=image_base64,
            source_language=source_language,
            target_language=target_language,
            field=field,
            need_editor_data=need_editor_data,
            ignore_entity_recognize=ignore_entity_recognize,
        )
    
    def _translate_image(
        self,
        image_url: Optional[str],
        image_base64: Optional[str],
        source_language: str,
        target_language: str,
        field: str,
        need_editor_data: bool,
        ignore_entity_recognize: bool,
    ) -> AliyunTranslateResult:
        """
        内部方法: 调用阿里云API翻译图片
        
        Args:
            image_url: 图片URL (与image_base64二选一)
            image_base64: 图片Base64 (优先于image_url)
            source_language: 源语言
            target_language: 目标语言
            field: 翻译领域
            need_editor_data: 是否需要译后编辑器数据
            ignore_entity_recognize: 是否忽略商品主体识别
        
        Returns:
            AliyunTranslateResult: 翻译结果
        """
        try:
            # 构建扩展参数
            ext = {}
            if need_editor_data:
                ext["needEditorData"] = "true"
            if ignore_entity_recognize:
                ext["ignoreEntityRecognize"] = "true"
            
            # 构建请求
            request = alimt_models.TranslateImageRequest(
                source_language=source_language,
                target_language=target_language,
                field=field,
                image_url=image_url,
                image_base_64=image_base64,
                ext=json.dumps(ext) if ext else None,
            )
            
            # 运行时选项 - 增加超时时间
            runtime = util_models.RuntimeOptions(
                connect_timeout=30000,  # 连接超时30秒
                read_timeout=120000,    # 读取超时120秒
            )
            
            logger.info(
                f"调用阿里云图片翻译API: {source_language} -> {target_language}, "
                f"领域: {field}, URL: {image_url is not None}, "
                f"region_id: {self.region_id}, endpoint: {self.endpoint}"
            )
            
            # 调用API
            response = self.client.translate_image_with_options(request, runtime)
            
            # 解析响应
            if response.body:
                body = response.body
                data = body.data
                
                # 获取原始响应用于调试
                raw_response = self._response_to_dict(response)
                logger.info(f"原始响应: {json.dumps(raw_response, ensure_ascii=False,indent=2)}")
                
                # 构建更详细的错误信息
                code = body.code
                message = ""
                if code != "200":
                    message = f"API返回错误码: {code}, 错误信息: {body.message}"
                
                result = AliyunTranslateResult(
                    success= code == "200",
                    request_id=body.request_id,
                    code=code,
                    message=message,
                    final_image_url=data.final_image_url if data else "",
                    inpainting_url=data.in_painting_url if data else "",
                    template_json=data.template_json if data else "",
                    raw_response=raw_response,
                )
                
                if result.success:
                    logger.info(f"图片翻译成功: {result.final_image_url}")
                else:
                    logger.warning(f"图片翻译失败: code={result.code}, message={result.message}")
                    logger.debug(f"原始响应: {raw_response}")
                
                return result
            else:
                return AliyunTranslateResult(
                    success=False,
                    message="API响应为空"
                )
            
        except Exception as e:
            error_msg = str(e)
            # 尝试提取更详细的错误信息
            if hasattr(e, 'data'):
                error_msg = f"{error_msg}, data: {e.data}"
            if hasattr(e, 'message'):
                error_msg = f"{e.message}"
            logger.error(f"调用阿里云图片翻译API失败: {error_msg}")
            return AliyunTranslateResult(
                success=False,
                message=error_msg
            )
    
    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """将响应对象转换为字典"""
        try:
            if hasattr(response, "to_map"):
                return response.to_map()
            return {}
        except Exception:
            return {}


# 单例模式
_aliyun_translate_service: Optional[AliyunTranslateService] = None


def get_aliyun_translate_service() -> AliyunTranslateService:
    """获取阿里云翻译服务单例"""
    global _aliyun_translate_service
    if _aliyun_translate_service is None:
        _aliyun_translate_service = AliyunTranslateService()
    return _aliyun_translate_service


# ============================================
# 命令行使用示例
# ============================================
if __name__ == "__main__":
    import argparse
    import sys
    from dotenv import load_dotenv
    
    # 加载.env文件
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="阿里云图片翻译工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译URL图片
  python aliyun_translate_service.py --url "https://example.com/image.jpg"
  
  # 翻译本地文件
  python aliyun_translate_service.py --file "/path/to/image.jpg"
  
  # 指定语言和领域
  python aliyun_translate_service.py --file image.jpg --source zh --target en --field e-commerce
  
  # 下载翻译后的图片
  python aliyun_translate_service.py --file image.jpg --output translated.jpg
        """
    )
    
    # 输入参数 (二选一)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", help="图片URL")
    input_group.add_argument("--file", help="本地图片文件路径")
    
    # 翻译参数
    parser.add_argument("--source", "-s", default="zh", help="源语言 (默认: zh)")
    parser.add_argument("--target", "-t", default="en", help="目标语言 (默认: en)")
    parser.add_argument(
        "--field", 
        choices=["general", "e-commerce"], 
        default="general",
        help="翻译领域 (默认: general)"
    )
    
    # 输出参数
    parser.add_argument("--output", "-o", help="输出文件路径 (可选，会下载翻译后的图片)")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出结果")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # 创建服务
        service = AliyunTranslateService()
        
        # 调用翻译
        if args.url:
            result = service.translate_image_url(
                image_url=args.url,
                source_language=args.source,
                target_language=args.target,
                field=args.field,
            )
        else:
            result = service.translate_image_file(
                file_path=args.file,
                source_language=args.source,
                target_language=args.target,
                field=args.field,
            )
        
        # 输出结果
        if args.json:
            print(json.dumps({
                "success": result.success,
                "request_id": result.request_id,
                "code": result.code,
                "message": result.message,
                "final_image_url": result.final_image_url,
                "inpainting_url": result.inpainting_url,
            }, ensure_ascii=False, indent=2))
        else:
            if result.success:
                print(f"✅ 翻译成功!")
                print(f"   Request ID: {result.request_id}")
                print(f"   翻译后图片: {result.final_image_url}")
                if result.inpainting_url:
                    print(f"   背景图: {result.inpainting_url}")
            else:
                print(f"❌ 翻译失败: {result.message}")
                print(f"   错误码: {result.code}")
                if result.raw_response:
                    print(f"   原始响应: {json.dumps(result.raw_response, ensure_ascii=False, indent=2)}")
                sys.exit(1)
        
        # 下载图片 (如果指定了输出路径)
        if args.output and result.success and result.final_image_url:
            import urllib.request
            print(f"\n📥 正在下载翻译后的图片...")
            urllib.request.urlretrieve(result.final_image_url, args.output)
            print(f"   已保存到: {args.output}")
    
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)
