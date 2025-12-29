# Product Translate API

将产品图片中的中文文字（尤其是尺码表）翻译为英文，同时保持原有样式和布局的 API 服务。

## 功能特性

- 🔍 **OCR 识别** - 支持 RapidOCR 和 PaddleOCR
- 🎨 **智能修复** - 使用 LaMa/OpenCV 擦除原文字
- 🌐 **AI 翻译** - 使用 Google Gemini 进行翻译
- ✨ **样式保持** - 保持原有字体大小、颜色和位置

## 快速开始

### 1. 安装依赖

```bash
# 需要 Python 3.10+
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，设置 GEMINI_API_KEY
```

### 3. 启动服务

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 访问文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/translate` | 上传图片翻译 |
| POST | `/api/v1/translate/url` | URL 图片翻译 |

### 示例请求

```bash
# 上传图片翻译
curl -X POST "http://localhost:8000/api/v1/translate" \
  -F "file=@size_chart.jpg" \
  -F "return_base64=true"

# URL 图片翻译
curl -X POST "http://localhost:8000/api/v1/translate/url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/chart.jpg"}'
```

## 配置说明

### Gemini API Key

获取地址: https://aistudio.google.com/apikey

```bash
GEMINI_API_KEY="your-api-key"
```

### 图像修复后端

| 环境 | 配置 | 说明 |
|------|------|------|
| CUDA GPU | `lama` | 最佳效果 |
| Mac/CPU | `opencv` | 快速稳定 |

```bash
# CUDA GPU 服务器
DEFAULT_INPAINT_BACKEND="lama"
LAMA_DEVICE="cuda"

# Mac / CPU 服务器
DEFAULT_INPAINT_BACKEND="opencv"
```

## 项目结构

```
product-translate/
├── main.py                     # FastAPI 入口
├── preserve_style_translate.py # 核心翻译逻辑
├── chart_translate_schema.json # 翻译配置
├── app/
│   ├── api/translate.py        # API 路由
│   ├── config.py               # 配置管理
│   ├── models/translate.py     # 数据模型
│   └── services/translate_service.py  # 服务层
├── .env.example                # 环境变量模板
└── API_DOC.md                  # API 文档
```

## License

MIT
