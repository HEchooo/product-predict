# Product Translate API 文档

## 概述

该 API 用于将产品图片中的中文文字（如尺码表）翻译为英文，同时保持原有的样式和布局。

---

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API 版本**: v1
- **文档**: `/docs` (Swagger UI) 或 `/redoc` (ReDoc)

---

## 接口列表

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应示例：**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### 2. 图片翻译（上传文件）

**POST** `/api/v1/translate`

**Content-Type**: `multipart/form-data`

#### 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file` | File | ✅ | - | 图片文件（支持 jpg, png, webp） |
| `ocr_backend` | string | ❌ | `rapid` | OCR 引擎：`rapid` / `paddle` / `auto` |
| `inpaint_backend` | string | ❌ | `lama` | 图像修复引擎：`lama` / `opencv` / `none` |
| `gate_mode` | string | ❌ | `auto` | 图表检测模式：`auto` / `off` / `strict` |
| `min_font_px` | int | ❌ | `10` | 最小字体大小（像素），范围 6-100 |
| `max_font_px` | int | ❌ | `64` | 最大字体大小（像素），范围 10-200 |
| `allow_extra_line` | bool | ❌ | `true` | 是否允许额外换行以适应翻译文本 |
| `max_extra_lines` | int | ❌ | `1` | 最多允许的额外行数，范围 0-3 |
| `return_base64` | bool | ❌ | `true` | 是否返回 base64 编码的图片 |

#### curl 示例

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -F "file=@size_chart.jpg" \
  -F "ocr_backend=rapid" \
  -F "inpaint_backend=lama" \
  -F "return_base64=true"
```

#### 响应示例

**成功 (200)：**
```json
{
  "success": true,
  "data": {
    "status": "processed",
    "reason": "ok",
    "chinese_tokens": 15,
    "total_tokens": 20,
    "regions": 5,
    "time_ms": 2345,
    "ocr_backend": "rapid",
    "inpaint_backend": "lama",
    "output_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "output_filename": "size_chart.en.jpg"
  },
  "error": null
}
```

**跳过处理（无中文）：**
```json
{
  "success": true,
  "data": {
    "status": "skipped",
    "reason": "no_chinese",
    "chinese_tokens": 0,
    "total_tokens": 10,
    "regions": 0,
    "time_ms": 123,
    "ocr_backend": "rapid",
    "inpaint_backend": "lama",
    "output_image_base64": null,
    "output_filename": null
  },
  "error": null
}
```

**失败：**
```json
{
  "success": false,
  "data": {
    "status": "error",
    "reason": "Mask too large (85.0%). Refusing to inpaint entire image."
  },
  "error": "Mask too large (85.0%). Refusing to inpaint entire image."
}
```

---

### 3. 图片翻译（URL）

**POST** `/api/v1/translate/url`

**Content-Type**: `application/json`

#### 请求体

```json
{
  "image_url": "https://example.com/size_chart.jpg",
  "options": {
    "ocr_backend": "rapid",
    "inpaint_backend": "lama",
    "gate_mode": "auto",
    "min_font_px": 10,
    "max_font_px": 64,
    "allow_extra_line": true,
    "max_extra_lines": 1,
    "return_base64": true
  }
}
```

#### 请求参数说明

**顶层参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `image_url` | string | ✅ | 图片 URL |
| `options` | object | ❌ | 翻译选项（见下表） |

**options 对象：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `ocr_backend` | string | ❌ | `rapid` | OCR 引擎 |
| `inpaint_backend` | string | ❌ | `lama` | 图像修复引擎 |
| `gate_mode` | string | ❌ | `auto` | 图表检测模式 |
| `min_font_px` | int | ❌ | `10` | 最小字体大小 |
| `max_font_px` | int | ❌ | `64` | 最大字体大小 |
| `allow_extra_line` | bool | ❌ | `true` | 允许额外换行 |
| `max_extra_lines` | int | ❌ | `1` | 最多额外行数 |
| `return_base64` | bool | ❌ | `true` | 返回 base64 图片 |

#### curl 示例

```bash
curl -X POST "http://localhost:8000/api/v1/translate/url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/size_chart.jpg",
    "options": {
      "return_base64": true
    }
  }'
```

---

## 参数详解

### ocr_backend（OCR 引擎）

| 值 | 说明 |
|----|------|
| `rapid` | RapidOCR（默认，速度快） |
| `paddle` | PaddleOCR（精度更高，需要额外依赖） |
| `auto` | 自动选择（默认使用 rapid） |

### inpaint_backend（图像修复引擎）

| 值 | 说明 |
|----|------|
| `lama` | LaMa 深度学习模型（默认，效果最好） |
| `opencv` | OpenCV 传统算法（速度快但效果一般） |
| `none` | 不进行修复 |

### gate_mode（图表检测模式）

| 值 | 说明 |
|----|------|
| `auto` | 自动检测是否为尺码表/图表（默认） |
| `off` | 关闭检测，处理所有图片 |
| `strict` | 严格模式，必须检测到表格特征才处理 |

### status（处理状态）

| 值 | 说明 |
|----|------|
| `processed` | 成功处理并翻译 |
| `skipped` | 跳过处理（无中文、不符合图表特征等） |
| `error` | 处理失败 |

---

## 响应字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `success` | bool | 请求是否成功 |
| `data.status` | string | 处理状态 |
| `data.reason` | string | 状态说明 |
| `data.chinese_tokens` | int | 检测到的中文文本区域数 |
| `data.total_tokens` | int | 检测到的总文本区域数 |
| `data.regions` | int | 处理的文本区域数 |
| `data.time_ms` | int | 处理耗时（毫秒） |
| `data.ocr_backend` | string | 使用的 OCR 引擎 |
| `data.inpaint_backend` | string | 使用的修复引擎 |
| `data.output_image_base64` | string | Base64 编码的结果图片 |
| `data.output_filename` | string | 输出文件名 |
| `error` | string | 错误信息（失败时） |

---

## 环境变量配置

通过 `.env` 文件配置默认参数：

```bash
# GCP 配置（必须）
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-east4
GEMINI_MODEL=gemini-2.0-flash-001

# OCR 配置
DEFAULT_OCR_BACKEND=rapid
PADDLE_LANG=ch

# 图像修复配置
DEFAULT_INPAINT_BACKEND=lama
LAMA_DEVICE=auto  # auto/cpu/cuda/mps

# 字体配置
MIN_FONT_PX=10
MAX_FONT_PX=64

# 其他
OUTPUT_DIR=output
```

---

## 错误码

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功（即使 status=skipped 也返回 200） |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |
