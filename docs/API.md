# 图片翻译服务 API 文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

---

## 1. 同步翻译

同步翻译图片，等待翻译完成后返回结果。

**POST** `/api/v1/translate`

### 入参

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| imgUrl | string | ✅ | - | 图片URL |
| sourceLanguage | string | ❌ | zh | 源语言 |
| targetLanguage | string | ❌ | en | 目标语言 |
| type | string | ❌ | all | 翻译服务类型: all(自动降级)/aliyun(仅阿里云)/online(仅在线) |

### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| success | boolean | 是否成功 |
| translated | boolean | 是否进行了翻译 |
| service | string | 翻译服务 (aliyun/online/none) |
| imageUrl | string | 翻译后图片URL |
| message | string | 消息/错误信息 |

### curl 示例

```bash
# 默认模式 (阿里云优先，失败降级到在线)
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/image.jpg"
  }'

# 仅使用阿里云
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/image.jpg",
    "type": "aliyun"
  }'

# 仅使用在线翻译
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/image.jpg",
    "type": "online"
  }'
```

### 响应示例

```json
{
  "success": true,
  "translated": true,
  "service": "aliyun",
  "imageUrl": "https://cdn.translate.alibaba.com/xxx.jpg",
  "message": null
}
```

---

## 2. 提交异步翻译任务

提交翻译任务，立即返回任务ID，后台异步处理。

**POST** `/api/v1/translate/submit`

### 入参

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| imgUrl | string | ✅ | - | 图片URL |
| sourceLanguage | string | ❌ | zh | 源语言 |
| targetLanguage | string | ❌ | en | 目标语言 |
| enableCallback | boolean | ❌ | false | 是否启用回调 |
| callbackEnv | string | ❌ | test | 回调环境 (test/prod) |

### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| success | boolean | 是否成功提交 |
| taskId | string | 任务ID (UUID) |
| message | string | 消息 |

### curl 示例

```bash
curl -X POST http://localhost:8000/api/v1/translate/submit \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/image.jpg",
    "enableCallback": true,
    "callbackEnv": "prod"
  }'
```

### 响应示例

```json
{
  "success": true,
  "taskId": "5394ff94-c03a-4df4-b6cd-2abce994706a",
  "message": "Task submitted successfully"
}
```

---

## 3. 查询任务状态

查询异步翻译任务的状态和结果。

**GET** `/api/v1/translate/task/{taskId}`

### 路径参数

| 字段 | 类型 | 说明 |
|------|------|------|
| taskId | string | 任务ID |

### 出参

| 字段 | 类型 | 说明 |
|------|------|------|
| taskId | string | 任务ID |
| taskResult | string | 翻译后图片URL |
| translateSource | string | 翻译来源 (aliyun/online) |
| errorMessage | string | 错误信息 |
| status | string | 任务状态 |

### 任务状态

| 状态 | 说明 |
|------|------|
| PENDING | 等待处理 |
| PROCESSING | 处理中 |
| SUCCESS | 成功 |
| FAILED | 失败 |

### curl 示例

```bash
curl http://localhost:8000/api/v1/translate/task/5394ff94-c03a-4df4-b6cd-2abce994706a
```

### 响应示例

```json
{
  "taskId": "5394ff94-c03a-4df4-b6cd-2abce994706a",
  "taskResult": "https://cdn.translate.alibaba.com/xxx.jpg",
  "translateSource": "aliyun",
  "errorMessage": null,
  "status": "SUCCESS"
}
```

---

## 4. 回调通知

任务完成时，系统会向配置的回调URL发送POST请求。

**POST** `{CALLBACK_URL_TEST/CALLBACK_URL_PROD}`

### 回调参数

| 字段 | 类型 | 说明 |
|------|------|------|
| taskId | string | 任务ID |
| taskResult | string | 翻译后图片URL |
| translateSource | string | 翻译来源 |
| errorMessage | string | 错误信息 |
| status | string | 任务状态 (SUCCESS/FAILED) |

### 回调示例

```json
{
  "taskId": "5394ff94-c03a-4df4-b6cd-2abce994706a",
  "taskResult": "https://cdn.translate.alibaba.com/xxx.jpg",
  "translateSource": "aliyun",
  "errorMessage": "",
  "status": "SUCCESS"
}
```

---

## 配置说明

`.env` 文件配置：

```env
# MySQL 配置
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=translate_db

# Worker 配置
WORKER_CONCURRENCY=3
WORKER_POLL_INTERVAL=2.0

# 回调 URL
CALLBACK_URL_TEST=https://test.example.com/callback
CALLBACK_URL_PROD=https://prod.example.com/callback
```
