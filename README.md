# journal-backend · 语音日记 + 个人知识库后端

把 **Obsidian** 和你自己服务器上的 **vLLM** 连起来：

- 🎙 用语音讲今天的事 → 自动转文字（Whisper，默认 CPU，不占 3080）
- ❓ AI 先追问 2-3 个问题（学到了什么？恐慌几分？孩子原话是什么？）
- 📝 整理成结构化日记：**今天学到的 / 投资记录 / 和孩子 / 其他**，保留你的原意
- 🤖 AI 补充的内容一律放在 `> [!ai] AI 补充` 里，和你说的话分开
- 📈 投资记录自动生成 **+1 周 / +1 个月 / +3 个月** 回看提醒
- 🔍 以后可以问："上次大跌时我是怎么想的？"，回答带 `[[日记链接]]` 出处
- 🔒 全部跑在你自己的 Homelab，数据不出家门；日记只追加，**绝不覆盖**已有内容

---

## 1. 架构

```
 手机 / 电脑 Obsidian ──Syncthing──► 服务器上的 vault 文件夹
        │                                   ▲
        │ Copilot 插件 (OpenAI 格式)         │ 读写 Markdown
        ▼                                   │
 journal-backend (FastAPI, 端口 8765) ───────┘
   ├─ /v1/chat/completions  加上日记系统提示 + 检索到的日记片段，转发给 vLLM（支持流式）
   ├─ /journal/followups    生成追问
   ├─ /journal/compose      整理成结构化日记，写入 Journal/YYYY-MM-DD.md（已存在就追加）
   ├─ /search  /ask         检索日记（BM25，可选向量），带出处回答
   ├─ /transcribe           语音转文字
   ├─ Inbox 监听             Journal/Inbox 里的录音/文字 → 草稿 + 追问 → 你回答 → 正式日记
   └─ /invest/reviews       到期的投资回看
        │
        ▼
 vLLM (OpenAI 兼容, 端口 8000, RTX 3080 10GB)
```

检索默认用 **BM25（中文按字和双字切分）**，不需要任何显存；想要语义检索可以接任意 OpenAI 兼容的 `/embeddings` 服务，或在 CPU 上跑 `bge-m3`。向量服务挂了会自动退回 BM25。

---

## 2. 安装和运行

先在服务器上复制配置：

```bash
git clone https://github.com/jiahualihuanahuan/journal-backend.git
cd journal-backend
cp .env.example .env      # 然后编辑 .env：VLLM_BASE_URL、VLLM_MODEL、VAULT_PATH、BACKEND_API_KEY
```

### 方式 A：Docker Compose（推荐，服务器上）

```bash
# .env 里还要设置 VAULT_HOST_PATH（vault 在宿主机上的路径）和 BIND_ADDR（服务器的局域网 IP）
# vLLM 在同一台机器上时：VLLM_BASE_URL=http://host.docker.internal:8000/v1
docker compose up -d --build
curl -H "Authorization: Bearer <你的 BACKEND_API_KEY>" "http://<服务器IP>:8765/status?check_upstream=true"
```

要在容器里转写语音：`.env` 里设 `WITH_WHISPER=1`、`WHISPER_BACKEND=local`，再 `docker compose up -d --build`。

### 方式 B：Python 直接跑（Linux venv）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # 要语音：再 pip install -r requirements-optional.txt
./scripts/run.sh                           # 读取 .env 里的 HOST / PORT
```

### 方式 C：Windows + Anaconda（在你的台式机上试）

```bat
conda create -n journal python=3.11 -y
conda activate journal
pip install -r requirements.txt
copy .env.example .env
scripts\run.bat
```

开机自启（Linux 不用 Docker 时）可以写一个 systemd 服务，`ExecStart` 指向 `scripts/run.sh`。

### 跑测试（完全离线，用假的 vLLM）

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 3. vLLM 设置建议（RTX 3080 10GB）

推荐 **Qwen2.5-7B-Instruct-AWQ**（4-bit，权重约 5.5GB，中英文都好）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY"          # 可选；有的话 .env 里 VLLM_API_KEY 填同一个
```

- **上下文长度**：7B AWQ 在 10GB 上 `--max-model-len 8192~16384` 比较稳；整理一天的日记 8192 就够，检索问答多给点更好。启动报 KV cache 不够时先降到 8192。
- 也可以用 **Qwen3-8B-AWQ**，这时 `.env` 里设 `VLLM_DISABLE_THINKING=true`（关掉思考模式，整理更快）。代码也会自动去掉 `<think>...</think>`。
- vLLM 版本支持的话可以设 `VLLM_JSON_MODE=true`，让追问/整理稳定输出 JSON。
- `VLLM_MODEL` 必须和 vLLM 实际的模型名一致：`curl http://<服务器>:8000/v1/models` 看一下。
- **Whisper 用 CPU 还是 GPU？** vLLM 用 0.90 显存后几乎没剩。默认 CPU（`small` 模型，int8），一段 3 分钟的录音大约 1 分钟内转完（视 CPU 而定）。想要更准用 `medium` 或 `large-v3-turbo`（CPU 上更慢）。要用 GPU，把 vLLM 的 `--gpu-memory-utilization` 降到 0.80 左右，再设 `WHISPER_DEVICE=cuda`、`WHISPER_COMPUTE_TYPE=int8_float16`。

---

## 4. Obsidian 设置

### 4.1 Copilot 插件（聊天、问日记）

1. 社区插件里安装 **Copilot**（作者 Logan Yang）。
2. 设置 → Copilot → Model → **Add Custom Model**：
   - Model Name：`journal`（带日记检索和日记教练提示）——再加一个 `journal-raw`（纯转发，不加任何东西）
   - Provider：**OpenAI Format**（第三方 OpenAI 兼容）
   - Base URL：`http://<服务器局域网IP>:8765/v1`
   - API Key：`.env` 里的 `BACKEND_API_KEY`
   - 如果报 CORS 错误，打开该模型的 **CORS** 开关
3. 在 Copilot 聊天里选 `journal`，就可以：
   - 直接说今天的事 → 它会追问，再整理成日记格式（复制进笔记即可）
   - 问"上次纳指大跌时我怎么想的？" → 根据日记回答，带 `[[链接]]`

> 插件界面会随版本变化，以上字段名大致如此。Text Generator 等其他插件也一样：选"OpenAI 兼容 / Custom"，Base URL 填 `http://<服务器>:8765/v1`，模型填 `journal`。

### 4.2 AI 补充的样式

把 `obsidian/journal-ai-callout.css` 复制到 `<vault>/.obsidian/snippets/`，在 设置 → 外观 → CSS 代码片段 里打开。`[!ai]` 会显示成紫色机器人框，一眼分清哪些是 AI 补的。

### 4.3 推荐插件（可选）

- **Tasks**：日记里的 `- [ ] 一周后回看 📅 2026-11-16` 会出现在 Tasks 查询里
- **Dataview**：`fear_level`、`mood` 等字段可以做表，例如"所有恐慌 ≥ 8 分的日子"：

  ````
  ```dataview
  TABLE fear_level, mood FROM "Journal" WHERE fear_level >= 8 SORT date DESC
  ```
  ````

---

## 5. 每天怎么用（Inbox 流程）

`.env` 里打开：`INBOX_ENABLED=true`、`WHISPER_BACKEND=local`。

1. **录音**：手机录一段语音，存进 vault 的 `Journal/Inbox/` 文件夹（也可以放 `.md`/`.txt` 文字）。
   - Android：用任意录音 App 保存到 Syncthing 同步的 `Journal/Inbox`。
   - Obsidian 手机版自带「录音机」核心插件：设置 → 文件与链接 → 附件默认位置 设为 `Journal/Inbox`。（录音被处理后会移到 `Inbox/processed/`，原笔记里的录音链接可能失效，不影响日记。）
2. **追问**：半分钟内，`Journal/Inbox/` 里出现 `…-草稿.md`，里面是转写原文和 2-3 个追问。
3. **回答**：在每个问题下面写回答（可以留空），最后勾选 `- [x] 我已回答完，生成日记`（或把 `status` 改成 `answered`）。
4. **生成**：下一轮自动整理成 `Journal/2026-11-09.md`。当天已经有日记就**在末尾追加**一段「📝 21:30 补充记录」，原有内容一个字都不动。草稿归档到 `Inbox/processed/`。

同一个文件重复出现（同步冲突等）会按内容哈希识别，不会重复生成。

### 生成的日记长这样

```markdown
---
date: 2026-11-09
type: journal
mood: "紧张但坚定"
tags: [journal, journal/learnings, journal/investing, journal/kid]
themes: [learnings, investing, kid]
fear_level: 8
---
# 2026-11-09 大跌时加仓

## 今天学到的
- 恐慌时按计划分批买，比猜底更重要

> [!ai] AI 补充（不是我说的，仅供参考）
> - 可以提前写好分批加仓规则

## 投资记录
- **发生了什么**：NASDAQ 100 单日跌 6%
- **恐慌程度**：8/10
- **决策**：买入一小部分 QQQ
...
### 回看
- [ ] 一周后回看 📅 2026-11-16：
- [ ] 一个月后回看 📅 2026-12-09：
- [ ] 三个月后回看 📅 2027-02-09：

## 和孩子
- **孩子的原话**：
  - “妈妈，这是火箭！”
```

### 投资回看

- 到期时在原日记的回看行后面写结果，然后勾选。
- `GET /invest/reviews` 列出到期和过期的回看；`REVIEW_DIGEST_ENABLED=true` 时每小时更新 `Journal/投资回看清单.md`（只会覆盖本程序生成的这个文件）。

---

## 6. API 一览

所有接口（除 `/health`）都要 `Authorization: Bearer $BACKEND_API_KEY`（如果设置了）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活检查 |
| GET | `/status?check_upstream=true` | vault、索引、vLLM 连通性 |
| GET | `/v1/models` | `journal`、`journal-raw` |
| POST | `/v1/chat/completions` | OpenAI 兼容，支持 `stream` |
| POST | `/transcribe` | multipart `file=@录音.m4a` → `{"text": ...}` |
| POST | `/journal/followups` | `{"text": "..."}` → `{"questions": [...]}` |
| POST | `/journal/compose` | `{"text", "answers": [{"question","answer"}], "date"?, "dry_run"?}` |
| POST | `/search` | `{"query", "top_k"?}` |
| POST | `/ask` | `{"question"}` → 回答 + 出处 |
| POST | `/reindex` | 重建索引（新增笔记后，或 Syncthing 同步来的新文件） |
| POST | `/inbox/process` | 手动跑一轮 Inbox |
| GET | `/invest/reviews?as_of=&upcoming_days=&write=` | 投资回看 |

例子：

```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  http://<服务器>:8765/ask -d '{"question":"上次纳指大跌时我是怎么想的？"}'
```

---

## 7. 同步（Syncthing）

- vault 放在服务器上（例如 `/srv/obsidian/MyVault`），Syncthing 同步到 Windows 台式机和手机。
- Android 用 Syncthing-Fork；iPhone 可以用 Möbius Sync（或 Obsidian 官方 Sync，付费）。
- 在 Syncthing 里忽略 `.obsidian/workspace*.json`，减少冲突。
- 后端只会：新建当天日记、在末尾追加、在 Inbox 里建/改草稿、移动 Inbox 里的文件。

---

## 8. 安全

- **只在局域网用**：`BIND_ADDR` / `HOST` 设成服务器的局域网 IP 或 `127.0.0.1`，不要做端口转发暴露到公网。出门在外要用就走 Tailscale / WireGuard。
- **设置 `BACKEND_API_KEY`**：`python -c "import secrets;print(secrets.token_urlsafe(32))"` 生成，Obsidian 插件里填同一个。
- 密钥只放 `.env`（已在 `.gitignore`），代码里没有任何密钥。
- vLLM 也建议加 `--api-key`，并且只监听局域网。
- 提示词在 `prompts/zh/`，可以随时改（改完立即生效，不用重启）。

## 9. 自定义

- `prompts/zh/followups.md`：追问的风格和重点
- `prompts/zh/compose.md`：日记结构和字段
- `prompts/zh/ask.md`：回忆过去时的回答方式
- `prompts/zh/chat_system.md`：Copilot 聊天时的系统提示

## License

MIT
