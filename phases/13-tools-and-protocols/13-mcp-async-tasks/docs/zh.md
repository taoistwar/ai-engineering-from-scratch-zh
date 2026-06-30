# 异步任务（SEP-1686）— 立即调用、稍后获取的长时间运行工作

> 真正的代理工作需要数分钟到数小时：CI 运行、深度研究合成、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686 于 2025-11-25 合并，添加了任务原语：任何请求都可以增强为任务，并且结果可以稍后获取或通过状态通知流式传输。漂移风险说明：任务在 2026 年上半年期间是实验性的；SDK 接口仍在围绕规范进行设计。

**Type:** Build
**Languages:** Python（stdlib，异步任务状态机）
**Prerequisites:** Phase 13 · 07（MCP 服务器），Phase 13 · 09（传输）
**Time:** ~75 分钟

## 学习目标

- 识别何时将工具从同步提升为任务增强（服务器端工作超过 30 秒）。
- 遍历任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

一个 `generate_report` 工具运行一个多分钟的提取流水线。同步模型下的选项：

1. 保持连接打开三分钟。远程传输会断开；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏 MCP 统一性。
3. 触发后不管；无结果。

都不好。SEP-1686 添加了第四个选项：任务增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回任务 ID。客户端轮询 `tasks/status` 并在完成时获取 `tasks/result`。服务器端状态在重启后存活。

## 概念

### 任务增强

请求通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）成为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；ttl 之后任务结果被丢弃。

### 每个工具的选择性加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` — 此工具始终同步运行。对快速工具安全。
- `taskSupport: "optional"` — 客户端可以请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

一个 `generate_report` 工具会是 `required`。一个 `notes_search` 工具会是 `forbidden`。

### 状态

```
working  -> input_required -> working  （通过引导循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦 `completed`、`failed` 或 `cancelled`，任务就是终态的。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 阻塞或返回 404（如果尚未完成）。
- `tasks/cancel {taskId}` — 幂等；终态忽略。
- `tasks/list` — 可选；枚举活动任务和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

流式传输而非轮询的客户端获得更好的 UX。轮询始终作为最低限度接口支持。

### 持久状态

规范要求声明任务支持的服务器持久化状态。崩溃不应在 ttl 内丢失已完成的结果。存储范围从 SQLite 到 Redis 到文件系统。第 13 课的工具使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器尝试停止（检查执行器协作式取消）。如果已经是终态，请求是无操作的。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将进程已死的任何 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在其 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步任务加采样

一个任务本身可以调用 `sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器的任务线程根据需要采样客户端的模型，而客户端的 UI 将任务显示为 `working` 并附带周期性进度更新。

### 为什么这是实验性的

SEP-1686 于 2025-11-25 发布，但更广泛的路线图指出了三个未解决问题：持久订阅原语、子任务（父子任务关系）以及结果 TTL 标准化。预计规范将在 2026 年继续演变。生产代码应仅在常见情况下将任务视为稳定，并防范子任务相关的未来 SDK 变更。

## 使用

`code/main.py` 实现了持久化任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用工具，立即获取任务 ID，在工作线程更新进度时轮询 `tasks/status`，并在完成时获取 `tasks/result`。取消有效；崩溃恢复通过杀死工作线程并重新加载状态来模拟。

关注要点：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示其进展。
- 来自客户端的取消设置事件；工作线程检查并提前退出。
- "崩溃"时重新加载状态将进行中的任务标记为 `failed`，并附带 `CRASH_RECOVERY`。

## 交付物

本课程产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形态、ttl、持久性），选择正确的 taskSupport 标志，并绘制进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加 `tasks/cancel` 调用。验证工作线程响应它且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死工作线程，重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展到 SQLite。持久性收益相同；查询选项更多（列出会话 X 中的所有任务）。

5. 阅读 2026 年 MCP 路线图帖子。找出最可能影响下一年 SDK API 设计的与任务相关的未解决问题。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 任务 | "长时间运行的工具调用" | 带有 `_meta.task` 增强的请求，用于异步执行 |
| SEP-1686 | "任务规范" | 2025-11-25 添加了任务的 Spec Evolution Proposal |
| `_meta.task` | "任务信封" | 包含 id、state、ttl 的每个请求的元数据 |
| taskSupport | "工具标志" | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | "轮询方法" | 获取当前状态和可选的进度提示 |
| `tasks/result` | "获取结果" | 返回完成的负载或 404（如果尚未完成） |
| `tasks/cancel` | "停止它" | 幂等取消请求 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件 |
| 持久存储 | "崩溃安全状态" | 文件系统 / SQLite / Redis 持久化层 |

## 进一步阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案和完整讨论
- [WorkOS — MCP 用于 AI 代理工作流的异步任务](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带有理由的设计演练
- [DeepWiki — MCP 任务系统和异步操作](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制和状态机
- [FastMCP — 任务](https://gofastmcp.com/servers/tasks) — SDK 级任务实现模式
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 未解决问题和 2026 年优先级，包括子任务
