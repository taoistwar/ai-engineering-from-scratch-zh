# 构建 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。一个真正的服务器需要暴露工具、资源和提示词，处理能力协商，发出结构化错误，并且在不同 SDK 间行为一致。本课程端到端构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三种服务器原语，以及一种纯函数风格，让你可以轻松迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06（MCP 基础）
**Time:** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 根据 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 在不重写工具逻辑的前提下，将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题

在使用远程传输（第 13 阶段 · 09）或认证层（第 13 阶段 · 16）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程启动，消息通过 stdin/stdout 以换行分隔的方式流动。

2025-11-25 规范规定 stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，正在 2026 年中被移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是全部的传输格式。

笔记服务器是一个好的示例形态，因为它涵盖了所有三种服务器原语。工具执行变更操作（`notes_create`）。资源暴露数据（`notes://{id}`）。提示词提供模板（`review_note`）。本课程的形态可泛化到任何领域。

## 概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个请求必须匹配一个带有相同 `id` 的响应。
- 通知绝对不能回复。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来控管功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是有类型的。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误分为两种形式。协议级错误（未知方法、错误参数）是 JSON-RPC 错误。工具级错误（有效调用但工具失败）返回为 `{content: [...], isError: true}`。这让模型在其上下文中看到失败。

### 实现资源

资源设计为只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据作为资源而非工具暴露时：

- 模型不会"调用"它；客户端可以按用户请求将其注入上下文。
- 订阅使服务器能够在资源发生变化时推送更新（第 13 阶段 · 10）。
- 第 13 阶段 · 14 通过 `ui://` 扩展了交互式资源的功能。

### 实现提示词

提示词是带有命名参数的模板。宿主将其呈现为斜杠命令。一个 `review_note` 提示词可能接受 `note_id` 参数，并生成一个多消息提示词模板供客户端馈送至其模型。

### stdio 传输细节

- 换行分隔的 JSON。没有长度前缀的帧格式。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，重试安全。
- `destructiveHint: true` — 不可逆副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（第 13 阶段 · 17）。

### 升级路径

`code/main.py` 中的标准库服务器大约 180 行。FastMCP（Python）将相同的逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 具有等效的形态。升级路径是即插即用的——概念（能力、分发、内容块）是相同的。

## 使用

`code/main.py` 是一个基于 stdio 的完整笔记 MCP 服务器，仅使用标准库。它处理 `initialize`、三个工具的 `tools/list` 和 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示词。你可以通过管道传入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

关注要点：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个内容块列表，而不是裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## 交付物

本课程产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能会搭建一个 MCP 服务器，包含正确的工具/资源/提示词拆分以及 SDK 升级路径。

## 练习

1. 运行 `code/main.py` 并使用手工构建的 JSON-RPC 消息驱动它。测试 `notes_create`，然后使用 `resources/read` 检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端是否会弹出确认对话框（这需要一个真实的宿主；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，使服务器在任何笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器迁移到 FastMCP。Python 文件应缩减到 80 行以内。传输行为必须相同；用相同的 JSON-RPC 测试框架验证。

5. 阅读规范中的 `server/tools` 部分，找出本课程服务器未实现的工具定义中的一个字段。（提示：有好几个；选一个并添加它。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| MCP 服务器 | "暴露工具的东西" | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端启动；通过 stdin/stdout 通信 |
| 分发器 | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| 内容块 | "工具结果块" | 工具响应中 `content` 数组里的有类型元素 |
| `isError` | "工具级失败" | 表示工具失败；与 JSON-RPC 错误区分 |
| 注解 | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | MCP 协议之上的基于装饰器的高级框架 |
| 资源 URI | "可寻址数据" | `file://`、`db://` 或标识资源的自定义方案 |
| 提示词模板 | "斜杠命令简述" | 服务器提供的带参数槽的模板，供宿主 UI 使用 |
| 能力声明 | "功能开关" | `initialize` 中声明的每种原语标志 |

## 进一步阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python 参考实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — TS 并行实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — 装饰器风格的 MCP 服务器 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
