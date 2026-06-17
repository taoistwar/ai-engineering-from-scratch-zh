# Git 与协作

> 版本控制不是可选的。你在这里编写的每个实验、每个模型、每节课都会被追踪。

**类型：** 学习
**语言：** --
**前置要求：** 阶段 0, 第 01 课
**时间：** 约 30 分钟

## 学习目标

- 配置 git 身份并使用日常的 add、commit 和 push 工作流
- 创建和合并分支以进行隔离实验，而不破坏 main 分支
- 编写 `.gitignore` 来排除模型检查点和大型二进制文件
- 使用 `git log` 导航提交历史以了解项目演进

## 问题所在

你即将在 20 个阶段中编写数百个代码文件。没有版本控制，你会丢失工作、破坏无法恢复的东西，并且无法与他人协作。

Git 是工具。GitHub 是代码存放的地方。本课程只涵盖你需要的内容，仅此而已。

## 概念

```mermaid
sequenceDiagram
    participant WD as 工作目录
    participant SA as 暂存区
    participant LR as 本地仓库
    participant R as 远程 (GitHub)
    WD->>SA: git add
    SA->>LR: git commit
    LR->>R: git push
    R->>LR: git fetch
    LR->>WD: git pull
```

要记住的三件事：
1. 经常保存 (`git commit`)
2. 推送到远程 (`git push`)
3. 为实验创建分支 (`git checkout -b experiment`)

## 构建它

### 步骤 1：配置 git

```bash
git config --global user.name "你的名字"
git config --global user.email "you@example.com"
```

### 步骤 2：日常工作流

```bash
git status
git add file.py
git commit -m "添加感知机实现"
git push origin main
```

### 步骤 3：为实验创建分支

```bash
git checkout -b experiment/new-optimizer

# ... 进行修改，提交 ...

git checkout main
git merge experiment/new-optimizer
```

### 步骤 4：使用本课程仓库

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# 完成课程，提交你的代码
git push origin my-progress
```

## 使用它

对于本课程，你只需要这些命令：

| 命令 | 何时使用 |
|------|----------|
| `git clone` | 获取课程仓库 |
| `git add` + `git commit` | 保存你的工作 |
| `git push` | 备份到 GitHub |
| `git checkout -b` | 尝试某些东西而不破坏 main |
| `git log --oneline` | 查看你已完成的工作 |

就是这样。你不需要 rebase、cherry-pick 或 submodules 来完成本课程。

## 练习

1. 克隆此仓库，创建一个名为 `my-progress` 的分支，创建一个文件，提交它，推送它
2. 创建一个 `.gitignore` 来排除模型检查点文件（`.pt`、`.pth`、`.safetensors`）
3. 使用 `git log --oneline` 查看此仓库的提交历史，了解课程是如何添加的

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Commit | "保存" | 项目在某个时间点的完整快照 |
| Branch | "副本" | 指向提交的指针，随着工作向前移动 |
| Merge | "合并代码" | 将一个分支的更改应用到另一个分支 |
| Remote | "云端" | 托管在其他地方的仓库副本（GitHub、GitLab） |
