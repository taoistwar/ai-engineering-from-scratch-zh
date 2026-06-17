# Linux for AI

> 大多数 AI 都运行在 Linux 上。你需要掌握足够的知识，避免卡壳。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## Learning Objectives

- 从命令行导航 Linux 文件系统并执行基本文件操作
- 使用 `chmod` 和 `chown` 管理文件权限，解决"Permission denied"错误
- 使用 `apt` 安装系统包，并为 AI 工作配置全新的 GPU 机器
- 识别 macOS 与 Linux 的差异，避免在远程机器上踩坑

## The Problem

你在 macOS 或 Windows 上开发。但当你 SSH 到云 GPU 机器、租用 Lambda 实例或启动 EC2 机器时，你会进入 Ubuntu。终端是你唯一的界面。没有 Finder，没有 Explorer，没有 GUI。如果你无法从命令行导航文件系统、安装包和管理进程，你只能一边谷歌"如何在 Linux 中解压文件"，一边为闲置的 GPU 时间付费。

这是一份生存指南。它涵盖了你在远程 Linux 机器上进行 AI 工作所需的全部内容。仅此而已。

## File System Layout

Linux 将所有内容组织在单个根目录 `/` 下。没有 `C:\` 或 `/Volumes`。你实际会接触的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>你的文件 — 克隆仓库、运行训练"]
    root --> tmp["tmp/<br/>临时文件，重启时清除"]
    root --> usr["usr/<br/>系统程序和库"]
    root --> etc["etc/<br/>配置文件"]
    root --> varlog["var/log/<br/>日志 — 出问题时查看"]
    root --> mnt["mnt/ 或 /media/<br/>外部驱动器和卷"]
    root --> proc["proc/ 和 /sys/<br/>虚拟文件 — 内核和硬件信息"]
```

你的主目录是 `~` 或 `/home/your-username`。几乎所有操作都在这里进行。

## Essential Commands

这 15 个命令涵盖了你在远程 GPU 机器上 95% 的操作。

### Moving Around

```bash
pwd                         # 我在哪里？
ls                          # 这里有什么？
ls -la                      # 这里有什么，包括隐藏文件的详细信息？
cd /path/to/dir             # 去那里
cd ~                        # 回家
cd ..                       # 上一级
```

### Files and Directories

```bash
mkdir my-project            # 创建目录
mkdir -p a/b/c              # 一次性创建嵌套目录

cp file.txt backup.txt      # 复制文件
cp -r src/ src-backup/      # 复制目录（递归）

mv old.txt new.txt          # 重命名文件
mv file.txt /tmp/           # 移动文件

rm file.txt                 # 删除文件（没有回收站，文件彻底消失）
rm -rf my-dir/              # 删除目录及其中的所有内容
```

`rm -rf` 是永久性的。没有撤销。在按回车之前仔细检查路径。

### Reading Files

```bash
cat file.txt                # 打印整个文件
head -20 file.txt           # 前 20 行
tail -20 file.txt           # 后 20 行
tail -f log.txt             # 实时跟踪日志文件（按 Ctrl+C 停止）
less file.txt               # 滚动查看文件（按 q 退出）
```

### Searching

```bash
grep "error" training.log           # 查找包含"error"的行
grep -r "learning_rate" .           # 搜索当前目录下的所有文件
grep -i "cuda" config.yaml          # 不区分大小写的搜索

find . -name "*.py"                 # 查找当前目录下的所有 Python 文件
find . -name "*.ckpt" -size +1G     # 查找大于 1GB 的 checkpoint 文件
```

## Permissions

Linux 中的每个文件都有所有者和权限位。当脚本无法执行或你无法写入目录时，你会遇到这个问题。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             所有者权限：读、写、执行
#     ^^^          组权限：读、执行
#        ^^        其他人：只读
```

常见修复：

```bash
chmod +x train.sh           # 使脚本可执行
chmod 755 deploy.sh         # 所有者：完全权限，其他人：读+执行
chmod 644 config.yaml       # 所有者：读+写，其他人：只读

chown user:group file.txt   # 更改文件所有者（需要 sudo）
```

当出现"Permission denied"时，几乎总是权限问题。`chmod +x` 或 `sudo` 可以解决大多数情况。

## Package Management (apt)

Ubuntu 使用 `apt`。这是你安装系统级软件的方式。

```bash
sudo apt update             # 刷新包列表（总是先做这个）
sudo apt install -y htop    # 安装包（-y 跳过确认）
sudo apt install -y build-essential  # C 编译器、make 等。许多 Python 包需要
sudo apt install -y tmux    # 终端复用器（断开连接后保持会话活跃）

apt list --installed        # 安装了什么？
sudo apt remove htop        # 卸载
```

在全新的 GPU 机器上你通常会安装的包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## Users and sudo

你通常以普通用户身份登录。某些操作需要 root（管理员）权限。

```bash
whoami                      # 我是什么用户？
sudo command                # 以 root 身份运行单个命令
sudo su                     # 成为 root（输入 exit 返回，谨慎使用）
```

在云 GPU 实例上，你通常是唯一的用户，并且已经拥有 sudo 权限。不要以 root 身份运行所有操作。仅在需要时使用 sudo。

## Processes and systemd

当你的训练卡住，或者你需要检查正在运行的内容时：

```bash
htop                        # 交互式进程查看器（按 q 退出）
ps aux | grep python        # 查找正在运行的 Python 进程
kill 12345                  # 优雅地停止 PID 为 12345 的进程
kill -9 12345               # 强制终止（当优雅方式不起作用时使用）
nvidia-smi                  # GPU 进程和内存使用情况
```

systemd 管理服务（后台守护进程）。如果你运行推理服务器，你会用到它：

```bash
sudo systemctl start nginx          # 启动服务
sudo systemctl stop nginx           # 停止服务
sudo systemctl restart nginx        # 重启服务
sudo systemctl status nginx         # 检查是否正在运行
sudo systemctl enable nginx         # 开机自动启动
```

## Disk Space

GPU 机器通常磁盘空间有限。模型和数据集会快速填满它。

```bash
df -h                       # 所有挂载驱动器的磁盘使用情况
df -h /home                 # 专门查看 /home 的磁盘使用情况

du -sh *                    # 当前目录中每个项目的大小
du -sh ~/.cache             # 你的缓存大小（pip、huggingface 模型存放在这里）
du -sh /data/checkpoints/   # 检查你的 checkpoint 有多大

# 找出最大的空间占用者
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见的节省空间方法：

```bash
# 清除 pip 缓存
pip cache purge

# 清除 apt 缓存
sudo apt clean

# 删除你不需要的旧 checkpoint
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## Networking

你将从命令行下载模型、传输文件和调用 API。

```bash
# 下载文件
wget https://example.com/model.bin                   # 下载文件
curl -O https://example.com/data.tar.gz              # 用 curl 做同样的事
curl -s https://api.example.com/health | python3 -m json.tool  # 调用 API，格式化输出 JSON

# 在机器之间传输文件
scp model.bin user@remote:/data/                     # 复制文件到远程机器
scp user@remote:/data/results.csv .                  # 从远程复制文件到本地
scp -r user@remote:/data/checkpoints/ ./local-dir/   # 复制目录

# 同步目录（对于大传输比 scp 更快，支持断点续传）
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

对于任何大文件，使用 `rsync` 而不是 `scp`。它只传输更改的字节，并处理中断的连接。

## tmux: Keep Sessions Alive

当你 SSH 到远程机器时，关闭笔记本电脑会终止你的训练运行。tmux 可以防止这种情况。

```bash
tmux new -s train           # 启动名为"train"的新会话
# ... 开始你的训练，然后：
# Ctrl+B，然后 D            # 分离（训练继续运行）

tmux ls                     # 列出会话
tmux attach -t train        # 重新连接到会话

# 在 tmux 内部：
# Ctrl+B，然后 %            # 垂直分割窗格
# Ctrl+B，然后 "            # 水平分割窗格
# Ctrl+B，然后方向键        # 在窗格之间切换
```

总是在 tmux 内部运行长时间训练任务。总是。

## WSL2 for Windows Users

如果你使用 Windows，WSL2 为你提供真正的 Linux 环境，无需双启动。

```bash
# 在 PowerShell（管理员）中
wsl --install -d Ubuntu-24.04

# 重启后，从开始菜单打开 Ubuntu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真正的 Linux 内核。本课中的所有内容都可以在其中工作。你的 Windows 文件位于 WSL 内部的 `/mnt/c/Users/YourName/`。

GPU 直通在 Windows 端安装 NVIDIA 驱动程序后可用。安装 Windows NVIDIA 驱动程序（而不是 Linux 的），CUDA 将在 WSL2 内可用。

## Gotchas: macOS to Linux

如果你来自 macOS，这些事情会让你踩坑：

| macOS | Linux | 说明 |
|-------|-------|------|
| `brew install` | `sudo apt install` | 有时包名不同。`brew install htop` 与 `sudo apt install htop` 相同，但 `brew install readline` 与 `sudo apt install libreadline-dev` 不同。 |
| `open file.txt` | `xdg-open file.txt` | 但在远程机器上你没有 GUI。使用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过 SSH 不存在管道到/从剪贴板的功能。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认使用 zsh。大多数 Linux 服务器使用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`、`/usr/local/bin/` | 二进制文件位于不同的位置。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS 的 sed 在 `-i` 后需要空字符串。Linux 不需要。 |
| 不区分大小写的文件系统 | 区分大小写的文件系统 | 在 Linux 上，`Model.py` 和 `model.py` 是两个不同的文件。 |
| 行尾符 `\n` | 行尾符 `\n` | 相同。但 Windows 使用 `\r\n`，这会破坏 bash 脚本。运行 `dos2unix` 修复。 |

## Quick Reference Card

```
导航：        pwd, ls, cd, find
文件：        cp, mv, rm, mkdir, cat, head, tail, less
搜索：        grep, find
权限：        chmod, chown, sudo
包：          apt update, apt install
进程：        htop, ps, kill, nvidia-smi
服务：        systemctl start/stop/restart/status
磁盘：        df -h, du -sh
网络：        curl, wget, scp, rsync
会话：        tmux new/attach/detach
```

## Exercises

1. SSH 到任何 Linux 机器（或打开 WSL2）并导航到你的主目录。创建一个项目文件夹，在其中使用 `touch` 创建三个空文件，然后用 `ls -la` 列出它们。
2. 使用 apt 安装 `htop`，运行它，并识别哪个进程使用内存最多。
3. 启动 tmux 会话，在其中运行 `sleep 300`，分离，列出会话，然后重新连接。
4. 使用 `df -h` 检查可用磁盘空间，然后使用 `du -sh ~/.cache/*` 查找缓存中占用空间的内容。
5. 使用 `scp` 将文件从本地机器传输到远程机器，然后使用 `rsync` 进行相同的传输并比较体验。
