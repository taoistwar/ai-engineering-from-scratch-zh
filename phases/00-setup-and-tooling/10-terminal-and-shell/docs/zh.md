# 终端与 Shell

> 终端是 AI 工程师的生存之地。在这里要感到自在。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~35 分钟

## 学习目标

- 使用管道、重定向和 `grep` 从命令行过滤和处理训练日志
- 创建具有多个窗格的持久化 tmux 会话，用于并发训练和 GPU 监控
- 使用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统和 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地和远程机器之间传输文件

## 问题所在

你在终端上花费的时间会比在任何编辑器上都多。训练运行、GPU 监控、日志跟踪、远程 SSH 会话、环境管理。每个 AI 工作流都会触及 shell。如果你在这里很慢，那你在任何地方都很慢。

本课涵盖 AI 工作所需的终端技能。不涉及 Unix 历史。不深入 Bash 脚本。只讲你需要的内容。

## 概念

```mermaid
graph TD
    subgraph tmux["tmux 会话：training"]
        subgraph top["顶部行"]
            P1["窗格 1: 训练运行<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["窗格 2: GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | 内存：14/24G"]
        end
        P3["窗格 3: 日志 + 实验<br/>tail -f logs/train.log | grep loss"]
    end
```

三个任务同时运行。一个终端。你可以分离、回家、SSH 回来、重新附加。训练继续运行。

## 构建它

### 步骤 1：了解你的 shell

检查你正在运行哪个 shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`。两者都可以正常工作。本课程中的命令在两者中都适用。

需要知道的关键事项：

```bash
# 移动位置
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# 历史搜索（你会学到的最有用的快捷方式）
# Ctrl+R 然后输入之前命令的部分内容
# 再次按 Ctrl+R 循环匹配项

# 清屏
clear   # 或 Ctrl+L

# 取消运行中的命令
# Ctrl+C

# 挂起运行中的命令（用 fg 恢复）
# Ctrl+Z
```

### 步骤 2：管道和重定向

管道将命令连接在一起。这是你处理日志、过滤输出和链接工具的方式。你会不断使用这个。

```bash
# 计算日志中 "loss" 出现的次数
cat train.log | grep "loss" | wc -l

# 从训练输出中提取仅损失值
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# 实时监视日志文件更新，过滤错误
tail -f train.log | grep --line-buffered "ERROR"

# 按最终准确率对实验排序
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# 将标准输出和标准错误重定向到单独的文件
python train.py > output.log 2> errors.log

# 将两者重定向到同一个文件
python train.py > train_full.log 2>&1
```

你需要知道的三个重定向：

| 符号 | 作用 |
|--------|-------------|
| `>` | 将标准输出写入文件（覆盖） |
| `>>` | 将标准输出追加到文件 |
| `2>` | 将标准错误写入文件 |
| `2>&1` | 将标准错误发送到与标准输出相同的位置 |
| `\|` | 将一个命令的标准输出作为标准输入发送给下一个命令 |

### 步骤 3：后台进程

训练运行需要数小时。你不想一直开着终端。

```bash
# 在后台运行（输出仍然发送到终端）
python train.py &

# 在后台运行，不受挂断影响（关闭终端不会杀死它）
nohup python train.py > train.log 2>&1 &

# 检查后台运行的内容
jobs
ps aux | grep train.py

# 将后台作业带到前台
fg %1

# 杀死后台进程
kill %1
# 或找到它的 PID 并杀死
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 之间的区别：

| 方法 | 终端关闭后存活？| 可以重新附加？|
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（检查日志文件） |
| `screen` / `tmux` | 是 | 是 |

对于超过几分钟的任务，使用 tmux。

### 步骤 4：tmux

tmux 让你创建具有多个窗格的持久化终端会话。这是管理训练运行最有用的工具。

```bash
# 安装
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# 启动命名会话
tmux new -s training

# 水平分割
# Ctrl+B 然后 "

# 垂直分割
# Ctrl+B 然后 %

# 在窗格之间导航
# Ctrl+B 然后方向键

# 分离（会话继续运行）
# Ctrl+B 然后 d

# 重新附加
tmux attach -t training

# 列出会话
tmux ls

# 杀死会话
tmux kill-session -t training
```

典型的 AI 工作流会话：

```bash
tmux new -s train

# 窗格 1：启动训练
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " 分割，然后运行 GPU 监控
watch -n1 nvidia-smi

# Ctrl+B, % 垂直分割，跟踪日志
tail -f logs/experiment.log

# 现在用 Ctrl+B, d 分离
# SSH 退出，去喝杯咖啡，回来
# tmux attach -t train
```

### 步骤 5：使用 htop 和 nvtop 监控

```bash
# 系统进程（比 top 更好）
htop

# GPU 进程（如果你有 NVIDIA GPU）
# 安装：sudo apt install nvtop (Ubuntu) 或 brew install nvtop (macOS)
nvtop

# 不使用 nvtop 的快速 GPU 检查
nvidia-smi

# 每秒更新观看 GPU 使用情况
watch -n1 nvidia-smi

# 查看哪些进程正在使用 GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

你会用到的 `htop` 键绑定：
- `F6` 或 `>` 按列排序（按内存排序以查找内存泄漏）
- `F5` 切换树视图（查看子进程）
- `F9` 杀死进程
- `/` 搜索进程名称

### 步骤 6：SSH 连接远程 GPU 机器

当你租用云 GPU（Lambda、RunPod、Vast.ai）时，你通过 SSH 连接。

```bash
# 基本连接
ssh user@gpu-box-ip

# 使用特定密钥
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# 复制文件到远程
scp model.pt user@gpu-box-ip:~/models/

# 从远程复制文件
scp user@gpu-box-ip:~/results/metrics.json ./

# 同步整个目录（多文件时更快）
rsync -avz ./data/ user@gpu-box-ip:~/data/

# 端口转发（在本地访问远程 Jupyter/TensorBoard）
ssh -L 8888:localhost:8888 user@gpu-box-ip
# 现在在浏览器中打开 localhost:8888

# SSH 配置以便捷使用
# 添加到 ~/.ssh/config：
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# 然后只需：
# ssh gpu
```

### 步骤 7：AI 工作的有用别名

将这些添加到你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或复制你想要的别名。关键别名：

```bash
# 一目了然的 GPU 状态
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# 杀死所有 Python 训练进程
alias killtraining='pkill -f "python.*train"'

# 快速虚拟环境激活
alias ae='source .venv/bin/activate'

# 监视训练损失
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

参见 `code/shell_aliases.sh` 获取完整集合。

### 步骤 8：常见的 AI 终端模式

这些在实践中反复出现：

```bash
# 运行训练，记录所有内容，完成时通知
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "训练完成" you@email.com

# 并排比较两个实验日志
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# 查找最大的模型文件（清理磁盘空间）
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# 从 Hugging Face 下载模型
wget https://huggingface.co/model/resolve/main/model.safetensors

# 解压数据集
tar xzf dataset.tar.gz -C ./data/

# 统计所有 Python 文件的行数（查看项目规模）
find . -name "*.py" | xargs wc -l | tail -1

# 检查磁盘空间（训练数据很快填满磁盘）
df -h
du -sh ./data/*

# 训练前检查环境变量
env | grep -i cuda
env | grep -i torch
```

## 使用它

以下是每个工具在本课程中何时发挥作用：

| 工具 | 使用时机 |
|------|----------------|
| tmux | 每次训练运行（Phase 3+） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 调试慢速训练、OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 节省重复命令的时间 |

## 练习

1. 安装 tmux，创建一个有三个窗格的会话，在一个中运行 `htop`，另一个中运行 `watch -n1 date`，第三个中运行 Python 脚本。分离并重新附加。
2. 将 `code/shell_aliases.sh` 中的别名添加到你的 shell 配置并用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 创建一个假训练日志：`for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log`，然后使用 `grep`、`tail` 和 `awk` 提取仅损失值。
4. 为你有权访问的服务器设置 SSH 配置条目（或使用 `localhost` 练习语法）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Shell | "终端" | 解释你命令的程序（bash、zsh、fish） |
| tmux | "终端复用器" | 一个程序，让你在一个窗口内运行多个终端会话，并可以分离/重新附加 |
| Pipe | "管道符" | `\|` 操作符，将一个命令的输出作为输入发送给另一个命令 |
| PID | "进程 ID" | 分配给每个运行进程的唯一编号，用于监控或杀死它 |
| nohup | "不挂断" | 运行命令时不受挂断信号影响，这样关闭终端不会杀死它 |
| SSH | "连接到服务器" | 安全 Shell，一种用于在远程机器上运行命令的加密协议 |
