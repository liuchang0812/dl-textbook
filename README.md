# 从零理解深度学习

**一本写给程序员的交互式教材：从一行线性回归，一路走到亲手实现 mini-GPT。**

在线阅读：**<https://www.liuchang0812.com/dl-textbook/>**
（`https://liuchang0812.github.io/dl-textbook/` 会 301 跳转到此地址）

无需安装、无需联网、无需构建。双击 `index.html` 就能读——公式、图表、交互演示全部本地化。

---

## 这本书想解决什么问题

市面上讲深度学习的材料大致分两类：一类是推导严密但读起来像天书的论文和教科书，另一类是调用 `model.fit()` 就完事、让人始终不知道里面发生了什么的教程。

这本书走第三条路：**假设你会写代码，但不假设你有数学博士学位**。每个概念都先讲清楚"它到底在解决什么问题"，再给可运行的代码，最后才落到公式。所有关键机制都配了可以拖动的交互演示，让你亲手把参数推歪、看它怎么崩掉。

## 内容

七部、二十一章：

| 部 | 章节 | 主题 |
|---|---|---|
| 一 · 起点 | 1–4 | 从规则到学习、神经元与激活、前向传播、梯度下降 |
| 二 · 核心机制 | 5–7 | 反向传播、训练的技艺、卷积网络 |
| 三 · 序列与注意力 | 8–10 | RNN/LSTM、注意力机制、Transformer |
| 四 · 大模型 | 11–13 | 语言模型、预训练、对齐（SFT / RLHF / DPO） |
| 五 · 使用 | 14–16 | 推理与采样、微调与 LoRA、提示工程 |
| 六 · 应用 | 17–19 | RAG、智能体与工具调用、评测与安全 |
| 七 · 实践与前沿 | 20–21 | 从零实现 mini-GPT、MoE / 长上下文 / 推理模型 |

**第 20 章是全书的重心**：用纯 NumPy 从零写一个能训练的 GPT——反向传播引擎、softmax、LayerNorm、GELU、注意力、AdamW、采样，全部手写。这段代码不是伪代码，它真的跑过梯度检查（10 个算子全部通过，最大相对误差 ≤ 8.1e-3），也真的训练收敛过（loss 5.90 → 0.04）。

## 规模

| | |
|---|---|
| 章节 | 21 |
| 正文 | 约 11.2 万字 |
| 可交互演示 | 23 个 |
| 可运行代码 | 75 段 |
| 公式 | 732 个 |
| SVG 插图 | 36 张 |
| 表格 | 65 张 |
| 折叠习题 | 101 道 |
| 术语表条目 | 408 条 |
| 体积 | 2.5 MB（含内置 KaTeX） |

## 23 个可交互演示

不是截图，是真的能拖的：

- **neuron** 改权重看决策边界怎么转
- **activation** 对比 Sigmoid / Tanh / ReLU / GELU 及其导数
- **gd** 拖动学习率，看它收敛、震荡、还是发散
- **backprop** 逐层看梯度如何回流
- **conv** 滑动卷积核，逐格看特征图怎么算出来
- **attention** 改 query，看注意力权重如何重新分配
- **softmax** 拖温度，看概率分布从尖锐到平坦
- **sampling** temperature / top-k / top-p 对生成的影响
- **lora** 调秩 r，看参数量与表达能力怎么权衡
- **agentloop** 单步 ReAct 循环，逐步看思考→行动→观察
- **metrics** 拖阈值，看 precision / recall / F1 此消彼长
- **moe** Top-K 路由，看专家坍缩是怎么发生的
- ……共 23 个

## 技术实现

零构建、零框架、零依赖安装：

```
index.html            封面与目录
chapters/*.html       21 章正文
assets/style.css      设计系统（明/暗主题、字号调节）
assets/app.js         运行时引擎（导航、代码高亮、公式渲染）
assets/widgets.js     23 个交互组件
assets/vendor/katex/  内置 KaTeX 0.16.9（含 60 个字体文件）
code/mini_gpt.py      第 20 章的完整实现，可独立运行
```

- 公式渲染走 **本地 → CDN → 纯文本降级** 三级链路，断网也不会看到一屏 `$\nabla_\theta$` 源码。
- 交互组件用一套统一的 canvas 绘图工具（坐标映射、网格、曲线、标签），主题切换时自动重绘。
- 代码高亮是自己写的 Python/Bash 词法分析器，不引第三方库。

## 本地运行

```bash
git clone https://github.com/liuchang0812/dl-textbook.git
cd dl-textbook
open index.html          # 就这样，不需要 npm install
```

如果想跑第 20 章的代码：

```bash
cd code
python3 mini_gpt.py      # 需要 numpy
```

## 适合谁

- 会写 Python，想真正搞懂神经网络内部在干什么的程序员
- 用过 PyTorch / Transformers，但说不清 attention 和 LayerNorm 到底在算什么的人
- 想系统补一遍从反向传播到大模型对齐这条链路的人
- 准备面试，需要把"梯度消失为什么发生"讲明白的人

## 不适合谁

- 想找一本能直接调包做项目的速查手册
- 需要严格数学证明的读者（本书给直觉和推导骨架，不给定理证明）
- 完全不写代码的读者

## License

内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 许可，代码采用 MIT 许可。
