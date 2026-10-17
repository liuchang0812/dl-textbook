"""
mini_gpt.py —— 用 NumPy 从零实现并训练一个字符级 GPT
=========================================================================
只依赖 NumPy。全部代码约 300 行，没有任何"魔法"被藏起来。

运行：
    python mini_gpt.py              # 用内置小语料做演示
    python mini_gpt.py corpus.txt   # 用你自己的文本文件训练

作者注：这份代码是教学用的。它追求的是"每一行都能读懂"，
而不是速度。真实训练请用 PyTorch。
"""
import sys
import time
import numpy as np

# =========================================================================
# 1 · 自动微分引擎（约 120 行）
#    一个极小的反向模式自动微分：每个 Tensor 记录它由谁计算而来，
#    反向时按拓扑序调用各自注册的 _backward。
# =========================================================================
_GRAD_ON = True


class Tensor:
    """一个带梯度的 n 维数组。只实现 GPT 需要的那些算子。"""

    def __init__(self, data, requires_grad=False, _prev=(), _op=''):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._prev = tuple(_prev) if _GRAD_ON else ()
        self._op = _op
        self._backward = lambda: None
        # 参数一定需要梯度；中间激活只在开启梯度时分配
        self.grad = (np.zeros_like(self.data)
                     if (requires_grad or _GRAD_ON) else None)

    # ---------- 基础属性 ----------
    @property
    def shape(self):
        return self.data.shape

    # ---------- 加法（支持广播）----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _prev=(self, other), _op='add')

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # ---------- 减法 / 取负 ----------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    # ---------- 逐元素乘法（支持广播）----------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _prev=(self, other), _op='mul')

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # ---------- 矩阵乘法（支持批量与广播）----------
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _prev=(self, other), _op='matmul')

        def _backward():
            if self.requires_grad:
                g = out.grad @ other.data.swapaxes(-1, -2)
                self.grad += _unbroadcast(g, self.data.shape)
            if other.requires_grad:
                g = self.data.swapaxes(-1, -2) @ out.grad
                other.grad += _unbroadcast(g, other.data.shape)
        out._backward = _backward
        return out

    # ---------- 形状变换 ----------
    def reshape(self, *shape):
        old = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                     _prev=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(old)
        out._backward = _backward
        return out

    def transpose(self, perm):
        out = Tensor(self.data.transpose(perm), requires_grad=self.requires_grad,
                     _prev=(self,), _op='transpose')
        inv = tuple(np.argsort(perm))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    # ---------- 反向传播：按拓扑序调用 ----------
    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for c in t._prev:
                build(c)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()


def _unbroadcast(g, shape):
    """把广播后的梯度缩回原始形状。"""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# =========================================================================
# 2 · 融合算子（每个都手写前向 + 反向，约 60 行）
#    这些算子在 NumPy 里没有现成的反向，必须自己推。
# =========================================================================

def softmax(x):
    """沿最后一维做 softmax。dL/dx = y * (g - sum(g*y))"""
    m = x.data.max(axis=-1, keepdims=True)
    e = np.exp(x.data - m)
    y = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(y, requires_grad=x.requires_grad, _prev=(x,), _op='softmax')

    def _backward():
        if x.requires_grad:
            x.grad += y * (out.grad - (out.grad * y).sum(axis=-1, keepdims=True))
    out._backward = _backward
    return out


def layernorm(x, gamma, beta, eps=1e-5):
    """对最后一维做归一化，再仿射变换。"""
    mu = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x.data - mu) / std
    out = Tensor(xhat * gamma.data + beta.data,
                 requires_grad=(x.requires_grad or gamma.requires_grad
                                or beta.requires_grad),
                 _prev=(x, gamma, beta), _op='layernorm')
    D = x.data.shape[-1]

    def _backward():
        g = out.grad
        if gamma.requires_grad:
            gamma.grad += (g * xhat).reshape(-1, D).sum(axis=0)
        if beta.requires_grad:
            beta.grad += g.reshape(-1, D).sum(axis=0)
        if x.requires_grad:
            dxhat = g * gamma.data
            dvar = (dxhat * (x.data - mu) * -0.5 * std ** -3).sum(-1, keepdims=True)
            dmu = ((dxhat * -1.0 / std).sum(-1, keepdims=True)
                   + dvar * (-2.0 * (x.data - mu)).sum(-1, keepdims=True) / D)
            x.grad += dxhat / std + dvar * 2.0 * (x.data - mu) / D + dmu / D
    out._backward = _backward
    return out


def gelu(x):
    """tanh 近似的 GELU。"""
    c = 0.7978845608
    xd = x.data
    t = np.tanh(c * (xd + 0.044715 * xd ** 3))
    out = Tensor(0.5 * xd * (1.0 + t), requires_grad=x.requires_grad,
                 _prev=(x,), _op='gelu')
    d = 0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * xd * xd)

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * d
    out._backward = _backward
    return out


def embedding(W, idx):
    """查表：从 W 里按 idx 取出若干行。反向用 np.add.at 累加。"""
    out = Tensor(W.data[idx], requires_grad=W.requires_grad,
                 _prev=(W,), _op='embedding')

    def _backward():
        if W.requires_grad:
            np.add.at(W.grad, idx.reshape(-1),
                      out.grad.reshape(-1, W.data.shape[1]))
    out._backward = _backward
    return out


def cross_entropy(logits, targets):
    """logits: (N, V)  targets: (N,)  返回标量平均损失。"""
    x = logits.data
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    p = e / e.sum(axis=-1, keepdims=True)
    N = targets.size
    flat = p.reshape(-1, p.shape[-1])
    rows = np.arange(N)
    loss = float(-np.log(flat[rows, targets.reshape(-1)] + 1e-9).mean())

    out = Tensor(loss, requires_grad=logits.requires_grad,
                 _prev=(logits,), _op='ce')

    def _backward():
        if logits.requires_grad:
            d = p.copy()
            d.reshape(-1, p.shape[-1])[rows, targets.reshape(-1)] -= 1.0
            logits.grad += d / N
    out._backward = _backward
    return out


class no_grad:
    """推理时关掉图的构建，省内存也更快。"""

    def __enter__(self):
        global _GRAD_ON
        _GRAD_ON = False

    def __exit__(self, *a):
        global _GRAD_ON
        _GRAD_ON = True


# =========================================================================
# 3 · 模型（约 90 行）
# =========================================================================
class GPT:
    def __init__(self, vocab_size, block=64, n_embd=128, n_head=4,
                 n_layer=4, seed=0):
        assert n_embd % n_head == 0
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.block, self.n_embd, self.n_head, self.n_layer = \
            block, n_embd, n_head, n_layer
        self.hd = n_embd // n_head

        def param(*shape, scale=0.02):
            return Tensor(rng.normal(0, scale, size=shape), requires_grad=True)

        def zeros(*shape):
            return Tensor(np.zeros(shape, np.float32), requires_grad=True)

        def ones(n):
            return Tensor(np.ones(n, np.float32), requires_grad=True)

        # 词嵌入 + 位置嵌入
        self.wte = param(vocab_size, n_embd)
        self.wpe = param(block, n_embd)

        # 因果掩码：上三角（含对角线右侧）全部填 -1e9
        # 存成普通 numpy 数组，用到时再包成 Tensor（它是常数，不需要梯度）
        self.mask = np.triu(np.ones((block, block), np.float32), 1) * -1e9

        # n_layer 个 Transformer 块
        self.blocks = []
        for _ in range(n_layer):
            self.blocks.append({
                'ln1g': ones(n_embd), 'ln1b': zeros(n_embd),
                'wq': param(n_embd, n_embd), 'bq': zeros(n_embd),
                'wk': param(n_embd, n_embd), 'bk': zeros(n_embd),
                'wv': param(n_embd, n_embd), 'bv': zeros(n_embd),
                'wo': param(n_embd, n_embd, scale=0.02 / np.sqrt(2 * n_layer)),
                'bo': zeros(n_embd),
                'ln2g': ones(n_embd), 'ln2b': zeros(n_embd),
                'w1': param(n_embd, 4 * n_embd), 'b1': zeros(4 * n_embd),
                'w2': param(4 * n_embd, n_embd,
                            scale=0.02 / np.sqrt(2 * n_layer)),
                'b2': zeros(n_embd),
            })

        # 末端 LayerNorm。注意没有单独的 lm_head：
        # 输出投影直接复用 wte 的转置（权重共享），能省一半参数。
        self.lnfg = ones(n_embd)
        self.lnfb = zeros(n_embd)

    # ---------- 参数列表（给优化器用）----------
    def parameters(self):
        ps = [self.wte, self.wpe, self.lnfg, self.lnfb]
        for b in self.blocks:
            ps.extend(b.values())
        return ps

    def num_params(self):
        return sum(p.data.size for p in self.parameters())

    # ---------- 多头因果自注意力 ----------
    def attention(self, x, blk, B, T):
        C, H, hd = self.n_embd, self.n_head, self.hd

        def heads(proj, bias):
            y = x @ proj + bias                       # (B*T, C)
            y = y.reshape(B, T, H, hd)                # 拆成多头
            return y.transpose((0, 2, 1, 3))          # (B, H, T, hd)

        q = heads(blk['wq'], blk['bq'])
        k = heads(blk['wk'], blk['bk'])
        v = heads(blk['wv'], blk['bv'])

        # 打分：(B,H,T,hd) @ (B,H,hd,T) -> (B,H,T,T)
        scores = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / np.sqrt(hd))
        scores = scores + Tensor(self.mask[:T, :T])   # 屏蔽未来位置
        attn = softmax(scores)

        out = attn @ v                                # (B,H,T,hd)
        out = out.transpose((0, 2, 1, 3)).reshape(B * T, C)   # 合并多头
        return out @ blk['wo'] + blk['bo']

    # ---------- 前馈网络 ----------
    @staticmethod
    def mlp(x, blk):
        return gelu(x @ blk['w1'] + blk['b1']) @ blk['w2'] + blk['b2']

    # ---------- 前向 ----------
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = embedding(self.wte, idx)                  # (B, T, C)
        x = x + embedding(self.wpe, np.arange(T))     # 加上位置信息
        x = x.reshape(B * T, self.n_embd)

        for blk in self.blocks:
            # Pre-LN + 残差：先归一化再进子层，最后加回原输入
            x = x + self.attention(layernorm(x, blk['ln1g'], blk['ln1b']),
                                   blk, B, T)
            x = x + self.mlp(layernorm(x, blk['ln2g'], blk['ln2b']), blk)

        x = layernorm(x, self.lnfg, self.lnfb)
        logits = x @ self.wte.transpose((1, 0))       # (B*T, V)，权重共享

        if targets is None:
            return logits
        loss = cross_entropy(logits, targets.reshape(-1))
        return logits, loss

    # ---------- 采样生成 ----------
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 rng=None):
        rng = rng or np.random.default_rng(0)
        idx = idx.copy()
        with no_grad():
            for _ in range(max_new_tokens):
                cond = idx[:, -self.block:]           # 只保留最后 block 个 token
                logits = self.forward(cond)           # (B*T, V)
                logits = logits.reshape(cond.shape[0], cond.shape[1], -1)
                last = logits.data[:, -1, :] / max(temperature, 1e-6)

                if top_k is not None:
                    kth = np.sort(last, axis=-1)[:, -top_k][:, None]
                    last = np.where(last < kth, -np.inf, last)

                last = last - last.max(axis=-1, keepdims=True)
                p = np.exp(last)
                p = p / p.sum(axis=-1, keepdims=True)
                nxt = np.array([rng.choice(len(row), p=row) for row in p])
                idx = np.concatenate([idx, nxt[:, None]], axis=1)
        return idx


# =========================================================================
# 4 · 优化器：AdamW（约 25 行）
# =========================================================================
class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01):
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mhat = self.m[i] / (1 - b1 ** self.t)
            vhat = self.v[i] / (1 - b2 ** self.t)
            # 解耦权重衰减：直接作用在参数上，不进入梯度
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps)
                                 + self.wd * p.data)


# =========================================================================
# 5 · 数据：字符级分词 + 取样（约 35 行）
# =========================================================================
class CharTokenizer:
    """最朴素的分词：每个字符是一个 token。

    真实模型用 BPE（见第 11 章），但字符级足够让我们把
    注意力、反向传播、采样这条链路完整走通，且不需要任何外部依赖。
    """

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)


def get_batch(data, block, batch_size, rng):
    ix = rng.integers(0, len(data) - block - 1, size=batch_size)
    x = np.stack([data[i:i + block] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


# =========================================================================
# 6 · 训练与生成（约 45 行）
# =========================================================================
DEMO_TEXT = (
    "深度学习是机器学习的一个分支，它使用多层神经网络从数据中学习表示。"
    "神经网络由许多神经元组成，每个神经元接收输入，做加权求和，再通过激活函数。"
    "反向传播算法让梯度从输出层流回输入层，从而更新每一个参数。"
    "注意力机制让模型能够直接关注序列中任意位置的信息。"
    "Transformer 完全基于注意力，它由多头自注意力和前馈网络交替堆叠而成。"
    "语言模型的训练目标是预测下一个词，这个简单的目标却能带来惊人的能力。"
    "梯度下降是最基础的优化方法，它沿着梯度的反方向更新参数。"
    "学习率决定了每一步走多远，太大容易震荡，太小收敛缓慢。"
    "过拟合是指模型记住了训练数据，却没有学到可泛化的规律。"
    "正则化和 dropout 是常用的缓解过拟合的手段。\n"
    "卷积神经网络利用图像的局部性，用同一个卷积核在整张图上滑动。"
    "权值共享让参数量大幅下降，同时带来了平移不变性。"
    "残差连接让梯度可以跳过若干层直接回传，使训练上百层的网络成为可能。"
    "循环网络按时间顺序处理序列，但长距离依赖会导致梯度消失。"
    "LSTM 用门控机制控制信息的保留与遗忘，缓解了这个问题。"
    "批归一化让每一层的输入分布保持稳定，从而允许更大的学习率。"
    "初始化很重要：全零初始化会让所有神经元学到完全相同的东西。\n"
    "预训练让模型在海量文本上学会语言的一般规律。"
    "指令微调用成对的指令和回答，教模型听懂人类的请求。"
    "人类反馈强化学习用偏好数据进一步打磨模型的输出风格。"
    "检索增强生成先从知识库找出相关段落，再让模型基于这些段落回答。"
    "提示工程的核心不是念咒语，而是把任务、约束和示例讲清楚。"
    "思维链让模型把中间推理写出来，从而获得更多的计算步骤。"
    "量化用更少的比特表示权重，换取更小的显存和更快的推理。\n"
    "评测是工程的分水岭：没有评测集的迭代只是随机游走。"
    "幻觉来自训练目标本身，模型从来没有被奖励过说不知道。"
    "准确率在样本不平衡时会骗人，这时候要看精确率和召回率。"
    "混合专家让每个词只激活少数几个专家，从而把参数和计算解耦。"
    "长上下文不等于有效上下文，中间位置的信息容易被忽略。"
    "推理模型在给出答案之前先想很久，用推理时的算力换准确率。"
) * 3


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = open(path, encoding='utf-8').read() if path else DEMO_TEXT

    tok = CharTokenizer(text)
    data = tok.encode(text)
    n = int(len(data) * 0.9)
    train, val = data[:n], data[n:]

    print(f'语料 {len(text)} 字 · 词表 {tok.vocab_size} · '
          f'train {len(train)} / val {len(val)}')

    model = GPT(tok.vocab_size, block=64, n_embd=128, n_head=4, n_layer=4)
    print(f'参数量 {model.num_params() / 1e3:.1f}K')

    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(1)

    @np.errstate(all='ignore')
    def evaluate(iters=20):
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(val, model.block, 16, rng)
            with no_grad():
                _, loss = model.forward(xb, yb)
            losses.append(loss.data)
        return float(np.mean(losses))

    max_iters, batch = 600, 16
    t0 = time.time()
    for it in range(max_iters + 1):
        xb, yb = get_batch(train, model.block, batch, rng)
        _, loss = model.forward(xb, yb)

        opt.zero_grad()
        loss.backward()
        # 梯度裁剪：防止偶发的梯度爆炸把参数打飞
        total = np.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
        if total > 1.0:
            for p in model.parameters():
                p.grad *= 1.0 / (total + 1e-6)
        opt.step()

        if it % 100 == 0:
            print(f'  step {it:4d}  loss {loss.data:.4f}  '
                  f'val {evaluate():.4f}  {time.time() - t0:.1f}s')

    print('\n训练完成。采样三段看看效果：\n')
    for i, temp in enumerate((0.5, 0.8, 1.0)):
        start = tok.encode('深度')[None, :]
        out = model.generate(start, 60, temperature=temp, top_k=20,
                             rng=np.random.default_rng(100 + i))
        print(f'[temperature={temp}] {tok.decode(out[0])}\n')


if __name__ == '__main__':
    main()
