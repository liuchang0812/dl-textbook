/* ==========================================================================
   《深度学习之路》· 交互式小组件库
   每个组件由章节 HTML 中的 <div class="widget" data-widget="名字"> 触发。
   本文件只做"计算 + 绘图 + 事件绑定"，文案写在 HTML 里。
   ========================================================================== */
(function () {
  'use strict';

  const D = window.DL || {};
  const $ = D.$, $$ = D.$$, cssVar = D.cssVar;
  const registry = {};
  const instances = [];

  /* ---------------- 通用绘图工具 ---------------- */
  const PAL = ['#3b6ef5', '#e0483f', '#0f9d8f', '#c2410c', '#7c3aed', '#0891b2', '#d946ef', '#65a30d'];
  const GRID = 'rgba(130,136,150,.26)';
  const GRID_SOFT = 'rgba(130,136,150,.14)';

  function ink()  { return cssVar('--ink-3') || '#7b828c'; }
  function ink2() { return cssVar('--ink-2') || '#454b54'; }
  function ink4() { return cssVar('--ink-4') || '#a8aeb8'; }
  function surface() { return cssVar('--surface') || '#fff'; }

  function canvasOf(el) { return el.querySelector('canvas'); }

  function fit(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 420;
    const h = parseInt(cv.getAttribute('data-h') || '320', 10);
    cv.style.height = h + 'px';
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function makePlot(ctx, w, h, o) {
    o = Object.assign({ xmin: -5, xmax: 5, ymin: -1.5, ymax: 3, pad: 30 }, o);
    const { xmin, xmax, ymin, ymax, pad } = o;
    const X = (v) => pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad);
    const Y = (v) => h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad);
    const p = { X, Y, xmin, xmax, ymin, ymax, pad, w, h, ctx };
    p.grid = function (nx, ny) {
      ctx.save();
      ctx.lineWidth = 1;
      for (let i = 0; i <= nx; i++) {
        const v = xmin + (xmax - xmin) * i / nx;
        ctx.strokeStyle = Math.abs(v) < 1e-9 ? GRID : GRID_SOFT;
        ctx.beginPath(); ctx.moveTo(X(v), Y(ymin)); ctx.lineTo(X(v), Y(ymax)); ctx.stroke();
      }
      for (let i = 0; i <= ny; i++) {
        const v = ymin + (ymax - ymin) * i / ny;
        ctx.strokeStyle = Math.abs(v) < 1e-9 ? GRID : GRID_SOFT;
        ctx.beginPath(); ctx.moveTo(X(xmin), Y(v)); ctx.lineTo(X(xmax), Y(v)); ctx.stroke();
      }
      ctx.restore();
      return p;
    };
    p.curve = function (f, color, width, dash) {
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width || 2;
      if (dash) ctx.setLineDash(dash);
      ctx.lineJoin = 'round';
      const N = Math.max(120, Math.round(w * 1.2));
      for (let i = 0; i <= N; i++) {
        const x = xmin + (xmax - xmin) * i / N;
        const y = f(x);
        if (!isFinite(y)) continue;
        const px = X(x), py = Y(Math.max(ymin - 2, Math.min(ymax + 2, y)));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
      return p;
    };
    p.label = function (text, x, y, color, align, size) {
      ctx.save();
      ctx.fillStyle = color || ink();
      ctx.font = (size || 11) + 'px -apple-system,"PingFang SC",sans-serif';
      ctx.textAlign = align || 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y);
      ctx.restore();
    };
    return p;
  }

  function debounce(fn, ms) {
    let t;
    return function () {
      clearTimeout(t);
      const a = arguments, s = this;
      t = setTimeout(() => fn.apply(s, a), ms || 120);
    };
  }

  /* ==========================================================================
     组件 1 · 激活函数图鉴
     ========================================================================== */
  const ACT = {
    relu:     { label: 'ReLU',      f: (x) => Math.max(0, x),                 d: (x) => (x > 0 ? 1 : 0) },
    leaky:    { label: 'LeakyReLU', f: (x) => (x > 0 ? x : 0.1 * x),          d: (x) => (x > 0 ? 1 : 0.1) },
    sigmoid:  { label: 'Sigmoid',   f: (x) => 1 / (1 + Math.exp(-x)),         d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); } },
    tanh:     { label: 'Tanh',      f: (x) => Math.tanh(x),                   d: (x) => 1 - Math.tanh(x) ** 2 },
    gelu:     { label: 'GELU',      f: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
                                    d: (x) => { const t = Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3));
                                                return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * x * x); } },
    silu:     { label: 'SiLU',      f: (x) => x / (1 + Math.exp(-x)),
                                    d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s + x * s * (1 - s); } },
    softplus: { label: 'Softplus',  f: (x) => Math.log(1 + Math.exp(x)),      d: (x) => 1 / (1 + Math.exp(-x)) }
  };

  registry.activation = function (el) {
    const cv = canvasOf(el);
    const chips = $$('[data-fn]', el);
    const info = el.querySelector('[data-role="info"]');
    const derivBtn = el.querySelector('[data-role="toggle-deriv"]');
    let showDeriv = false;

    function draw() {
      const { ctx, w, h } = fit(cv);
      const p = makePlot(ctx, w, h, { xmin: -4.2, xmax: 4.2, ymin: -1.6, ymax: 3.2, pad: 28 });
      p.grid(8, 6);
      const active = chips.filter((c) => c.classList.contains('is-on'));
      const labels = [];
      active.forEach((c) => {
        const key = c.getAttribute('data-fn');
        const col = PAL[chips.indexOf(c) % PAL.length];
        const fn = ACT[key];
        p.curve(fn.f, col, 2.4);
        if (showDeriv) p.curve(fn.d, col, 1.4, [5, 4]);
        const px = p.X(3.95), py = p.Y(Math.max(-1.5, Math.min(3.1, fn.f(3.95))));
        labels.push({ t: fn.label, x: px, y: py - 9, col: col });
      });
      // Sigmoid/Tanh 在 x=3.95 处几乎重合，ReLU/GELU/SiLU/Softplus 也是，
      // 直接画标签会叠成一团。按 y 排序后强制拉开最小间距再画。
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) {
        if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
      }
      labels.forEach((l) => p.label(l.t, l.x, l.y, l.col, 'right', 11));
      p.label('x', p.X(4.2) - 4, p.Y(0) + 12, ink4(), 'center');
      p.label('y', p.X(0) + 10, p.Y(3.2) + 4, ink4());
      if (info) {
        info.innerHTML = active.length
          ? active.map((c) => {
              const k = c.getAttribute('data-fn'), fn = ACT[k];
              const col = PAL[chips.indexOf(c) % PAL.length];
              // 固定两行：一行放 f(-1)/f(0)，一行放 f(1)/f'(0)，
              // 这样无论容器多窄都不会折成三行。
              return '<b style="color:' + col + '">' + fn.label.padEnd(11) + '</b>' +
                     'f(-1)=' + fn.f(-1).toFixed(3) + '  f(0)=' + fn.f(0).toFixed(3) + '\n' +
                     ' '.repeat(11) +
                     'f(1)=' + fn.f(1).toFixed(3) + '   f\'(0)=' + fn.d(0).toFixed(3);
            }).join('\n')
          : '至少选择一个函数';
      }
    }

    chips.forEach((c) => c.addEventListener('click', () => { c.classList.toggle('is-on'); draw(); }));
    if (derivBtn) derivBtn.addEventListener('click', () => {
      showDeriv = !showDeriv;
      derivBtn.setAttribute('aria-pressed', String(showDeriv));
      derivBtn.textContent = showDeriv ? '隐藏导数' : '显示导数';
      draw();
    });
    return { draw };
  };

  /* ==========================================================================
     组件 2 · 单个神经元 / 感知机
     ========================================================================== */
  registry.neuron = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const outVerdict = el.querySelector('[data-role="verdict"]');
    const inputs = {};
    $$('input[data-p]', el).forEach((s) => {
      const key = s.getAttribute('data-p');
      inputs[key] = s;
      const lab = el.querySelector('[data-v="' + key + '"]');
      const upd = () => { if (lab) lab.textContent = (+s.value).toFixed(2); };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });

    const PTS = [
      [1.6, 1.7, 1], [2.2, 0.9, 1], [0.9, 2.3, 1],
      [-1.7, -1.1, 0], [-0.9, -2.1, 0], [-2.1, -1.6, 0],
      [1.9, -1.4, 1], [-1.3, 1.9, 0]
    ];

    function draw() {
      const { ctx, w, h } = fit(cv);
      const v = (k) => +inputs[k].value;
      const w1 = v('w1'), w2 = v('w2'), b = v('b'), x1 = v('x1'), x2 = v('x2');
      const z = w1 * x1 + w2 * x2 + b;
      const a = 1 / (1 + Math.exp(-z));

      const p = makePlot(ctx, w, h, { xmin: -3, xmax: 3, ymin: -3, ymax: 3, pad: 30 });
      p.grid(6, 6);

      // 决策边界： w1*x + w2*y + b = 0
      if (Math.abs(w2) > 1e-6) {
        p.curve((x) => (-w1 * x - b) / w2, '#3b6ef5', 2.4);
      } else if (Math.abs(w1) > 1e-6) {
        const xx = -b / w1;
        ctx.save(); ctx.strokeStyle = '#3b6ef5'; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(p.X(xx), p.Y(-3)); ctx.lineTo(p.X(xx), p.Y(3)); ctx.stroke(); ctx.restore();
      }
      // 正类方向
      const nx = w1 / (Math.hypot(w1, w2) || 1), ny = w2 / (Math.hypot(w1, w2) || 1);
      ctx.save();
      ctx.strokeStyle = 'rgba(59,110,245,.5)'; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(p.X(0), p.Y(0)); ctx.lineTo(p.X(nx * 1.1), p.Y(ny * 1.1)); ctx.stroke();
      ctx.restore();

      // 数据点
      PTS.forEach((pt) => {
        const pred = (w1 * pt[0] + w2 * pt[1] + b) > 0 ? 1 : 0;
        const cx = p.X(pt[0]), cy = p.Y(pt[1]);
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
        ctx.fillStyle = pt[2] ? '#3b6ef5' : '#e0483f';
        ctx.fill();
        if (pred !== pt[2]) {
          ctx.lineWidth = 2.2; ctx.strokeStyle = '#c2410c';
          ctx.beginPath(); ctx.arc(cx, cy, 10.5, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.restore();
      });

      // 当前输入点
      ctx.save();
      ctx.beginPath(); ctx.arc(p.X(x1), p.Y(x2), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#0f9d8f'; ctx.fill();
      ctx.strokeStyle = surface(); ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();

      let wrong = 0;
      PTS.forEach((pt) => { if (((w1 * pt[0] + w2 * pt[1] + b) > 0 ? 1 : 0) !== pt[2]) wrong++; });

      if (out) {
        out.innerHTML =
          'z = w1·x1 + w2·x2 + b\n' +
          '  = ' + w1.toFixed(2) + '×' + x1.toFixed(2) + ' + ' + w2.toFixed(2) + '×' + x2.toFixed(2) + ' + ' + b.toFixed(2) + '\n' +
          '  = <b>' + z.toFixed(3) + '</b>\n' +
          'a = σ(z) = <b>' + a.toFixed(3) + '</b>   (阈值 0.5 时判为 ' + (a > 0.5 ? '正类' : '负类') + ')';
      }
      if (outVerdict) {
        outVerdict.innerHTML = wrong === 0
          ? '<b style="color:#0f9d8f">✔ 全部分对！</b>这条直线就是这组数据的分界。'
          : '还有 <b style="color:#c2410c">' + wrong + '</b> 个点被分错（红圈标出）。拖动滑块把它们分开。';
      }
    }
    return { draw };
  };

  /* ==========================================================================
     组件 3 · 梯度下降轨迹
     ========================================================================== */
  registry.gd = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => { if (lab) lab.textContent = (+s.value).toFixed(s.getAttribute('data-p') === 'lr' ? 3 : 2); };
      s.addEventListener('input', () => { upd(); reset(); });
      upd();
    });
    const runBtn = el.querySelector('[data-role="run"]');
    const resetBtn = el.querySelector('[data-role="reset"]');
    const A = 1.0, B = 8.0;
    const loss = (p) => 0.5 * (A * p[0] * p[0] + B * p[1] * p[1]);
    const grad = (p) => [A * p[0], B * p[1]];
    const START = [-2.5, 1.55];
    let path = [START.slice()], vel = [0, 0], running = false, timer = null, step = 0;

    function reset() {
      path = [START.slice()]; vel = [0, 0]; step = 0;
      running = false;
      if (timer) { clearInterval(timer); timer = null; }
      if (runBtn) { runBtn.textContent = '开始下降'; runBtn.setAttribute('aria-pressed', 'false'); }
      draw();
    }

    function advance() {
      const lr = +sl.lr.value, mom = +sl.momentum.value;
      const p = path[path.length - 1];
      const g = grad(p);
      vel = [mom * vel[0] - lr * g[0], mom * vel[1] - lr * g[1]];
      const np = [p[0] + vel[0], p[1] + vel[1]];
      if (!isFinite(np[0]) || Math.abs(np[0]) > 60 || Math.abs(np[1]) > 60) {
        path.push([Math.max(-12, Math.min(12, np[0])), Math.max(-12, Math.min(12, np[1]))]);
        stop(true);
        return false;
      }
      path.push(np);
      step++;
      if (step >= 80) { stop(false); return false; }
      return true;
    }

    function stop(diverged) {
      running = false;
      if (timer) { clearInterval(timer); timer = null; }
      if (runBtn) { runBtn.textContent = diverged ? '已发散，重置' : '开始下降'; runBtn.setAttribute('aria-pressed', 'false'); }
      draw();
    }

    function draw() {
      const { ctx, w, h } = fit(cv);
      const p = makePlot(ctx, w, h, { xmin: -4, xmax: 4, ymin: -3, ymax: 3, pad: 26 });
      p.grid(8, 6);

      // 等值线（椭圆）
      ctx.save();
      for (let k = 1; k <= 14; k++) {
        const c = 0.05 + k * 0.075;
        ctx.beginPath();
        ctx.ellipse(p.X(0), p.Y(0), Math.abs(p.X(Math.sqrt(2 * c / A)) - p.X(0)), Math.abs(p.Y(0) - p.Y(Math.sqrt(2 * c / B))), 0, 0, Math.PI * 2);
        ctx.strokeStyle = k === 1 ? 'rgba(130,136,150,.5)' : GRID_SOFT;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();
      ctx.save();
      ctx.beginPath(); ctx.arc(p.X(0), p.Y(0), 3, 0, Math.PI * 2);
      ctx.fillStyle = '#0f9d8f'; ctx.fill(); ctx.restore();
      p.label('最优解', p.X(0) + 8, p.Y(0), '#0f9d8f', 'left', 11);

      // 路径
      ctx.save();
      ctx.beginPath();
      path.forEach((q, i) => { const x = p.X(q[0]), y = p.Y(q[1]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.strokeStyle = '#3b6ef5'; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.stroke();
      path.forEach((q, i) => {
        const x = p.X(q[0]), y = p.Y(q[1]);
        if (x < -50 || x > w + 50 || y < -50 || y > h + 50) return;
        ctx.beginPath(); ctx.arc(x, y, i === path.length - 1 ? 4.5 : 2.2, 0, Math.PI * 2);
        ctx.fillStyle = i === path.length - 1 ? '#c2410c' : 'rgba(59,110,245,.75)';
        ctx.fill();
      });
      ctx.restore();

      const cur = path[path.length - 1];
      const lr = +sl.lr.value, mom = +sl.momentum.value;
      if (out) {
        const effLr = lr / Math.max(0.05, 1 - mom);
        const ratio = effLr * B;
        const verdict = ratio < 1.2 ? '稳定收敛' : (ratio < 2 ? '来回震荡' : '发散！');
        out.innerHTML =
          '步数 <b>' + step + '</b>   位置 (' + cur[0].toFixed(3) + ', ' + cur[1].toFixed(3) + ')\n' +
          '损失 L = <b>' + loss(cur).toExponential(3) + '</b>   |∇L| = ' + Math.hypot(...grad(cur)).toExponential(2) + '\n' +
          '学习率 η = ' + lr.toFixed(3) + '   动量 β = ' + mom.toFixed(2) + '\n' +
          '等效步长 η/(1−β) = ' + effLr.toFixed(3) + '\n' +
          '稳定判据 η_eff × 曲率 8 = ' + ratio.toFixed(2) + '   →   <b>' + verdict + '</b>' +
          (ratio >= 1.2 ? '  （超过 2 就会在陡峭方向炸开）' : '');
      }
    }

    if (runBtn) runBtn.addEventListener('click', () => {
      if (running) { stop(false); return; }
      if (step >= 80) reset();
      running = true;
      runBtn.textContent = '暂停';
      runBtn.setAttribute('aria-pressed', 'true');
      timer = setInterval(() => { if (!advance()) draw(); else draw(); }, 220);
    });
    if (resetBtn) resetBtn.addEventListener('click', reset);
    reset();
    return { draw };
  };

  /* ==========================================================================
     组件 4 · 反向传播逐步演示（2-2-1 网络）
     ========================================================================== */
  registry.backprop = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const stepBox = el.querySelector('[data-role="step"]');
    const prevBtn = el.querySelector('[data-role="prev"]');
    const nextBtn = el.querySelector('[data-role="next"]');
    const resetBtn = el.querySelector('[data-role="reset"]');

    const X = [1.0, 0.5], Y = 1.0, LR = 0.5;
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const dsig = (a) => a * (1 - a);

    const W1 = [[0.80, -0.40], [0.30, 0.90]], B1 = [0.10, -0.20];
    const W2 = [0.60, -0.70], B2 = 0.05;

    // 前向
    const z1 = [W1[0][0] * X[0] + W1[0][1] * X[1] + B1[0], W1[1][0] * X[0] + W1[1][1] * X[1] + B1[1]];
    const a1 = z1.map(sig);
    const z2 = W2[0] * a1[0] + W2[1] * a1[1] + B2;
    const yhat = sig(z2);
    const L = 0.5 * (yhat - Y) * (yhat - Y);
    // 反向
    const d2 = (yhat - Y) * dsig(yhat);
    const gW2 = [d2 * a1[0], d2 * a1[1]];
    const gB2 = d2;
    const d1 = [W2[0] * d2 * dsig(a1[0]), W2[1] * d2 * dsig(a1[1])];
    const gW1 = [[d1[0] * X[0], d1[0] * X[1]], [d1[1] * X[0], d1[1] * X[1]]];
    const gB1 = [d1[0], d1[1]];

    const STEPS = [
      { t: '前向 · 输入', hl: ['in'], d: '把样本喂进去：x₁ = ' + X[0] + '，x₂ = ' + X[1] + '，正确答案 y = ' + Y + '。' },
      { t: '前向 · 隐藏层线性部分', hl: ['in', 'h'], d: 'z₁ = w₁₁x₁ + w₁₂x₂ + b₁ = ' + z1[0].toFixed(3) + '；z₂ = ' + z1[1].toFixed(3) + '。矩阵写法 z = W₁x + b₁。' },
      { t: '前向 · 隐藏层激活', hl: ['h'], d: 'a₁ = σ(z₁) = ' + a1[0].toFixed(4) + '，a₂ = ' + a1[1].toFixed(4) + '。非线性把直线弯成了曲线。' },
      { t: '前向 · 输出层', hl: ['h', 'out'], d: 'z = w₂₁a₁ + w₂₂a₂ + b₂ = ' + z2.toFixed(4) + '，ŷ = σ(z) = ' + yhat.toFixed(4) + '。' },
      { t: '损失', hl: ['out'], d: 'L = ½(ŷ − y)² = ½(' + yhat.toFixed(4) + ' − ' + Y + ')² = ' + L.toFixed(6) + '。这就是"错得有多离谱"。' },
      { t: '反向 · 输出层误差', hl: ['out'], d: 'δ = (ŷ − y) · σ′(z) = ' + (yhat - Y).toFixed(4) + ' × ' + dsig(yhat).toFixed(4) + ' = ' + d2.toFixed(5) + '。注意 σ′(z) 用激活值就能算出来：ŷ(1−ŷ)。' },
      { t: '反向 · 输出层梯度', hl: ['h', 'out'], d: '∂L/∂w₂₁ = δ·a₁ = ' + gW2[0].toFixed(5) + '；∂L/∂w₂₂ = δ·a₂ = ' + gW2[1].toFixed(5) + '。规律：本地误差 × 上游输入。' },
      { t: '反向 · 误差回传到隐藏层', hl: ['h'], d: 'δ₁ = (w₂₁δ)·σ′(z₁) = ' + d1[0].toFixed(5) + '；δ₂ = (w₂₂δ)·σ′(z₂) = ' + d1[1].toFixed(5) + '。误差按权重"原路退回"，再乘本层激活的导数。' },
      { t: '反向 · 隐藏层梯度', hl: ['in', 'h'], d: '∂L/∂w₁₁ = δ₁·x₁ = ' + gW1[0][0].toFixed(5) + '，∂L/∂w₁₂ = δ₁·x₂ = ' + gW1[0][1].toFixed(5) + '（第二行同理）。' },
      { t: '更新参数', hl: ['in', 'h', 'out'], d: 'w ← w − η·∂L/∂w，η = ' + LR + '。例如 w₁₁: ' + W1[0][0].toFixed(3) + ' → ' + (W1[0][0] - LR * gW1[0][0]).toFixed(4) + '。一次完整训练就结束了，下一轮损失会更小。' }
    ];

    let idx = 0;

    function draw() {
      const { ctx, w, h } = fit(cv);
      const hl = STEPS[idx].hl;
      const LAY = [[0.16, 0.30], [0.16, 0.70], [0.52, 0.22], [0.52, 0.78], [0.86, 0.5]];
      const pt = (i) => [LAY[i][0] * w, LAY[i][1] * h];
      const vals = [X[0], X[1], a1[0], a1[1], yhat];

      // 连线
      const edges = [
        [0, 2, W1[0][0]], [1, 2, W1[0][1]], [0, 3, W1[1][0]], [1, 3, W1[1][1]],
        [2, 4, W2[0]], [3, 4, W2[1]]
      ];
      edges.forEach(([a, b, wv]) => {
        const A = pt(a), B = pt(b);
        ctx.save();
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]);
        ctx.lineWidth = Math.min(7, 1 + Math.abs(wv) * 5);
        ctx.strokeStyle = wv >= 0 ? 'rgba(59,110,245,.55)' : 'rgba(224,72,63,.55)';
        ctx.stroke();
        ctx.restore();
      });

      // 节点
      const nodeHL = { in: [0, 1], h: [2, 3], out: [4] };
      const lit = new Set();
      Object.keys(nodeHL).forEach((k) => { if (hl.indexOf(k) >= 0) nodeHL[k].forEach((i) => lit.add(i)); });

      LAY.forEach((_, i) => {
        const [x, y] = pt(i);
        const on = lit.has(i);
        ctx.save();
        if (on) { ctx.shadowColor = 'rgba(59,110,245,.55)'; ctx.shadowBlur = 16; }
        ctx.beginPath(); ctx.arc(x, y, 17, 0, Math.PI * 2);
        const t = Math.max(0, Math.min(1, vals[i]));
        const g = ctx.createLinearGradient(x - 17, y - 17, x + 17, y + 17);
        g.addColorStop(0, 'rgba(59,110,245,' + (0.25 + 0.75 * t) + ')');
        g.addColorStop(1, 'rgba(124,58,237,' + (0.25 + 0.75 * t) + ')');
        ctx.fillStyle = g; ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = on ? '#3b6ef5' : 'rgba(130,136,150,.45)';
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#fff';
        ctx.font = '600 11px -apple-system,"PingFang SC",sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(vals[i].toFixed(2), x, y);
        ctx.restore();
      });

      const names = ['x₁', 'x₂', 'a₁', 'a₂', 'ŷ'];
      LAY.forEach((_, i) => {
        const [x, y] = pt(i);
        ctx.save();
        ctx.fillStyle = ink4();
        ctx.font = '11px -apple-system,"PingFang SC",sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(names[i], x, y + 22);
        ctx.restore();
      });

      if (stepBox) {
        stepBox.innerHTML = '<div style="font-weight:700;color:var(--ink);margin-bottom:.35rem">' +
          (idx + 1) + '/' + STEPS.length + ' · ' + STEPS[idx].t + '</div>' + STEPS[idx].d;
      }
      if (out) {
        out.innerHTML =
          'x = [' + X[0].toFixed(1) + ', ' + X[1].toFixed(1) + ']   y = ' + Y.toFixed(1) + '\n' +
          'z₁ = [' + z1[0].toFixed(3) + ', ' + z1[1].toFixed(3) + ']   a₁ = [' + a1[0].toFixed(3) + ', ' + a1[1].toFixed(3) + ']\n' +
          'z  = ' + z2.toFixed(3) + '   ŷ = ' + yhat.toFixed(3) + '\n' +
          'L  = ' + L.toFixed(5) + '\n' +
          'δ₂ = ' + d2.toFixed(5) + '   δ₁ = [' + d1[0].toFixed(5) + ', ' + d1[1].toFixed(5) + ']\n' +
          '∂L/∂W₁ = [[' + gW1[0][0].toFixed(4) + ', ' + gW1[0][1].toFixed(4) + '], [' + gW1[1][0].toFixed(4) + ', ' + gW1[1][1].toFixed(4) + ']]\n' +
          '∂L/∂W₂ = [' + gW2[0].toFixed(4) + ', ' + gW2[1].toFixed(4) + ']';
      }
      if (prevBtn) prevBtn.disabled = idx === 0;
      if (nextBtn) nextBtn.disabled = idx === STEPS.length - 1;
    }

    if (prevBtn) prevBtn.addEventListener('click', () => { if (idx > 0) { idx--; draw(); } });
    if (nextBtn) nextBtn.addEventListener('click', () => { if (idx < STEPS.length - 1) { idx++; draw(); } });
    if (resetBtn) resetBtn.addEventListener('click', () => { idx = 0; draw(); });
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 5 · Softmax 与温度
     ========================================================================== */
  registry.softmax = function (el) {
    const bars = el.querySelector('[data-role="bars"]');
    const out = el.querySelector('[data-role="info"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => { if (lab) lab.textContent = (+s.value).toFixed(s.getAttribute('data-p') === 'T' ? 2 : 1); };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const WORDS = ['猫', '狗', '鱼', '汽车'];

    function draw() {
      const T = Math.max(0.05, +sl.T.value);
      const logits = [+sl.l1.value, +sl.l2.value, +sl.l3.value, +sl.l4.value];
      const mx = Math.max.apply(null, logits);
      const ex = logits.map((l) => Math.exp((l - mx) / T));
      const s = ex.reduce((a, b) => a + b, 0);
      const probs = ex.map((e) => e / s);
      const H = -probs.reduce((a, p) => a + (p > 1e-12 ? p * Math.log(p) : 0), 0);

      bars.innerHTML = probs.map((p, i) =>
        '<div class="bar-row' + (p > 0.5 ? ' bar-row--hot' : '') + '">' +
          '<span class="bar-row__name">' + WORDS[i] + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (p * 100).toFixed(2) + '%"></span></span>' +
          '<span class="bar-row__val">' + (p * 100).toFixed(1) + '%</span>' +
        '</div>').join('');

      if (out) {
        out.innerHTML =
          'logits  = [' + logits.join(', ') + ']\n' +
          'T       = ' + T.toFixed(2) + '\n' +
          '概率    = [' + probs.map((p) => p.toFixed(3)).join(', ') + ']\n' +
          '最大概率 <b>' + (Math.max.apply(null, probs) * 100).toFixed(1) + '%</b>   ' +
          '熵 <b>' + H.toFixed(3) + '</b> nats' + (T < 0.4 ? '  （分布很尖锐）' : T > 1.6 ? '  （分布被抹平）' : '');
      }
    }
    return { draw };
  };

  /* ==========================================================================
     组件 6 · 注意力矩阵
     ========================================================================== */
  registry.attention = function (el) {
    const grid = el.querySelector('[data-role="grid"]');
    const bars = el.querySelector('[data-role="bars"]');
    const note = el.querySelector('[data-role="info"]');
    const TOK = ['小猫', '坐在', '垫子', '上', '因为', '它', '很', '累'];
    // 示意用的注意力权重（每行和为 1），模拟一个训练好的模型学到的模式
    const RAW = [
      [0.42, 0.20, 0.14, 0.10, 0.05, 0.05, 0.02, 0.02],
      [0.30, 0.28, 0.18, 0.12, 0.04, 0.04, 0.02, 0.02],
      [0.22, 0.16, 0.32, 0.18, 0.05, 0.04, 0.02, 0.01],
      [0.16, 0.14, 0.24, 0.30, 0.06, 0.05, 0.03, 0.02],
      [0.10, 0.08, 0.10, 0.10, 0.34, 0.16, 0.07, 0.05],
      [0.34, 0.06, 0.20, 0.04, 0.05, 0.14, 0.08, 0.09],
      [0.05, 0.04, 0.06, 0.05, 0.10, 0.12, 0.34, 0.24],
      [0.06, 0.04, 0.05, 0.04, 0.08, 0.20, 0.26, 0.27]
    ];
    let sel = 5;

    function draw() {
      grid.innerHTML =
        '<div style="display:grid;grid-template-columns:4.6rem repeat(' + TOK.length + ',1fr);gap:2px">' +
        '<div></div>' +
        TOK.map((t) => '<div style="font-size:.66rem;color:var(--ink-4);text-align:center;padding-bottom:2px;font-family:var(--font-mono)">' + t + '</div>').join('') +
        TOK.map((rt, i) =>
          '<div style="font-size:.7rem;color:var(--ink-3);text-align:right;padding-right:.4rem;line-height:1.9;font-family:var(--font-mono)">' + rt + '</div>' +
          RAW[i].map((v, j) =>
            '<div title="' + TOK[i] + ' → ' + TOK[j] + ' : ' + v.toFixed(3) + '" ' +
            'style="aspect-ratio:1;border-radius:4px;cursor:pointer;' +
            'background:rgba(59,110,245,' + (0.06 + 0.94 * v / 0.42) + ');' +
            (i === sel ? 'outline:1.5px solid var(--accent);outline-offset:1px;' : '') +
            '" data-i="' + i + '" data-j="' + j + '"></div>').join('')
        ).join('') + '</div>';

      grid.querySelectorAll('[data-i]').forEach((c) => {
        c.addEventListener('click', () => { sel = +c.getAttribute('data-i'); draw(); });
      });

      const row = RAW[sel];
      bars.innerHTML = TOK.map((t, j) =>
        '<div class="bar-row' + (row[j] > 0.2 ? ' bar-row--hot' : '') + '">' +
          '<span class="bar-row__name">' + t + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (row[j] / 0.42 * 100).toFixed(1) + '%"></span></span>' +
          '<span class="bar-row__val">' + row[j].toFixed(2) + '</span>' +
        '</div>').join('');

      if (note) {
        note.innerHTML = '当前查询词是 <b>' + TOK[sel] + '</b>，它把注意力主要分给了 <b>' +
          TOK[row.indexOf(Math.max.apply(null, row))] + '</b>（权重 ' + Math.max.apply(null, row).toFixed(2) + '）。' +
          '真实模型里这些数字来自 softmax(QKᵀ/√d)，是训练出来的，不是人写的。';
      }
    }
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 7 · 卷积滑动窗口
     ========================================================================== */
  registry.conv = function (el) {
    const inCv = el.querySelector('[data-role="input"]');
    const outCv = el.querySelector('[data-role="output"]');
    const kerBox = el.querySelector('[data-role="kernel"]');
    const out = el.querySelector('[data-role="info"]');
    const playBtn = el.querySelector('[data-role="play"]');
    const stepBtn = el.querySelector('[data-role="step"]');
    const resetBtn = el.querySelector('[data-role="reset"]');

    const IMG = [
      [0, 0, 1, 1, 0, 0],
      [0, 1, 1, 1, 1, 0],
      [1, 1, 0, 0, 1, 1],
      [1, 1, 0, 0, 1, 1],
      [0, 1, 1, 1, 1, 0],
      [0, 0, 1, 1, 0, 0]
    ];
    const KERNELS = {
      edge:   { name: '垂直边缘', k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
      edgeh:  { name: '水平边缘', k: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
      blur:   { name: '均值模糊', k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
      sharpen:{ name: '锐化', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
      sobel:  { name: 'Sobel 边缘', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] }
    };
    let cur = 'edge', pos = 0, timer = null;
    const OUT = 4;

    function compute() {
      const K = KERNELS[cur].k;
      const res = [];
      for (let i = 0; i < OUT; i++) {
        const row = [];
        for (let j = 0; j < OUT; j++) {
          let s = 0;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += IMG[i + a][j + b] * K[a][b];
          row.push(s);
        }
        res.push(row);
      }
      return res;
    }

    function drawMatrix(cv, data, opt) {
      const { ctx, w, h } = fit(cv);
      const n = data.length, m = data[0].length;
      const cell = Math.min((w - 4) / m, (h - 4) / n);
      const ox = (w - cell * m) / 2, oy = (h - cell * n) / 2;
      const all = data.flat();
      const mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
      const rng = (mx - mn) || 1;
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
        const t = (data[i][j] - mn) / rng;
        const x = ox + j * cell, y = oy + i * cell;
        ctx.fillStyle = 'rgba(59,110,245,' + (0.08 + 0.85 * t) + ')';
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        if (cell > 22) {
          ctx.fillStyle = t > 0.6 ? '#fff' : ink2();
          ctx.font = (cell > 34 ? 11 : 9) + 'px ' + '-apple-system,"PingFang SC",sans-serif';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(data[i][j].toFixed(data[i][j] % 1 ? 2 : 0), x + cell / 2, y + cell / 2);
        }
      }
      return { ctx, ox, oy, cell, n, m };
    }

    function draw() {
      const R = compute();
      const g = drawMatrix(inCv, IMG);
      // 高亮窗口
      const ci = Math.floor(pos / OUT), cj = pos % OUT;
      g.ctx.save();
      g.ctx.strokeStyle = '#c2410c'; g.ctx.lineWidth = 2.5;
      g.ctx.strokeRect(g.ox + cj * g.cell + 1, g.oy + ci * g.cell + 1, g.cell * 3 - 2, g.cell * 3 - 2);
      g.ctx.restore();

      const og = drawMatrix(outCv, R);
      og.ctx.save();
      og.ctx.strokeStyle = '#c2410c'; og.ctx.lineWidth = 2.5;
      og.ctx.strokeRect(og.ox + cj * og.cell + 1, og.oy + ci * og.cell + 1, og.cell - 2, og.cell - 2);
      og.ctx.restore();

      const K = KERNELS[cur].k;
      kerBox.innerHTML = K.map((r) => r.map((v) =>
        '<div style="background:rgba(59,110,245,' + (0.06 + 0.85 * (v - Math.min.apply(null, K.flat())) / ((Math.max.apply(null, K.flat()) - Math.min.apply(null, K.flat())) || 1)) + ');' +
        'border-radius:4px;padding:.35rem 0;text-align:center;font-family:var(--font-mono);font-size:.72rem;' +
        'color:' + (v > (Math.max.apply(null, K.flat()) * 0.5) ? '#fff' : 'var(--ink-2)') + '">' + v.toFixed(2).replace(/\.00$/, '') + '</div>').join('')).join('');

      // 计算过程
      let s = '';
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const iv = IMG[ci + a][cj + b], kv = K[a][b];
        s += (s ? ' + ' : '') + iv + '×' + (Math.abs(kv) < 1 ? kv.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') : kv);
      }
      if (out) {
        out.innerHTML =
          '卷积核：<b>' + KERNELS[cur].name + '</b>\n' +
          '窗口位置：输出第 ' + (ci + 1) + ' 行第 ' + (cj + 1) + ' 列\n' +
          '计算：' + s + '\n' +
          '结果：<b>' + R[ci][cj].toFixed(3) + '</b>\n' +
          '输出尺寸 4×4（6−3+1），参数只有 <b>9 个</b>（外加 1 个偏置）';
      }
    }

    $$('[data-k]', el).forEach((b) => b.addEventListener('click', () => {
      cur = b.getAttribute('data-k');
      $$('[data-k]', el).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      pos = 0; draw();
    }));
    if (stepBtn) stepBtn.addEventListener('click', () => { pos = (pos + 1) % (OUT * OUT); draw(); });
    if (resetBtn) resetBtn.addEventListener('click', () => { pos = 0; draw(); });
    if (playBtn) playBtn.addEventListener('click', () => {
      if (timer) { clearInterval(timer); timer = null; playBtn.textContent = '自动播放'; playBtn.setAttribute('aria-pressed', 'false'); return; }
      playBtn.textContent = '暂停'; playBtn.setAttribute('aria-pressed', 'true');
      timer = setInterval(() => { pos = (pos + 1) % (OUT * OUT); draw(); }, 500);
    });
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 8 · 分词演示
     ========================================================================== */
  registry.tokenizer = function (el) {
    const input = el.querySelector('input[data-role="text"]');
    const outChips = el.querySelector('[data-role="chips"]');
    const out = el.querySelector('[data-role="info"]');

    const VOCAB = ['the', 'ing', 'tion', 'ation', 'ization', 'pre', 'un', 'able', 'ness', 'ment',
      'trans', 'former', 'attention', 'attention', 'model', 'learn', 'token', 'embed', 'deep',
      'neural', 'network', 'language', 'large', 'scale', 'data', 'train', 'gradient', 'descent',
      'back', 'prop', 'layer', 'weight', 'loss', 'output', 'input', 'self', 'multi', 'head',
      'position', 'encode', 'decode', 'context', 'prompt', 'fine', 'tune', 'align', 'agent'];

    function splitWord(w) {
      const low = w.toLowerCase();
      if (low.length <= 2) return [w];
      const parts = [];
      let i = 0, guard = 0;
      while (i < low.length && guard++ < 40) {
        let best = '';
        for (let L = Math.min(12, low.length - i); L >= 3; L--) {
          const cand = low.slice(i, i + L);
          if (VOCAB.indexOf(cand) >= 0) { best = cand; break; }
        }
        if (!best) { best = low[i]; }
        parts.push(w.slice(i, i + best.length));
        i += best.length;
      }
      return parts;
    }

    function draw() {
      const text = (input && input.value) || '';
      const toks = [];
      text.split(/(\s+)/).forEach((seg) => {
        if (!seg) return;
        if (/^\s+$/.test(seg)) { toks.push({ t: '·', ws: true }); return; }
        if (/[\u4e00-\u9fa5]/.test(seg)) {
          seg.split('').forEach((ch) => toks.push({ t: ch, cn: true }));
        } else {
          const m = /^([^\w]*)([\w']+)([^\w]*)$/.exec(seg);
          if (!m) { toks.push({ t: seg }); return; }
          if (m[1]) toks.push({ t: m[1] });
          splitWord(m[2]).forEach((p) => toks.push({ t: p }));
          if (m[3]) toks.push({ t: m[3] });
        }
      });
      outChips.innerHTML = toks.map((t) =>
        t.ws ? '<span class="chip" style="background:transparent;border-color:transparent;color:var(--ink-4)">·</span>'
             : '<span class="chip' + (t.cn ? '' : ' chip--sub') + '">' + t.t.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</span>'
      ).join('');
      const real = toks.filter((t) => !t.ws).length;
      const chars = text.replace(/\s/g, '').length;
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      if (out) {
        out.innerHTML =
          '字符数 <b>' + chars + '</b>   空格切词 <b>' + words + '</b>   子词 token <b>' + real + '</b>\n' +
          '压缩比：' + (chars / (real || 1)).toFixed(2) + ' 字符/token\n' +
          '注意：这只是<b>示意</b>。真实的 BPE 词表是从几十 GB 语料里统计出来的，' +
          '常见词是 1 个 token，罕见词会被切成多个碎片。';
      }
    }
    if (input) input.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 9 · 采样策略
     ========================================================================== */
  registry.sampling = function (el) {
    const bars = el.querySelector('[data-role="bars"]');
    const out = el.querySelector('[data-role="info"]');
    const pickBtn = el.querySelector('[data-role="pick"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => {
        if (!lab) return;
        const k = s.getAttribute('data-p');
        lab.textContent = k === 'k' ? s.value : (+s.value).toFixed(2);
      };
      s.addEventListener('input', () => { upd(); picked = -1; draw(); });
      upd();
    });
    const TOK = ['的', '了', '是', '在', '这', '人', '不', '有'];
    const LOGITS = [4.2, 3.6, 3.1, 2.4, 1.8, 1.2, 0.6, -0.4];
    let picked = -1;

    function probs() {
      const T = Math.max(0.05, +sl.temp.value);
      const mx = Math.max.apply(null, LOGITS);
      let e = LOGITS.map((l) => Math.exp((l - mx) / T));
      let s = e.reduce((a, b) => a + b, 0);
      let p = e.map((x) => x / s);
      const k = +sl.k.value;
      const top = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      const keep = new Set();
      for (let i = 0; i < k; i++) keep.add(top[i][1]);
      let cum = 0;
      for (let i = 0; i < top.length; i++) {
        cum += top[i][0];
        if (cum <= +sl.p.value + 1e-9 || i === 0) keep.add(top[i][1]);
      }
      const q = p.map((v, i) => (keep.has(i) ? v : 0));
      const qs = q.reduce((a, b) => a + b, 0);
      return { p: p, q: q.map((x) => x / qs), kept: keep };
    }

    function draw() {
      const r = probs();
      bars.innerHTML = TOK.map((t, i) =>
        '<div class="bar-row' + (i === picked ? ' bar-row--hot' : '') + '">' +
          '<span class="bar-row__name">' + t + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (r.q[i] * 100).toFixed(2) + '%;' +
            (r.kept.has(i) ? '' : 'opacity:.25') + '"></span></span>' +
          '<span class="bar-row__val">' + (r.q[i] * 100).toFixed(1) + '%</span>' +
        '</div>').join('');
      if (out) {
        const n = r.kept.size;
        out.innerHTML =
          '候选词：' + TOK.join(' / ') + '\n' +
          '原始概率 = [' + r.p.map((x) => x.toFixed(3)).join(', ') + ']\n' +
          '保留 <b>' + n + '</b> 个候选（top-k = ' + sl.k.value + '，top-p = ' + (+sl.p.value).toFixed(2) + '）\n' +
          '重归一化后 = [' + r.q.map((x) => x.toFixed(3)).join(', ') + ']' +
          (picked >= 0 ? '\n本次抽中：<b>' + TOK[picked] + '</b>' : '');
      }
    }

    if (pickBtn) pickBtn.addEventListener('click', () => {
      const r = probs();
      let u = Math.random(), acc = 0;
      picked = TOK.length - 1;
      for (let i = 0; i < r.q.length; i++) { acc += r.q[i]; if (u <= acc) { picked = i; break; } }
      draw();
    });
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 10 · 位置编码热力图
     ========================================================================== */
  registry.posenc = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    function draw() {
      const { ctx, w, h } = fit(cv);
      const P = 40, Dm = 64;
      const cw = (w - 46) / Dm, ch = (h - 30) / P;
      const maxpos = +((el.querySelector('[data-p="maxpos"]') || {}).value || P);
      for (let i = 0; i < P; i++) {
        for (let d = 0; d < Dm; d++) {
          const k = Math.floor(d / 2);
          const denom = Math.pow(10000, 2 * k / Dm);
          const v = d % 2 === 0 ? Math.sin(i / denom) : Math.cos(i / denom);
          const a = Math.abs(v);
          ctx.fillStyle = v >= 0
            ? 'rgba(59,110,245,' + (0.08 + 0.85 * a) + ')'
            : 'rgba(224,72,63,' + (0.08 + 0.85 * a) + ')';
          ctx.fillRect(44 + d * cw, 4 + i * ch, Math.max(1, cw - 0.6), Math.max(1, ch - 0.6));
        }
      }
      ctx.save();
      ctx.fillStyle = ink4();
      ctx.font = '10px -apple-system,"PingFang SC",sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let i = 0; i < P; i += 8) ctx.fillText('t=' + i, 40, 4 + i * ch + ch / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('维度 d →', 44 + (Dm * cw) / 2, 4 + P * ch + 5);
      ctx.restore();
      if (out) out.innerHTML =
        '蓝 = 正值，红 = 负值，颜色越深绝对值越大\n' +
        '每一行是一个位置的"指纹"；越靠左的维度振荡越快，越靠右越慢\n' +
        '因为不同维度频率不同，每个位置都拿到了一串独一无二的编码';
    }
    const s = el.querySelector('input[data-p="maxpos"]');
    if (s) s.addEventListener('input', () => {
      const lab = el.querySelector('[data-v="maxpos"]');
      if (lab) lab.textContent = s.value;
      draw();
    });
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 11 · 过拟合 / 欠拟合
     ========================================================================== */
  registry.overfit = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const degIn = el.querySelector('input[data-p="deg"]');

    const TRUE = (x) => Math.sin(2 * Math.PI * x);
    // 固定的训练点（伪随机但可复现）
    const PTS = (function () {
      const xs = [0.04, 0.11, 0.19, 0.27, 0.34, 0.41, 0.49, 0.56, 0.63, 0.71, 0.78, 0.86, 0.94];
      const ns = [0.19, -0.24, 0.31, -0.14, 0.27, -0.29, 0.12, 0.24, -0.31, 0.16, -0.21, 0.28, -0.12];
      return xs.map((x, i) => [x, TRUE(x) + ns[i]]);
    })();

    function polyfit(xs, ys, deg) {
      const n = deg + 1;
      const A = [], b = [];
      for (let i = 0; i < n; i++) { A.push(new Array(n).fill(0)); b.push(0); }
      for (let k = 0; k < xs.length; k++) {
        const powers = [];
        for (let p = 0; p < 2 * deg + 1; p++) powers.push(Math.pow(xs[k], p));
        for (let i = 0; i < n; i++) {
          b[i] += powers[i] * ys[k];
          for (let j = 0; j < n; j++) A[i][j] += powers[i + j];
        }
      }
      for (let i = 0; i < n; i++) A[i][i] += 1e-9;   // 轻微岭回归，保证数值稳定
      // 高斯消元
      for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
        const t = A[c]; A[c] = A[piv]; A[piv] = t;
        const tb = b[c]; b[c] = b[piv]; b[piv] = tb;
        if (Math.abs(A[c][c]) < 1e-14) continue;
        for (let r = 0; r < n; r++) {
          if (r === c) continue;
          const f = A[r][c] / A[c][c];
          for (let cc = c; cc < n; cc++) A[r][cc] -= f * A[c][cc];
          b[r] -= f * b[c];
        }
      }
      const coef = new Array(n).fill(0);
      for (let i = 0; i < n; i++) coef[i] = Math.abs(A[i][i]) < 1e-14 ? 0 : b[i] / A[i][i];
      return coef;
    }

    function draw() {
      const deg = +degIn.value;
      const lab = el.querySelector('[data-v="deg"]');
      if (lab) lab.textContent = deg;
      const coef = polyfit(PTS.map((p) => p[0]), PTS.map((p) => p[1]), deg);
      const f = (x) => coef.reduce((a, c, i) => a + c * Math.pow(x, i), 0);

      const { ctx, w, h } = fit(cv);
      const p = makePlot(ctx, w, h, { xmin: -0.05, xmax: 1.05, ymin: -2.4, ymax: 2.4, pad: 28 });
      p.grid(10, 8);
      p.curve(TRUE, 'rgba(15,157,143,.85)', 2, [6, 5]);
      p.curve(f, '#3b6ef5', 2.6);
      PTS.forEach((pt) => {
        ctx.save();
        ctx.beginPath(); ctx.arc(p.X(pt[0]), p.Y(pt[1]), 3.6, 0, Math.PI * 2);
        ctx.fillStyle = '#c2410c'; ctx.fill();
        ctx.strokeStyle = surface(); ctx.lineWidth = 1.4; ctx.stroke();
        ctx.restore();
      });
      p.label('真实函数 sin(2πx)', p.X(0.02), p.Y(2.1), '#0f9d8f', 'left', 11);
      p.label('模型拟合曲线', p.X(0.02), p.Y(1.75), '#3b6ef5', 'left', 11);

      const train = PTS.reduce((a, pt) => a + Math.pow(f(pt[0]) - pt[1], 2), 0) / PTS.length;
      let test = 0;
      const N = 200;
      for (let i = 0; i < N; i++) {
        const x = i / (N - 1);
        test += Math.pow(f(x) - TRUE(x), 2);
      }
      test /= N;
      let verdict, col;
      if (deg <= 1) { verdict = '欠拟合：模型太简单，连训练数据都拟合不好'; col = '#c2410c'; }
      else if (deg <= 5) { verdict = '刚好：训练误差和泛化误差都低'; col = '#0f9d8f'; }
      else if (deg <= 8) { verdict = '开始过拟合：训练误差极低，但曲线在点之间开始乱窜'; col = '#c2410c'; }
      else { verdict = '严重过拟合：模型把噪声也背下来了，真实数据上会一塌糊涂'; col = '#e0483f'; }
      if (out) out.innerHTML =
        '多项式次数 d = <b>' + deg + '</b>   参数量 = ' + (deg + 1) + '\n' +
        '训练误差 MSE = <b>' + train.toFixed(4) + '</b>\n' +
        '测试误差 MSE = <b>' + test.toFixed(4) + '</b>\n' +
        '<b style="color:' + col + '">' + verdict + '</b>';
    }
    degIn.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 12 · 词向量相似度（余弦）
     ========================================================================== */
  registry.embedding = function (el) {
    const out = el.querySelector('[data-role="info"]');
    const bars = el.querySelector('[data-role="bars"]');
    const chips = $$('[data-w]', el);
    const WORDS = {
      '猫':   [0.92, 0.71, 0.05, 0.30],
      '狗':   [0.90, 0.68, 0.02, 0.41],
      '老虎': [0.85, 0.80, 0.09, 0.15],
      '汽车': [0.12, 0.31, 0.94, 0.22],
      '飞机': [0.08, 0.44, 0.90, 0.11],
      '跑':   [0.30, 0.18, 0.55, 0.88],
      '吃':   [0.36, 0.24, 0.30, 0.79]
    };
    let sel = '猫';
    function cos(a, b) {
      let d = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
      return d / (Math.sqrt(na) * Math.sqrt(nb));
    }
    function draw() {
      chips.forEach((c) => c.classList.toggle('is-on', c.getAttribute('data-w') === sel));
      const list = Object.keys(WORDS)
        .filter((w) => w !== sel)
        .map((w) => [w, cos(WORDS[sel], WORDS[w])])
        .sort((a, b) => b[1] - a[1]);
      bars.innerHTML = list.map(([w, v]) =>
        '<div class="bar-row' + (v > 0.9 ? ' bar-row--hot' : '') + '">' +
          '<span class="bar-row__name">' + w + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (Math.max(0, v) * 100).toFixed(1) + '%"></span></span>' +
          '<span class="bar-row__val">' + v.toFixed(3) + '</span>' +
        '</div>').join('');
      if (out) out.innerHTML =
        '与「<b>' + sel + '</b>」余弦相似度最高的词：<b>' + list[0][0] + '</b>（' + list[0][1].toFixed(3) + '）\n' +
        '最低的词：' + list[list.length - 1][0] + '（' + list[list.length - 1][1].toFixed(3) + '）\n' +
        '每个词只是 4 维向量，真实模型里是 768 ~ 8192 维\n' +
        '相似度 = cos(θ) = (a·b) / (|a||b|)，只看向量方向，不看长度';
    }
    chips.forEach((c) => c.addEventListener('click', () => { sel = c.getAttribute('data-w'); draw(); }));
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 13 · 直线拟合与损失（第 1 章）
     ========================================================================== */
  registry.linefit = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => { if (lab) lab.textContent = (+s.value).toFixed(2); };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const autoBtn = el.querySelector('[data-role="auto"]');

    const PTS = [
      [0.6, 1.3], [1.1, 1.7], [1.6, 2.6], [2.2, 3.2], [2.8, 3.4],
      [3.4, 4.4], [4.0, 4.9], [4.6, 5.2], [5.2, 6.3], [5.8, 6.1]
    ];
    const xbar = PTS.reduce((a, p) => a + p[0], 0) / PTS.length;
    const ybar = PTS.reduce((a, p) => a + p[1], 0) / PTS.length;
    const sxx = PTS.reduce((a, p) => a + (p[0] - xbar) ** 2, 0);
    const sxy = PTS.reduce((a, p) => a + (p[0] - xbar) * (p[1] - ybar), 0);
    const OPT = [sxy / sxx, ybar - (sxy / sxx) * xbar];

    function draw() {
      const w0 = +sl.w.value, b0 = +sl.b.value;
      const { ctx, w, h } = fit(cv);
      const p = makePlot(ctx, w, h, { xmin: -0.4, xmax: 6.6, ymin: -0.6, ymax: 8, pad: 32 });
      p.grid(7, 8);

      // 残差
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(194,65,12,.55)';
      ctx.lineWidth = 1.2;
      let mse = 0;
      PTS.forEach((pt) => {
        const pred = w0 * pt[0] + b0;
        mse += (pred - pt[1]) ** 2;
        ctx.beginPath();
        ctx.moveTo(p.X(pt[0]), p.Y(pt[1]));
        ctx.lineTo(p.X(pt[0]), p.Y(pred));
        ctx.stroke();
      });
      ctx.restore();
      mse /= PTS.length;

      p.curve((x) => w0 * x + b0, '#3b6ef5', 2.6);

      PTS.forEach((pt) => {
        ctx.save();
        ctx.beginPath(); ctx.arc(p.X(pt[0]), p.Y(pt[1]), 4, 0, Math.PI * 2);
        ctx.fillStyle = '#0f9d8f'; ctx.fill();
        ctx.strokeStyle = surface(); ctx.lineWidth = 1.6; ctx.stroke();
        ctx.restore();
      });

      // 最优线参考
      p.curve((x) => OPT[0] * x + OPT[1], 'rgba(15,157,143,.5)', 1.6, [6, 5]);

      if (out) {
        const off = Math.hypot(w0 - OPT[0], b0 - OPT[1]);
        out.innerHTML =
          '模型：ŷ = <b>' + w0.toFixed(2) + '</b> · x + <b>' + b0.toFixed(2) + '</b>\n' +
          '平均平方误差 MSE = <b>' + mse.toFixed(4) + '</b>\n' +
          '每个点错多少（残差）：' + PTS.slice(0, 4).map((pt) => (w0 * pt[0] + b0 - pt[1]).toFixed(2)).join(', ') + ' …\n' +
          '离最优参数还有 ' + off.toFixed(3) + ' 的距离' + (off < 0.02 ? '  <b>≈ 已到最优</b>' : '');
      }
    }

    if (autoBtn) autoBtn.addEventListener('click', () => {
      let t = 0;
      const w0 = +sl.w.value, b0 = +sl.b.value;
      const step = () => {
        t = Math.min(1, t + 0.08);
        const e = 1 - Math.pow(1 - t, 3);
        sl.w.value = (w0 + (OPT[0] - w0) * e).toFixed(2);
        sl.b.value = (b0 + (OPT[1] - b0) * e).toFixed(2);
        el.querySelector('[data-v="w"]').textContent = (+sl.w.value).toFixed(2);
        el.querySelector('[data-v="b"]').textContent = (+sl.b.value).toFixed(2);
        draw();
        if (t < 1) requestAnimationFrame(step);
      };
      step();
    });
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 14 · 两层网络切开 XOR（第 3 章）
     ========================================================================== */
  registry.xor = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const tIn = el.querySelector('input[data-p="t"]');
    const tLab = el.querySelector('[data-v="t"]');

    const sig = (z) => 1 / (1 + Math.exp(-z));
    // 训练前：小随机权重；训练后：手调的"或 / 与"组合
    const W1a = [[0.62, -0.41], [-0.53, 0.77]], b1a = [0.11, -0.22];
    const W2a = [0.34, 0.28], b2a = 0.05;
    const W1b = [[10, 10], [10, 10]], b1b = [-5, -15];
    const W2b = [10, -10], b2b = -5;
    const PTS = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
    const lerp = (a, b, t) => a + (b - a) * t;

    function params(t) {
      const W1 = [[lerp(W1a[0][0], W1b[0][0], t), lerp(W1a[0][1], W1b[0][1], t)],
                  [lerp(W1a[1][0], W1b[1][0], t), lerp(W1a[1][1], W1b[1][1], t)]];
      const b1 = [lerp(b1a[0], b1b[0], t), lerp(b1a[1], b1b[1], t)];
      const W2 = [lerp(W2a[0], W2b[0], t), lerp(W2a[1], W2b[1], t)];
      const b2 = lerp(b2a, b2b, t);
      return { W1, b1, W2, b2 };
    }

    function forward(x, p) {
      const h = [sig(p.W1[0][0] * x[0] + p.W1[0][1] * x[1] + p.b1[0]),
                 sig(p.W1[1][0] * x[0] + p.W1[1][1] * x[1] + p.b1[1])];
      return { h, y: sig(p.W2[0] * h[0] + p.W2[1] * h[1] + p.b2) };
    }

    function draw() {
      const t = +tIn.value;
      if (tLab) tLab.textContent = Math.round(t * 100) + '%';
      const p = params(t);
      const { ctx, w, h } = fit(cv);
      const pad = 34;
      const xmin = -0.45, xmax = 1.45, ymin = -0.45, ymax = 1.45;
      const X = (v) => pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad);
      const Y = (v) => h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad);

      // 输出热力图
      const N = 74;
      const cw = (w - 2 * pad) / N, ch = (h - 2 * pad) / N;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const x1 = xmin + (xmax - xmin) * (j + 0.5) / N;
          const x2 = ymax - (ymax - ymin) * (i + 0.5) / N;
          const y = forward([x1, x2], p).y;
          const a = Math.min(1, Math.abs(y - 0.5) * 2);
          ctx.fillStyle = y >= 0.5
            ? 'rgba(59,110,245,' + (0.05 + 0.55 * a) + ')'
            : 'rgba(224,72,63,' + (0.05 + 0.55 * a) + ')';
          ctx.fillRect(pad + j * cw, pad + i * ch, cw + 1, ch + 1);
        }
      }

      // 两个隐藏神经元的决策线
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = 'rgba(15,157,143,.95)';
      ctx.beginPath();
      ctx.moveTo(X(xmin), Y(-p.b1[0] / p.W1[0][1] - p.W1[0][0] / p.W1[0][1] * xmin));
      ctx.lineTo(X(xmax), Y(-p.b1[0] / p.W1[0][1] - p.W1[0][0] / p.W1[0][1] * xmax));
      ctx.stroke();
      ctx.strokeStyle = 'rgba(194,65,12,.95)';
      ctx.beginPath();
      ctx.moveTo(X(xmin), Y(-p.b1[1] / p.W1[1][1] - p.W1[1][0] / p.W1[1][1] * xmin));
      ctx.lineTo(X(xmax), Y(-p.b1[1] / p.W1[1][1] - p.W1[1][0] / p.W1[1][1] * xmax));
      ctx.stroke();
      ctx.restore();

      // 坐标框
      ctx.save();
      ctx.strokeStyle = 'rgba(130,136,150,.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(X(xmin), Y(ymax), X(xmax) - X(xmin), Y(ymin) - Y(ymax));
      ctx.restore();

      // 四个样本点
      let wrong = 0;
      PTS.forEach((pt) => {
        const y = forward([pt[0], pt[1]], p).y;
        const pred = y > 0.5 ? 1 : 0;
        if (pred !== pt[2]) wrong++;
        const cx = X(pt[0]), cy = Y(pt[1]);
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2);
        ctx.fillStyle = pt[2] ? '#3b6ef5' : '#e0483f';
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = pred === pt[2] ? '#fff' : '#111';
        ctx.stroke();
        ctx.restore();
      });

      if (out) {
        const lines = PTS.map((pt) => {
          const f = forward([pt[0], pt[1]], p);
          return '(' + pt[0] + ',' + pt[1] + ')  h=[' + f.h[0].toFixed(2) + ',' + f.h[1].toFixed(2) +
                 ']  ŷ=' + f.y.toFixed(3) + '  期望 ' + pt[2] + (f.y > 0.5 === (pt[2] === 1) ? '  ✓' : '  ✗');
        }).join('\n');
        out.innerHTML = '训练进度 ' + Math.round(t * 100) + '%   分错 <b>' + wrong + '/4</b>\n' + lines +
          '\n\n隐藏神经元 1（绿线）：学到「或」—— x₁+x₂ > ' + (-p.b1[0] / (p.W1[0][0] || 1)).toFixed(2) +
          '\n隐藏神经元 2（橙线）：学到「与」—— x₁+x₂ > ' + (-p.b1[1] / (p.W1[1][0] || 1)).toFixed(2) +
          '\n输出层：h₁ 且 非 h₂  →  正好就是 XOR';
      }
    }

    tIn.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 15 · 梯度随时间步衰减（第 8 章）
     ========================================================================== */
  registry.vanishing = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => { if (lab) lab.textContent = (+s.value).toFixed(2); };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const T = 30;

    function draw() {
      const w = +sl.w.value;
      const { ctx, w: W, h: H } = fit(cv);
      const padL = 46, padR = 14, padT = 18, padB = 34;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const barW = plotW / T;

      // 对数刻度：1e-12 ~ 1e+2
      const lo = -12, hi = 2.5;
      const toY = (v) => {
        const e = Math.log10(Math.max(1e-12, v));
        return padT + (hi - e) / (hi - lo) * plotH;
      };
      const toX = (t) => padL + t * barW;

      // 网格
      ctx.save();
      ctx.font = '10px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink4();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let e = 0; e >= lo; e -= 3) {
        const y = padT + (hi - e) / (hi - lo) * plotH;
        ctx.strokeStyle = e === 0 ? GRID : GRID_SOFT;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillText('1e' + e, padL - 7, y);
      }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let t = 0; t < T; t += 5) ctx.fillText('t=' + (t + 1), toX(t) + barW / 2, H - padB + 8);
      ctx.restore();

      // 柱子
      for (let t = 0; t < T; t++) {
        const g = Math.pow(w, t + 1);
        const y0 = toY(1), y1 = toY(g);
        const top = Math.min(y0, y1), hgt = Math.max(1.5, Math.abs(y1 - y0));
        ctx.fillStyle = g > 1
          ? 'rgba(224,72,63,.8)'
          : (g > 1e-3 ? 'rgba(59,110,245,.8)' : 'rgba(130,136,150,.45)');
        ctx.fillRect(toX(t) + 1, top, barW - 2.5, hgt);
      }

      const last = Math.pow(w, T);
      let verdict, col;
      if (w > 1.05) { verdict = '梯度爆炸：越往前的时刻梯度越大，训练会失控'; col = '#e0483f'; }
      else if (w > 0.95) { verdict = '刚好：梯度可以稳定地传回早期时刻（这就是 LSTM 门控想达到的效果）'; col = '#0f9d8f'; }
      else if (w > 0.8) { verdict = '缓慢衰减：还能记住十几步之前的信息'; col = '#c2410c'; }
      else { verdict = '梯度消失：30 步之前的梯度几乎为 0，网络记不住长期依赖'; col = '#e0483f'; }
      if (out) out.innerHTML =
        '循环权重 w = ' + w.toFixed(2) + '   （梯度 ∝ wᵗ）\n' +
        't=1 的梯度: ' + w.toFixed(4) + '\n' +
        't=10 的梯度: ' + Math.pow(w, 10).toExponential(3) + '\n' +
        't=30 的梯度: <b>' + last.toExponential(3) + '</b>\n' +
        '<b style="color:' + col + '">' + verdict + '</b>';
    }
    return { draw };
  };

  /* ==========================================================================
     组件 16 · Transformer 参数量计算器（第 10 章）
     ========================================================================== */
  registry.params = function (el) {
    const out = el.querySelector('[data-role="info"]');
    const bars = el.querySelector('[data-role="bars"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => {
        if (lab) lab.textContent = (+s.value).toLocaleString('en-US');
      };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const presets = $$('[data-preset]', el);

    const PRESETS = {
      'gpt2s':  { name: 'GPT-2 Small',  d: 768,  L: 12, V: 50257 },
      'gpt2l':  { name: 'GPT-2 Large',  d: 1280, L: 36, V: 50257 },
      'gpt3':   { name: 'GPT-3',        d: 12288, L: 96, V: 50257 },
      'llama7': { name: 'LLaMA-7B',     d: 4096, L: 32, V: 32000 },
      'llama70':{ name: 'LLaMA-70B',    d: 8192, L: 80, V: 32000 }
    };

    function draw() {
      const d = +sl.d.value, L = +sl.L.value, V = +sl.V.value;
      const dff = 4 * d;

      const attn = L * (4 * d * d);                 // QKV 三个投影 + 输出投影
      const ffn  = L * (2 * d * dff);               // 两个线性层
      const norm = L * (4 * d) + 2 * d;             // 每层两个 LayerNorm
      const emb  = V * d;                           // 词嵌入（权重共享，输出头不额外算）

      const total = attn + ffn + norm + emb;
      const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + ' B' : (n / 1e6).toFixed(1) + ' M';

      const rows = [
        ['注意力（每层 4d²）', attn, '#3b6ef5'],
        ['前馈网络（每层 8d²）', ffn, '#7c3aed'],
        ['词嵌入（V×d）', emb, '#0f9d8f'],
        ['归一化等', norm, '#c2410c']
      ];
      bars.innerHTML = rows.map(([n, v, c]) =>
        '<div class="bar-row">' +
          '<span class="bar-row__name" style="text-align:left;width:auto;font-size:.68rem">' + n.split('（')[0] + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (v / total * 100).toFixed(1) + '%;background:' + c + '"></span></span>' +
          '<span class="bar-row__val">' + fmt(v) + '</span>' +
        '</div>').join('');

      if (out) out.innerHTML =
        'd_model = ' + d.toLocaleString('en-US') + '   层数 = ' + L + '   词表 = ' + V.toLocaleString('en-US') + '\n' +
        'd_ff = 4 × d_model = ' + dff.toLocaleString('en-US') + '\n' +
        '────────────────────────────────────\n' +
        '每层参数 ≈ 12 d² = ' + fmt(12 * d * d) + '\n' +
        '总参数量 ≈ <b>' + fmt(total) + '</b>  (' + total.toLocaleString('en-US') + ')\n' +
        '────────────────────────────────────\n' +
        '注意力占 ' + (attn / total * 100).toFixed(1) + '%   ' +
        '前馈占 ' + (ffn / total * 100).toFixed(1) + '%   ' +
        '嵌入占 ' + (emb / total * 100).toFixed(1) + '%\n' +
        '用 fp16 存储约需 ' + (total * 2 / 1e9).toFixed(1) + ' GB 显存（仅权重）';
    }

    presets.forEach((b) => b.addEventListener('click', () => {
      const p = PRESETS[b.getAttribute('data-preset')];
      if (!p) return;
      sl.d.value = p.d; sl.L.value = p.L; sl.V.value = p.V;
      ['d', 'L', 'V'].forEach((k) => {
        const lab = el.querySelector('[data-v="' + k + '"]');
        if (lab) lab.textContent = (+sl[k].value).toLocaleString('en-US');
      });
      presets.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      draw();
    }));
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 17 · 训练算力与成本估算（第 12 章）
     ========================================================================== */
  registry.compute = function (el) {
    const out = el.querySelector('[data-role="info"]');
    const bars = el.querySelector('[data-role="bars"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => {
        if (!lab) return;
        const k = s.getAttribute('data-p');
        lab.textContent = k === 'mfu' ? (+s.value).toFixed(0) + '%' : (+s.value).toLocaleString('en-US');
      };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const presets = $$('[data-preset]', el);
    const PRESETS = {
      gpt2:  { N: 1.5, D: 40, G: 64, price: 2 },
      gpt3:  { N: 175, D: 300, G: 1024, price: 2 },
      llama: { N: 7, D: 2000, G: 256, price: 2 },
      chin:  { N: 70, D: 1400, G: 512, price: 2 },
      big:   { N: 400, D: 15000, G: 16384, price: 2 }
    };
    let gpuPeak = 989;   // H100 bf16 dense, TFLOPS

    function draw() {
      const N = +sl.N.value * 1e9;          // 参数量
      const D = +sl.D.value * 1e9;          // token 数
      const G = +sl.G.value;                // GPU 数量
      const mfu = +sl.mfu.value / 100;
      const price = +sl.price.value;

      const flops = 6 * N * D;              // 前向 2 + 反向 4
      const effPerGpu = gpuPeak * 1e12 * mfu;
      const gpuSeconds = flops / (G * effPerGpu);
      const gpuHours = gpuSeconds / 3600;
      const days = gpuHours / 24;
      const totalGpuHours = gpuHours * G;
      const cost = totalGpuHours * price;

      const fmtBig = (n) => n >= 1e24 ? (n / 1e24).toFixed(2) + ' Y'
        : n >= 1e21 ? (n / 1e21).toFixed(2) + ' Z'
        : n >= 1e18 ? (n / 1e18).toFixed(2) + ' E'
        : (n / 1e15).toFixed(2) + ' P';
      const fmtNum = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : Math.round(n).toLocaleString('en-US');

      bars.innerHTML =
        '<div class="bar-row"><span class="bar-row__name" style="text-align:left;width:auto">前向</span>' +
        '<span class="bar-row__track"><span class="bar-row__fill" style="width:33%;background:#3b6ef5"></span></span>' +
        '<span class="bar-row__val">2ND</span></div>' +
        '<div class="bar-row"><span class="bar-row__name" style="text-align:left;width:auto">反向</span>' +
        '<span class="bar-row__track"><span class="bar-row__fill" style="width:67%;background:#7c3aed"></span></span>' +
        '<span class="bar-row__val">4ND</span></div>';

      if (out) out.innerHTML =
        '总计算量 ≈ 6ND = <b>' + fmtBig(flops) + ' FLOPs</b>\n' +
        '单卡有效算力 = ' + (gpuPeak * mfu).toFixed(0) + ' TFLOPS（峰值 ' + gpuPeak + ' × MFU ' + (mfu * 100).toFixed(0) + '%）\n' +
        '────────────────────────────────────\n' +
        'GPU 数量：' + G.toLocaleString('en-US') + ' 张\n' +
        '训练时长：<b>' + days.toFixed(1) + ' 天</b>  （' + fmtNum(gpuHours) + ' 小时）\n' +
        '总 GPU 时：<b>' + fmtNum(totalGpuHours) + ' GPU·h</b>\n' +
        '────────────────────────────────────\n' +
        '按 $' + price + '/GPU·h 计：<b>$' + (cost >= 1e6 ? (cost / 1e6).toFixed(2) + 'M' : Math.round(cost).toLocaleString('en-US')) + '</b>' +
        '\n仅算力成本，不含数据、人力、失败的实验。';

      if (days > 365) {
        out.innerHTML += '\n⚠ 超过一年。现实中要么加卡，要么缩小模型。';
      }
    }

    $$('[data-gpu]', el).forEach((b) => b.addEventListener('click', () => {
      gpuPeak = +b.getAttribute('data-gpu');
      $$('[data-gpu]', el).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      draw();
    }));

    presets.forEach((b) => b.addEventListener('click', () => {
      const p = PRESETS[b.getAttribute('data-preset')];
      if (!p) return;
      sl.N.value = p.N; sl.D.value = p.D; sl.G.value = p.G;
      ['N', 'D', 'G'].forEach((k) => {
        const lab = el.querySelector('[data-v="' + k + '"]');
        if (lab) lab.textContent = (+sl[k].value).toLocaleString('en-US');
      });
      presets.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      draw();
    }));
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 18 · 推理显存估算（第 14 章）
     ========================================================================== */
  registry.vram = function (el) {
    const out = el.querySelector('[data-role="info"]');
    const bars = el.querySelector('[data-role="bars"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => {
        if (!lab) return;
        const k = s.getAttribute('data-p');
        lab.textContent = k === 'bits' ? s.value + ' bit' : (+s.value).toLocaleString('en-US');
      };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const presets = $$('[data-preset]', el);
    const PRESETS = {
      p7:  { N: 7, L: 32, H: 32, d: 4096 },
      p13: { N: 13, L: 40, H: 40, d: 5120 },
      p70: { N: 70, L: 80, H: 64, d: 8192 },
      p405:{ N: 405, L: 126, H: 128, d: 16384 }
    };

    function draw() {
      const N = +sl.N.value;               // 参数量（B）
      const bits = +sl.bits.value;
      const L = +sl.L.value;               // 层数
      const H = +sl.H.value;               // KV 头数
      const d = +sl.d.value;               // d_model
      const seq = +sl.seq.value;
      const batch = +sl.batch.value;

      const dHead = d / (H * 2);            // 假设 GQA：KV 头数是 Q 的一半
      // KV Cache = 2 (K和V) × L × H_kv × d_head × seq × batch × bytes
      const kvBytes = 2 * L * (H / 2) * dHead * seq * batch * 2;
      const kvGB = kvBytes / 1e9;

      const wGB = N * 1e9 * (bits / 8) / 1e9;

      const total = wGB + kvGB;
      const rows = [
        ['模型权重', wGB, '#3b6ef5'],
        ['KV Cache', kvGB, '#c2410c']
      ];
      bars.innerHTML = rows.map(([n, v, c]) =>
        '<div class="bar-row">' +
          '<span class="bar-row__name" style="text-align:left;width:auto">' + n + '</span>' +
          '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + (v / total * 100).toFixed(1) + '%;background:' + c + '"></span></span>' +
          '<span class="bar-row__val">' + v.toFixed(1) + 'G</span>' +
        '</div>').join('');

      const cards = [
        ['RTX 4090', 24], ['RTX 5090', 32], ['A100 40G', 40],
        ['A100 80G', 80], ['H100 80G', 80], ['H200 141G', 141]
      ];
      const fits = cards.map(([n, g]) =>
        (total <= g * 0.9 ? '✅ ' : '❌ ') + n + ' (' + g + 'G)').join('\n');

      if (out) out.innerHTML =
        '模型权重：' + N + 'B × ' + bits + ' bit = <b>' + wGB.toFixed(1) + ' GB</b>\n' +
        'KV Cache：' + L + ' 层 × ' + (H / 2) + ' 个 KV 头 × ' + dHead + ' 维 × ' + seq + ' token × ' + batch + ' 并发\n' +
        '         = <b>' + kvGB.toFixed(2) + ' GB</b>\n' +
        '────────────────────────────────────\n' +
        '合计约 <b>' + total.toFixed(1) + ' GB</b>（不含中间激活和框架开销，实际再加 10~20%）\n' +
        '────────────────────────────────────\n' +
        '能跑吗：\n' + fits;

      if (kvGB > wGB) {
        out.innerHTML += '\n\n⚠ KV Cache 已经超过模型权重本身——说明并发量或上下文长度是瓶颈，考虑 GQA、量化 KV 或减小 batch。';
      }
    }

    presets.forEach((b) => b.addEventListener('click', () => {
      const p = PRESETS[b.getAttribute('data-preset')];
      if (!p) return;
      sl.N.value = p.N; sl.L.value = p.L; sl.H.value = p.H; sl.d.value = p.d;
      ['N', 'L', 'H', 'd'].forEach((k) => {
        const lab = el.querySelector('[data-v="' + k + '"]');
        if (lab) lab.textContent = (+sl[k].value).toLocaleString('en-US');
      });
      presets.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      draw();
    }));
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 19 · LoRA 可训练参数量（第 15 章）
     ========================================================================== */
  registry.lora = function (el) {
    const out = el.querySelector('[data-role="info"]');
    const bars = el.querySelector('[data-role="bars"]');
    const sl = {};
    $$('input[data-p]', el).forEach((s) => {
      sl[s.getAttribute('data-p')] = s;
      const lab = el.querySelector('[data-v="' + s.getAttribute('data-p') + '"]');
      const upd = () => {
        if (!lab) return;
        const k = s.getAttribute('data-p');
        lab.textContent = (+s.value).toLocaleString('en-US');
      };
      s.addEventListener('input', () => { upd(); draw(); });
      upd();
    });
    const toggles = {};
    $$('[data-mod]', el).forEach((b) => {
      const k = b.getAttribute('data-mod');
      toggles[k] = b.getAttribute('aria-pressed') === 'true';
      b.addEventListener('click', () => {
        toggles[k] = !toggles[k];
        b.setAttribute('aria-pressed', String(toggles[k]));
        draw();
      });
    });

    function draw() {
      const d = +sl.d.value;
      const L = +sl.L.value;
      const r = +sl.r.value;
      const alpha = 2 * r;

      // 每个模块的 (输入维度, 输出维度)
      const MODS = {
        q: [d, d], k: [d, d], v: [d, d], o: [d, d],
        w1: [d, 4 * d], w3: [d, 4 * d], w2: [4 * d, d]
      };

      let baseTotal = 0, loraTotal = 0;
      const rows = [];
      Object.keys(MODS).forEach((k) => {
        const [di, dout] = MODS[k];
        const base = di * dout * L;
        baseTotal += base;
        if (toggles[k]) {
          const lora = (di * r + r * dout) * L;
          loraTotal += lora;
          rows.push([k, base, lora]);
        }
      });

      // 词嵌入
      const emb = (sl.V ? +sl.V.value : 50000) * d;
      baseTotal += emb;

      const pct = loraTotal / baseTotal * 100;
      const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : (n / 1e3).toFixed(0) + 'K';

      bars.innerHTML =
        '<div class="bar-row"><span class="bar-row__name" style="text-align:left;width:auto">原模型参数</span>' +
        '<span class="bar-row__track"><span class="bar-row__fill" style="width:100%;background:#c7d5f8"></span></span>' +
        '<span class="bar-row__val">' + fmt(baseTotal) + '</span></div>' +
        '<div class="bar-row bar-row--hot"><span class="bar-row__name" style="text-align:left;width:auto">LoRA 参数</span>' +
        '<span class="bar-row__track"><span class="bar-row__fill" style="width:' + Math.max(0.6, pct) + '%"></span></span>' +
        '<span class="bar-row__val">' + fmt(loraTotal) + '</span></div>';

      const detail = rows.map(([k, b, l]) => {
        const [di, dout] = MODS[k];
        return '  ' + k.padEnd(3) + ' (' + di + '×' + dout + ')  →  ' +
               fmt(l) + '  [2 × ' + di + '×' + r + ' × ' + L + ' 层]';
      }).join('\n');

      if (out) out.innerHTML =
        'd_model = ' + d + '   层数 = ' + L + '   秩 r = ' + r + '   α = ' + alpha + '\n' +
        '────────────────────────────────────\n' +
        '原模型总参数：<b>' + fmt(baseTotal) + '</b>\n' +
        '可训练参数：<b>' + fmt(loraTotal) + '</b>   占 ' + pct.toFixed(3) + '%\n' +
        '────────────────────────────────────\n' +
        '各模块明细：\n' + (detail || '  （未选择任何模块）') + '\n' +
        '────────────────────────────────────\n' +
        'AdamW 优化器状态（2 份 fp32）：' + (loraTotal * 8 / 1e9).toFixed(2) + ' GB\n' +
        '对比全参数微调的优化器状态：' + (baseTotal * 8 / 1e9).toFixed(1) + ' GB';

      if (toggles.q && toggles.v && !toggles.w1) {
        out.innerHTML += '\n\n💡 只调 q 和 v 是原始 LoRA 论文的推荐配置，性价比最高。';
      }
      if (toggles.w1) {
        out.innerHTML += '\n\n💡 加上 FFN 层（w1/w3/w2）能显著提升效果，因为 FFN 存了大部分事实知识。';
      }
    }
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 20 · 向量检索演示（第 17 章）
     ========================================================================== */
  registry.retrieval = function (el) {
    const cv = canvasOf(el);
    const out = el.querySelector('[data-role="info"]');
    const list = el.querySelector('[data-role="list"]');
    const kIn = el.querySelector('input[data-p="k"]');
    const kLab = el.querySelector('[data-v="k"]');

    const QUERY = { t: '如何申请年假？', v: [0.86, 0.62] };
    const DOCS = [
      { t: '员工入职满一年可享受 5 天年假，申请需提前 3 天在系统提交', v: [0.88, 0.60], tag: '年假' },
      { t: '当年未休完的年假可以累计到次年 3 月底前使用',           v: [0.79, 0.56], tag: '年假' },
      { t: '病假需要提供二级以上医院的证明，并抄送直属主管',         v: [0.70, 0.38], tag: '病假' },
      { t: '产假为 158 天，配偶陪产假为 15 天',                     v: [0.61, 0.33], tag: '产假' },
      { t: '加班时长可以折算成调休，有效期 6 个月',                  v: [0.52, 0.47], tag: '加班' },
      { t: '差旅报销需在费用发生后 30 天内提交发票原件',             v: [0.18, 0.86], tag: '报销' },
      { t: '公司食堂营业时间为 11:30 - 13:30',                      v: [0.13, 0.28], tag: '后勤' },
      { t: '离职需提前 30 天提交书面申请，并完成交接',               v: [0.29, 0.58], tag: '离职' }
    ];

    function cos(a, b) {
      const d = a[0] * b[0] + a[1] * b[1];
      const na = Math.hypot(a[0], a[1]), nb = Math.hypot(b[0], b[1]);
      return d / (na * nb);
    }

    function draw() {
      const k = +kIn.value;
      if (kLab) kLab.textContent = k;

      const ranked = DOCS.map((d, i) => ({ ...d, i, s: cos(QUERY.v, d.v) }))
                         .sort((a, b) => b.s - a.s);
      const top = new Set(ranked.slice(0, k).map((d) => d.i));

      const { ctx, w, h } = fit(cv);
      const p = makePlot(ctx, w, h, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, pad: 34 });
      p.grid(5, 5);

      // 所有文档
      DOCS.forEach((d, i) => {
        const on = top.has(i);
        const x = p.X(d.v[0]), y = p.Y(d.v[1]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, on ? 7 : 4.5, 0, Math.PI * 2);
        ctx.fillStyle = on ? '#3b6ef5' : 'rgba(130,136,150,.45)';
        ctx.fill();
        if (on) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.6; ctx.stroke(); }
        ctx.restore();
        p.label(d.tag, x, y - (on ? 15 : 12), on ? '#3b6ef5' : ink4(), 'center', 10.5);
      });

      // 查询向量
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.X(QUERY.v[0]), p.Y(QUERY.v[1]), 9, 0, Math.PI * 2);
      ctx.fillStyle = '#c2410c';
      ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();
      p.label('查询', p.X(QUERY.v[0]), p.Y(QUERY.v[1]) + 22, '#c2410c', 'center', 11);

      // 连线到 top-k
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(59,110,245,.5)';
      ctx.lineWidth = 1.4;
      ranked.slice(0, k).forEach((d) => {
        ctx.beginPath();
        ctx.moveTo(p.X(QUERY.v[0]), p.Y(QUERY.v[1]));
        ctx.lineTo(p.X(d.v[0]), p.Y(d.v[1]));
        ctx.stroke();
      });
      ctx.restore();

      list.innerHTML = ranked.map((d, r) =>
        '<div class="bar-row' + (r < k ? ' bar-row--hot' : '') + '" style="grid-template-columns:1.5rem 1fr 3.4rem">' +
          '<span class="bar-row__name" style="text-align:left">' + (r + 1) + '</span>' +
          '<span style="font-size:.74rem;color:' + (r < k ? 'var(--ink)' : 'var(--ink-4)') +
            ';white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + d.t + '</span>' +
          '<span class="bar-row__val">' + d.s.toFixed(3) + '</span>' +
        '</div>').join('');

      if (out) out.innerHTML =
        '查询：「' + QUERY.t + '」\n' +
        '向量维度：2（真实模型是 768~3072 维）\n' +
        '相似度：余弦相似度 cos(θ) = (a·b)/(|a||b|)\n' +
        '────────────────────────────────────\n' +
        'Top-' + k + ' 进入提示词：\n' +
        ranked.slice(0, k).map((d) => '  ✓ ' + d.tag).join('\n') +
        '\n被排除：' + ranked.slice(k).map((d) => d.tag).join('、') +
        '\n────────────────────────────────────\n' +
        '注意排名 3~5 的「病假 / 产假 / 加班」——它们和年假语义相近但不回答' +
        '\n这个问题。这就是为什么纯向量检索不够，还需要重排序。';
    }

    kIn.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 21 · ReAct 循环演示
     ========================================================================== */
  const REACT_SCEN = [
    {
      name: '单步查询',
      task: '北京今天天气怎么样？',
      lesson: '最简单的形态：一轮就结束。注意「思考」这一步——它让模型在动手前先判断「我需不需要工具」，而不是凭记忆硬答。',
      steps: [
        { t: 'thought', x: '用户问的是实时天气。训练数据里没有今天的数据，必须调用工具。' },
        { t: 'action', tool: 'get_weather', args: '{"city":"北京","date":"today"}' },
        { t: 'obs', x: '{"city":"北京","temp_c":18,"desc":"小雨","precip_prob":0.8,"wind":"东北风3级"}' },
        { t: 'thought', x: '拿到了完整数据，可以直接回答，不需要再调工具。' },
        { t: 'final', x: '北京今天 18℃，小雨，降水概率 80%，东北风 3 级。出门记得带伞。' }
      ]
    },
    {
      name: '多步判断',
      task: '我明天去上海出差，需要带伞吗？',
      lesson: '多步任务的关键：每一步的「观察」都会改变下一步的决策。先查天气、发现要下雨，才去查行程——顺序是模型自己定的，不是你写死的。',
      steps: [
        { t: 'thought', x: '要判断带不带伞，先得知道上海明天的天气。' },
        { t: 'action', tool: 'get_weather', args: '{"city":"上海","date":"tomorrow"}' },
        { t: 'obs', x: '{"city":"上海","date":"明天","precip_prob":0.75,"desc":"中雨","temp_c":22}' },
        { t: 'thought', x: '降水概率 75%，大概率下雨。但还得知道明天在户外待多久——先查一下行程。' },
        { t: 'action', tool: 'get_calendar', args: '{"date":"tomorrow","city":"上海"}' },
        { t: 'obs', x: '{"events":[{"time":"09:30","title":"客户会议 · 浦东"},{"time":"19:00","title":"返程航班"}]}' },
        { t: 'thought', x: '09:30 就要在浦东开会，全天在外。中雨 + 全天在外 = 必须带伞。' },
        { t: 'final', x: '需要带伞。上海明天中雨，降水概率 75%，22℃，你 09:30 在浦东有客户会议，全天在外。' }
      ]
    },
    {
      name: '工具报错',
      task: '帮我查一下深圳办公室的会议室预订情况',
      lesson: '工具失败是常态。正确做法是把错误原文当作「观察」回给模型，让它自己决定重试还是换路——千万不要 try/except 之后返回空值，那样模型只会编一个答案。',
      steps: [
        { t: 'thought', x: '需要调用会议室系统查询。' },
        { t: 'action', tool: 'book_room', args: '{"action":"list","city":"深圳"}' },
        { t: 'obs', x: 'ERROR 503 Service Unavailable: booking-api internal error' },
        { t: 'thought', x: '服务端 503，属于临时故障，可以重试一次。' },
        { t: 'action', tool: 'book_room', args: '{"action":"list","city":"深圳"}  ← 第 2 次' },
        { t: 'obs', x: 'ERROR 503 Service Unavailable: booking-api internal error' },
        { t: 'thought', x: '连续两次 503，不是偶发抖动。换备用接口试试。' },
        { t: 'action', tool: 'search_room_cache', args: '{"city":"深圳","max_age_hours":24}' },
        { t: 'obs', x: '{"source":"cache","as_of":"昨天18:00","rooms":[{"name":"A301","free":["10:00-12:00","15:00-18:00"]},{"name":"B105","free":["14:00-17:00"]}]}' },
        { t: 'thought', x: '拿到缓存，但它是昨天 18:00 的快照，可能已过时。回答时必须说明这一点。' },
        { t: 'final', x: '会议室系统当前不可用（连续两次 503）。以下是昨天 18:00 的缓存：A301 在 10:00-12:00、15:00-18:00 空闲；B105 在 14:00-17:00 空闲。数据可能已变化，建议稍后重新确认。' }
      ]
    }
  ];

  registry.agentloop = function (el) {
    const cv = canvasOf(el);
    const chips = $$('[data-scn]', el);
    const trace = el.querySelector('[data-role="trace"]');
    const info = el.querySelector('[data-role="info"]');
    const btnNext = el.querySelector('[data-act="next"]');
    const btnPlay = el.querySelector('[data-act="play"]');
    const btnReset = el.querySelector('[data-act="reset"]');

    const STAGE = { thought: 0, action: 1, obs: 2, final: 3 };
    const NODES = [
      { a: -Math.PI / 2, label: '思考', sub: 'Thought' },
      { a: Math.PI / 6, label: '行动', sub: 'Action' },
      { a: Math.PI * 5 / 6, label: '观察', sub: 'Observation' }
    ];
    const NR = 31;

    let si = 0, k = 0, timer = null;

    function cols() {
      return {
        brand: cssVar('--brand') || '#2f5fd0',
        brandSoft: cssVar('--brand-soft') || '#eef2fe',
        brandLine: cssVar('--brand-line') || '#c7d5f8',
        violet: cssVar('--violet') || '#6d28d9',
        violetSoft: cssVar('--violet-soft') || '#f2edfd',
        accent: cssVar('--accent') || '#b45309',
        accentSoft: cssVar('--accent-soft') || '#fdf3e6',
        ok: cssVar('--ok') || '#0f766e',
        okSoft: cssVar('--ok-soft') || '#e8f6f4',
        line: cssVar('--line') || '#e7e4dd',
        line2: cssVar('--line-2') || '#d8d4ca',
        surf: surface()
      };
    }

    function rrect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      if (btnPlay) btnPlay.textContent = '自动播放';
    }

    function roundOf(n) {
      const st = REACT_SCEN[si].steps;
      let r = 0;
      for (let i = 0; i <= n && i < st.length; i++) if (st[i].t === 'thought') r++;
      return r;
    }

    function arrow(ctx, x1, y1, x2, y2, color, dash) {
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
      const ux = dx / L, uy = dy / L;
      const sx = x1 + ux * (NR + 2), sy = y1 + uy * (NR + 2);
      const ex = x2 - ux * (NR + 9), ey = y2 - uy * (NR + 9);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.7;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - ux * 8 - uy * 4.2, ey - uy * 8 + ux * 4.2);
      ctx.lineTo(ex - ux * 8 + uy * 4.2, ey - uy * 8 - ux * 4.2);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }

    function draw() {
      const C = cols();
      const sc = REACT_SCEN[si];
      const last = sc.steps.length - 1;
      if (k > last) k = last;
      if (k < 0) k = 0;
      const stage = STAGE[sc.steps[k].t];
      const seen = {};
      sc.steps.slice(0, k + 1).forEach((s) => { seen[STAGE[s.t]] = true; });

      const { ctx, w, h } = fit(cv);
      const cx = w / 2, cy = 126, R = Math.min(84, w * 0.28);
      const pos = NODES.map((n) => ({ x: cx + R * Math.cos(n.a), y: cy + R * Math.sin(n.a) }));

      /* ---- 顶部信息 ---- */
      ctx.save();
      ctx.font = '700 11px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = C.brand;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('第 ' + roundOf(k) + ' 轮', 14, 18);
      ctx.textAlign = 'right';
      ctx.fillStyle = ink();
      ctx.fillText('当前：' + ['思考', '行动', '观察', '输出答案'][stage], w - 14, 18);
      ctx.restore();

      /* ---- 循环箭头 ---- */
      const ring = [
        [0, 1], [1, 2], [2, 0]
      ];
      ring.forEach(([i, j]) => {
        const active = (stage === i) || (stage === j && j !== 0);
        arrow(ctx, pos[i].x, pos[i].y, pos[j].x, pos[j].y,
              active ? C.brand : C.line2);
      });

      /* ---- 中心：LLM ---- */
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, 23, 0, Math.PI * 2);
      ctx.fillStyle = C.brandSoft;
      ctx.fill();
      ctx.strokeStyle = C.brandLine;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.font = '700 12px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = C.brand;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('LLM', cx, cy - 4);
      ctx.font = '8px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink();
      ctx.fillText('唯一决策者', cx, cy + 9);
      ctx.restore();

      /* ---- 三个节点 ---- */
      pos.forEach((p, i) => {
        const on = stage === i;
        const was = !!seen[i] && !on;
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, NR, 0, Math.PI * 2);
        ctx.fillStyle = on ? C.brand : (was ? C.brandSoft : C.surf);
        ctx.fill();
        ctx.strokeStyle = on ? C.brand : (was ? C.brandLine : C.line2);
        ctx.lineWidth = on ? 2 : 1.3;
        ctx.stroke();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '700 12.5px -apple-system,"PingFang SC",sans-serif';
        ctx.fillStyle = on ? '#fff' : (was ? C.brand : ink());
        ctx.fillText(NODES[i].label, p.x, p.y - 5);
        ctx.font = '8px -apple-system,sans-serif';
        ctx.fillStyle = on ? 'rgba(255,255,255,.8)' : ink4();
        ctx.fillText(NODES[i].sub, p.x, p.y + 9);
        ctx.restore();
      });

      /* ---- 答案出口 ---- */
      const bx = cx - 92, by = 216, bw = 184, bh = 46;
      const done = stage === 3;
      ctx.save();
      rrect(ctx, bx, by, bw, bh, 9);
      ctx.fillStyle = done ? C.okSoft : C.surf;
      ctx.fill();
      ctx.strokeStyle = done ? C.ok : C.line2;
      ctx.lineWidth = done ? 1.8 : 1.2;
      if (!done) ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '700 11.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = done ? C.ok : ink();
      ctx.fillText('输出答案 · 结束', cx, by + 17);
      ctx.font = '9px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink4();
      ctx.fillText('把结果返回给用户', cx, by + 33);
      ctx.restore();

      /* ---- 指向答案的虚线 ---- */
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = done ? C.ok : C.line2;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(cx, cy + R + NR + 6);
      ctx.lineTo(cx, by - 6);
      ctx.stroke();
      ctx.restore();

      /* ---- 右栏：轨迹 ---- */
      const lines = ['任务：' + sc.task, ''];
      let r = 0;
      for (let i = 0; i <= k; i++) {
        const s = sc.steps[i];
        if (s.t === 'thought') {
          r++;
          lines.push('── 第 ' + r + ' 轮 ──────────────');
          lines.push('思考 │ ' + s.x);
        } else if (s.t === 'action') {
          lines.push('行动 │ ' + s.tool + '(' + s.args + ')');
        } else if (s.t === 'obs') {
          lines.push('观察 │ ' + s.x);
        } else {
          lines.push('');
          lines.push('答案 │ ' + s.x);
        }
      }
      if (k < last) lines.push('', '… 还有 ' + (last - k) + ' 步未展开');
      if (trace) {
        trace.textContent = lines.join('\n');
        trace.scrollTop = trace.scrollHeight;
      }
      if (info) {
        info.textContent = (k >= last ? '▸ 这一局的看点：' : '▸ 提示：') + sc.lesson;
      }
      if (btnNext) btnNext.disabled = (k >= last);
    }

    function next() {
      const last = REACT_SCEN[si].steps.length - 1;
      if (k >= last) { stop(); return; }
      k++;
      draw();
      if (k >= last) stop();
    }

    function pick(i) {
      stop();
      si = i;
      k = 0;
      chips.forEach((c, ci) => c.classList.toggle('is-on', ci === i));
      draw();
    }

    if (btnNext) btnNext.addEventListener('click', next);
    if (btnReset) btnReset.addEventListener('click', () => { stop(); k = 0; draw(); });
    if (btnPlay) {
      btnPlay.addEventListener('click', () => {
        if (timer) { stop(); return; }
        if (k >= REACT_SCEN[si].steps.length - 1) k = 0;
        btnPlay.textContent = '暂停 ⏸';
        timer = setInterval(next, 950);
      });
    }
    chips.forEach((c, ci) => c.addEventListener('click', () => pick(ci)));

    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 22 · 精确率 / 召回率 / F1 权衡
     ========================================================================== */
  registry.metrics = function (el) {
    const cv = canvasOf(el);
    const tIn = el.querySelector('[data-p="t"]');
    const tLab = el.querySelector('[data-v="t"]');
    const barsEl = el.querySelector('[data-role="bars"]');
    const out = el.querySelector('[data-role="info"]');

    /* ---- 逆正态分布（Acklam 近似），用于生成确定性的样本 ---- */
    function invNorm(p) {
      const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
      const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                 6.680131188771972e+01, -1.328068155288572e+01];
      const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                 -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
      const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                 3.754408661907416e+00];
      const pl = 0.02425, ph = 1 - pl;
      let q, r;
      if (p < pl) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
      }
      if (p > ph) {
        q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
      }
      q = p - 0.5; r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    const N = 200;
    function sample(mu, sd, n) {
      const out2 = [];
      for (let i = 0; i < n; i++) out2.push(mu + sd * invNorm((i + 0.5) / n));
      return out2;
    }
    const NEG = sample(0.38, 0.12, N);   // 负类：正常请求
    const POS = sample(0.62, 0.12, N);   // 正类：恶意请求
    const SD = 0.12;

    function pdf(x, mu) { return Math.exp(-((x - mu) * (x - mu)) / (2 * SD * SD)); }

    function draw() {
      const t = tIn ? +tIn.value : 0.5;
      if (tLab) tLab.textContent = t.toFixed(2);

      let TP = 0, FN = 0, FP = 0, TN = 0;
      POS.forEach((s) => (s >= t ? TP++ : FN++));
      NEG.forEach((s) => (s >= t ? FP++ : TN++));

      const P  = (TP + FP) ? TP / (TP + FP) : 0;
      const R  = (TP + FN) ? TP / (TP + FN) : 0;
      const F1 = (P + R) ? 2 * P * R / (P + R) : 0;
      const ACC = (TP + TN) / (N * 2);

      const brand = cssVar('--brand') || '#2f5fd0';
      const violet = cssVar('--violet') || '#6d28d9';
      const accent = cssVar('--accent') || '#b45309';
      const okc = cssVar('--ok') || '#0f766e';
      const line2 = cssVar('--line-2') || '#d8d4ca';
      const surf2 = cssVar('--surface-2') || '#faf9f6';

      const { ctx, w, h } = fit(cv);

      /* ---------------- 上半：两个分布 + 阈值 ---------------- */
      const px = 34, pw = w - px - 18;
      const X = (s) => px + Math.max(0, Math.min(1, s)) * pw;
      const baseY = 156, amp = 104;
      const Y = (d) => baseY - d * amp;

      // 坐标轴
      ctx.save();
      ctx.strokeStyle = line2;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, baseY); ctx.lineTo(px + pw, baseY); ctx.stroke();
      ctx.restore();

      // 重叠区提示
      ctx.save();
      ctx.fillStyle = 'rgba(130,136,150,.10)';
      ctx.fillRect(X(0.42), 34, X(0.58) - X(0.42), baseY - 34);
      ctx.restore();

      // 阴影：阈值右侧的样本被判定为正类
      function shade(samples, mu, color) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(X(t), baseY);
        const steps = 90;
        for (let i = 0; i <= steps; i++) {
          const s = t + (1 - t) * i / steps;
          ctx.lineTo(X(s), Y(pdf(s, mu)));
        }
        ctx.lineTo(X(1), baseY);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();
      }
      shade(NEG, 0.38, 'rgba(47,95,208,.20)');
      shade(POS, 0.62, 'rgba(180,83,9,.22)');

      // 曲线
      function curve(mu, color, width) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        for (let i = 0; i <= 200; i++) {
          const s = i / 200;
          const x = X(s), y = Y(pdf(s, mu));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }
      curve(0.38, brand, 2.2);
      curve(0.62, accent, 2.2);

      // 样本点（确定性分位点，越密表示密度越高）
      ctx.save();
      NEG.forEach((s) => {
        ctx.beginPath();
        ctx.arc(X(s), baseY + 9, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = s >= t ? brand : 'rgba(130,136,150,.30)';
        ctx.fill();
      });
      POS.forEach((s) => {
        ctx.beginPath();
        ctx.arc(X(s), baseY + 17, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = s >= t ? accent : 'rgba(130,136,150,.30)';
        ctx.fill();
      });
      ctx.restore();

      // 阈值线
      ctx.save();
      ctx.strokeStyle = okc;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(X(t), 28);
      ctx.lineTo(X(t), baseY + 22);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '700 10.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = okc;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('阈值 ' + t.toFixed(2), Math.max(px + 26, Math.min(px + pw - 26, X(t))), 20);
      ctx.restore();

      // 分布标签
      ctx.save();
      ctx.font = '10px -apple-system,"PingFang SC",sans-serif';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.fillStyle = brand;
      ctx.fillText('负类（正常）', X(0.12), 44);
      ctx.fillStyle = accent;
      ctx.textAlign = 'right';
      ctx.fillText('正类（恶意）', X(0.94), 44);
      ctx.fillStyle = ink4();
      ctx.textAlign = 'center';
      ctx.font = '9px -apple-system,"PingFang SC",sans-serif';
      ctx.fillText('重叠区：无论阈值定在哪里，这里都必然出错', px + pw / 2, 66);
      ctx.restore();

      /* ---------------- 下半：混淆矩阵 ---------------- */
      const cw = Math.min(112, (w - 110) / 2), ch = 42;
      const gx = (w - cw * 2) / 2 + 26;
      const gy = 216;

      const cells = [
        { r: 0, c: 0, n: TP, label: 'TP', color: okc, bg: cssVar('--ok-soft') || '#e8f6f4' },
        { r: 0, c: 1, n: FN, label: 'FN', color: accent, bg: cssVar('--accent-soft') || '#fdf3e6' },
        { r: 1, c: 0, n: FP, label: 'FP', color: accent, bg: cssVar('--accent-soft') || '#fdf3e6' },
        { r: 1, c: 1, n: TN, label: 'TN', color: okc, bg: surf2 }
      ];

      ctx.save();
      ctx.font = '9.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('预测为正', gx + cw * 0.5, gy - 12);
      ctx.fillText('预测为负', gx + cw * 1.5, gy - 12);
      ctx.textAlign = 'right';
      ctx.fillText('实际为正', gx - 8, gy + ch * 0.5);
      ctx.fillText('实际为负', gx - 8, gy + ch * 1.5);
      ctx.restore();

      cells.forEach((c) => {
        const x = gx + c.c * cw, y = gy + c.r * ch;
        ctx.save();
        ctx.fillStyle = c.bg;
        ctx.fillRect(x, y, cw, ch);
        ctx.strokeStyle = line2;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        ctx.font = '700 15px -apple-system,"PingFang SC",sans-serif';
        ctx.fillStyle = c.color;
        ctx.fillText(String(c.n), x + 12, y + ch / 2 - 1);
        ctx.font = '9.5px -apple-system,sans-serif';
        ctx.textAlign = 'right';
        ctx.fillStyle = ink4();
        ctx.fillText(c.label, x + cw - 10, y + ch / 2 - 1);
        ctx.restore();
      });

      /* ---------------- 右栏：指标条 ---------------- */
      if (barsEl) {
        const rows = [
          ['精确率', P, brand],
          ['召回率', R, violet],
          ['F1', F1, okc]
        ];
        barsEl.innerHTML = rows.map((r2) =>
          '<div class="bar-row" style="grid-template-columns:3.4rem 1fr 2.6rem">' +
            '<span class="bar-row__name" style="text-align:left">' + r2[0] + '</span>' +
            '<span class="bar-row__track"><span class="bar-row__fill" style="width:' +
              Math.max(1, r2[1] * 100).toFixed(1) + '%;background:' + r2[2] + '"></span></span>' +
            '<span class="bar-row__val">' + r2[1].toFixed(3) + '</span>' +
          '</div>').join('');
      }

      if (out) {
        const prec = (TP + FP) ? (TP / (TP + FP)).toFixed(3) : '—';
        const rec = (TP + FN) ? (TP / (TP + FN)).toFixed(3) : '—';
        out.innerHTML =
          '阈值 = ' + t.toFixed(2) + '\n' +
          '─────────────────────────────\n' +
          'TP ' + String(TP).padStart(3) + '   FN ' + String(FN).padStart(3) +
          '    ← 实际为正的 ' + N + ' 个\n' +
          'FP ' + String(FP).padStart(3) + '   TN ' + String(TN).padStart(3) +
          '    ← 实际为负的 ' + N + ' 个\n' +
          '─────────────────────────────\n' +
          '精确率 = TP/(TP+FP) = ' + TP + '/' + (TP + FP) + ' = ' + prec + '\n' +
          '召回率 = TP/(TP+FN) = ' + TP + '/' + (TP + FN) + ' = ' + rec + '\n' +
          'F1     = 2PR/(P+R)          = ' + F1.toFixed(3) + '\n' +
          '准确率 = (TP+TN)/总数       = ' + ACC.toFixed(3) +
          '\n\n' +
          '↑ 准确率看起来最高，但两类样本数量相等时它才有意义。\n' +
          '  如果正类只占 1%，准确率会虚高到 0.99，而召回率可能接近 0。';
      }
    }

    if (tIn) tIn.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     组件 23 · MoE 路由与专家坍缩
     ========================================================================== */
  registry.moe = function (el) {
    const cv = canvasOf(el);
    const kIn = el.querySelector('[data-p="k"]');
    const kLab = el.querySelector('[data-v="k"]');
    const sIn = el.querySelector('[data-p="s"]');
    const sLab = el.querySelector('[data-v="s"]');
    const out = el.querySelector('[data-role="info"]');

    const TOK = ['苹果', '股价', '量子', '纠缠', '李白', '的', '感冒', '药'];
    const E = 8;
    // 路由偏置：模拟训练中自然出现的偏好（越靠前的专家越容易被选中）
    const BIAS = [2.0, 1.2, 0.5, -0.2, -0.6, -0.9, -1.1, -1.3];

    function base(i, e) {
      const s = Math.sin(i * 12.9898 + e * 78.233) * 43758.5453;
      return s - Math.floor(s);
    }

    function draw() {
      const K = kIn ? +kIn.value : 2;
      const skew = sIn ? +sIn.value : 0;
      if (kLab) kLab.textContent = K;
      if (sLab) sLab.textContent = skew.toFixed(2);

      /* ---- 路由：每个 token 得到 8 个分数，取 Top-K ---- */
      const routes = TOK.map((_, i) =>
        Array.from({ length: E }, (_, e) => base(i, e) * 2.2 + skew * BIAS[e])
      );
      const picks = routes.map((r) => {
        const order = r.map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]);
        return order.slice(0, K).map(([v, e]) => [e, v]);
      });
      // 权重归一化（softmax over the selected K）
      const wts = picks.map((p) => {
        const mx = Math.max(...p.map((x) => x[1]));
        const es = p.map((x) => Math.exp(x[1] - mx));
        const sum = es.reduce((a, b) => a + b, 0);
        return p.map((x, i2) => [x[0], es[i2] / sum]);
      });

      const load = new Array(E).fill(0);
      wts.forEach((p) => p.forEach(([e]) => { load[e]++; }));
      const maxLoad = Math.max(1, ...load);
      const idle = load.filter((n) => n === 0).length;

      const { ctx, w, h } = fit(cv);
      const line2 = cssVar('--line-2') || '#d8d4ca';
      const accent = cssVar('--accent') || '#b45309';
      const margin = 10, colW = (w - margin * 2) / E;
      const chipY = 40, chipH = 30, boxY = 250, boxH = 64;
      const cx = (i) => margin + colW * i + colW / 2;

      /* ---- 标题 ---- */
      ctx.save();
      ctx.font = '9.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink4();
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('8 个 token', 10, 20);
      ctx.textAlign = 'right';
      ctx.fillText('每个 token 只走 Top-' + K + ' 个专家', w - 10, 20);
      ctx.restore();

      /* ---- 路由连线（先画，压在方块下面）---- */
      wts.forEach((p, i) => {
        p.forEach(([e, wv]) => {
          ctx.save();
          ctx.strokeStyle = PAL[e % PAL.length];
          ctx.globalAlpha = 0.18 + 0.62 * wv;
          ctx.lineWidth = 0.8 + 2.2 * wv;
          ctx.beginPath();
          ctx.moveTo(cx(i), chipY + chipH);
          ctx.lineTo(cx(e), boxY);
          ctx.stroke();
          ctx.restore();
        });
      });

      /* ---- 路由器标识 ---- */
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = line2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(10, 168);
      ctx.lineTo(w - 10, 168);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '700 9.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = cssVar('--violet') || '#6d28d9';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const mid = w / 2;
      ctx.fillStyle = cssVar('--surface') || '#fff';
      ctx.fillRect(mid - 92, 158, 184, 20);
      ctx.fillStyle = cssVar('--violet') || '#6d28d9';
      ctx.fillText('路由器：为每个 token 打分并选 Top-' + K, mid, 168);
      ctx.restore();

      /* ---- token 方块 ---- */
      TOK.forEach((t, i) => {
        const x = margin + colW * i + 2, bw = colW - 4;
        ctx.save();
        ctx.fillStyle = cssVar('--surface-2') || '#faf9f6';
        ctx.strokeStyle = line2;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(x, chipY, bw, chipH);
        ctx.fill();
        ctx.stroke();
        ctx.font = '9.5px -apple-system,"PingFang SC",sans-serif';
        ctx.fillStyle = ink2();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(t, x + bw / 2, chipY + chipH / 2);
        ctx.restore();
      });

      /* ---- 专家方块 ---- */
      for (let e = 0; e < E; e++) {
        const x = margin + colW * e + 2, bw = colW - 4;
        const c = PAL[e % PAL.length];
        const isIdle = load[e] === 0;
        ctx.save();
        // 底
        ctx.fillStyle = cssVar('--surface') || '#fff';
        ctx.strokeStyle = isIdle ? accent : line2;
        ctx.lineWidth = isIdle ? 1.6 : 1;
        if (isIdle) ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.rect(x, boxY, bw, boxH);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        // 负载条
        const fh = (load[e] / maxLoad) * (boxH - 2);
        if (fh > 0) {
          ctx.fillStyle = c;
          ctx.globalAlpha = 0.30;
          ctx.fillRect(x + 1, boxY + boxH - 1 - fh, bw - 2, fh);
          ctx.globalAlpha = 1;
        }
        // 文字
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '700 9px -apple-system,sans-serif';
        ctx.fillStyle = isIdle ? accent : c;
        ctx.fillText('E' + (e + 1), x + bw / 2, boxY + 12);
        ctx.font = '700 13px -apple-system,sans-serif';
        ctx.fillStyle = isIdle ? accent : ink();
        ctx.fillText(String(load[e]), x + bw / 2, boxY + boxH / 2 + 8);
        if (isIdle) {
          ctx.font = '8px -apple-system,"PingFang SC",sans-serif';
          ctx.fillStyle = accent;
          ctx.fillText('闲置', x + bw / 2, boxY + boxH - 9);
        }
        ctx.restore();
      }

      /* ---- 底部说明（分两行，单行会超出画布宽度被裁掉）---- */
      ctx.save();
      ctx.font = '9.5px -apple-system,"PingFang SC",sans-serif';
      ctx.fillStyle = ink4();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('激活参数 = ' + K + '/8 = ' + (K / E * 100).toFixed(0) +
                   '%　·　总参数 = 8 份', w / 2, boxY + boxH + 16);
      ctx.fillText('方块填充高度 = 该专家接收的 token 数', w / 2, boxY + boxH + 30);
      ctx.restore();

      /* ---- 读数 ---- */
      if (out) {
        const grid = [];
        for (let r = 0; r < 2; r++) {
          let s2 = '';
          for (let c = 0; c < 4; c++) {
            const e = r * 4 + c;
            s2 += ('E' + (e + 1)).padEnd(3) + String(load[e]).padStart(2) + '   ';
          }
          grid.push(s2.trimEnd());
        }
        out.innerHTML =
          'Top-K = ' + K + ' / ' + E + ' 个专家\n' +
          '激活参数比例 = ' + K + '/' + E + ' = ' + (K / E * 100).toFixed(0) +
          '%　（推理成本按这个算，不是按总参数）\n' +
          '─────────────────────────────\n' +
          '专家负载分布：\n' +
          '  ' + grid[0] + '\n' +
          '  ' + grid[1] + '\n' +
          '─────────────────────────────\n' +
          '最大负载 ' + maxLoad + ' · 最小负载 ' + Math.min(...load) +
          ' · 闲置专家 ' + idle + ' 个\n' +
          (idle === 0
            ? '✓ 负载均衡：所有专家都在工作，参数没有被浪费。'
            : '⚠ 专家坍缩：有 ' + idle + ' 个专家接收不到任何 token，\n' +
              '  它们的参数永远得不到梯度更新——等于白占显存。\n' +
              '  这就是 load_balance_loss 存在的理由。');
      }
    }

    if (kIn) kIn.addEventListener('input', draw);
    if (sIn) sIn.addEventListener('input', draw);
    draw();
    return { draw };
  };

  /* ==========================================================================
     启动与注册
     ========================================================================== */
  function init() {
    $$('[data-widget]').forEach((el) => {
      const name = el.getAttribute('data-widget');
      const fn = registry[name];
      if (!fn) return;
      try {
        const inst = fn(el);
        if (inst && inst.draw) {
          instances.push(inst);
          // 由框架保证首帧：组件不必自己记得在末尾调 draw()，
          // 否则漏掉的那个在用户拖滑块之前就是一片空白。
          // draw() 本身幂等（redrawAll 也会反复调用），重复一次没有副作用。
          inst.draw();
        }
      } catch (e) {
        if (window.console) console.warn('[widget:' + name + ']', e);
      }
    });
  }

  const redrawAll = debounce(function () {
    instances.forEach((i) => { try { i.draw(); } catch (e) {} });
  }, 140);

  window.DLWidgets = { init: init, redrawAll: redrawAll, registry: registry, ACT: ACT };
})();
