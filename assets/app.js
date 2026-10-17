/* ==========================================================================
   《深度学习之路》· 运行引擎
   负责：全局清单 / 顶栏 / 侧边目录 / 页内目录 / 阅读进度 / 主题 /
         数学公式渲染 / 代码高亮 / 复制按钮 / 上下章导航
   ========================================================================== */
(function () {
  'use strict';

  /* ----------------------------------------------------------------------
     0. 全书清单
     ---------------------------------------------------------------------- */
  const BOOK = {
    title: '深度学习之路',
    short: '深度学习之路',
    chapters: [
      { id: '01', file: '01-from-rules-to-learning.html', title: '从规则到学习', part: '第一部 · 直觉篇', desc: '程序写规则，模型学规则' },
      { id: '02', file: '02-neuron-and-activation.html', title: '神经元与激活函数', part: '第一部 · 直觉篇', desc: '一个会投票的小单元' },
      { id: '03', file: '03-forward-propagation.html', title: '多层网络与前向传播', part: '第一部 · 直觉篇', desc: '堆叠出来的表达能力' },

      { id: '04', file: '04-gradient-descent.html', title: '梯度下降与优化器', part: '第二部 · 引擎篇', desc: '模型到底怎么"变聪明"' },
      { id: '05', file: '05-backpropagation.html', title: '反向传播', part: '第二部 · 引擎篇', desc: '深度学习的发动机' },
      { id: '06', file: '06-training-craft.html', title: '训练的艺术', part: '第二部 · 引擎篇', desc: '初始化、归一化、正则化与调试' },

      { id: '07', file: '07-cnn.html', title: '卷积神经网络', part: '第三部 · 结构篇', desc: '让网络看懂图像' },
      { id: '08', file: '08-sequence-models.html', title: '序列模型与记忆', part: '第三部 · 结构篇', desc: 'RNN、LSTM 与梯度消失' },

      { id: '09', file: '09-attention.html', title: '注意力机制', part: '第四部 · 注意力革命', desc: 'Query、Key、Value 的魔法' },
      { id: '10', file: '10-transformer.html', title: 'Transformer 解剖', part: '第四部 · 注意力革命', desc: '一台完整的序列处理器' },
      { id: '11', file: '11-language-models.html', title: '语言模型与 GPT', part: '第四部 · 注意力革命', desc: '预测下一个词，就够了' },

      { id: '12', file: '12-pretraining.html', title: '大模型预训练', part: '第五部 · 大模型工程', desc: '数据、并行、精度与成本' },
      { id: '13', file: '13-alignment.html', title: '对齐：SFT / RLHF / DPO', part: '第五部 · 大模型工程', desc: '从会说话到会帮忙' },
      { id: '14', file: '14-inference.html', title: '推理与部署', part: '第五部 · 大模型工程', desc: 'KV Cache、量化与采样' },
      { id: '15', file: '15-fine-tuning.html', title: '微调与 LoRA', part: '第五部 · 大模型工程', desc: '用一张卡改造一个模型' },

      { id: '16', file: '16-prompting.html', title: '提示工程与上下文学习', part: '第六部 · 应用篇', desc: '把模型当同事，而不是搜索框' },
      { id: '17', file: '17-rag.html', title: '检索增强生成 RAG', part: '第六部 · 应用篇', desc: '给模型接一个外置记忆' },
      { id: '18', file: '18-agents.html', title: '智能体与工具调用', part: '第六部 · 应用篇', desc: '让模型自己动手做事' },
      { id: '19', file: '19-evaluation.html', title: '评测、幻觉与安全', part: '第六部 · 应用篇', desc: '怎么知道它到底行不行' },

      { id: '20', file: '20-mini-gpt.html', title: '从零实现 mini-GPT', part: '第七部 · 实践与前沿', desc: '三百行代码走完全程' },
      { id: '21', file: '21-frontier.html', title: '前沿与展望', part: '第七部 · 实践与前沿', desc: 'MoE、多模态、长上下文与推理' }
    ]
  };
  window.BOOK = BOOK;

  const ROOT = /\/chapters\//.test(location.pathname) ? '../' : '';
  window.__ROOT__ = ROOT;

  const $  = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  /* ----------------------------------------------------------------------
     1. 主题与字号
     ---------------------------------------------------------------------- */
  const html = document.documentElement;
  const THEME_KEY = 'dl-textbook-theme';
  const FONT_KEY  = 'dl-textbook-font';
  const FONT_ORDER = ['s', 'm', 'l'];
  const FONT_LABEL = { s: '小', m: '中', l: '大' };

  function applyTheme(t) {
    html.setAttribute('data-theme', t);
    try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
    const btn = $('#btn-theme');
    if (btn) {
      btn.textContent = t === 'dark' ? '☀' : '☾';
      btn.title = t === 'dark' ? '切换到亮色' : '切换到暗色';
    }
    window.dispatchEvent(new CustomEvent('dl:theme', { detail: t }));
  }
  function applyFont(f) {
    html.setAttribute('data-font', f);
    try { localStorage.setItem(FONT_KEY, f); } catch (e) {}
    const btn = $('#btn-font');
    if (btn) btn.textContent = 'Aa ' + FONT_LABEL[f];
  }

  (function initPrefs() {
    let t = 'light', f = 'm';
    try {
      t = localStorage.getItem(THEME_KEY) || t;
      f = localStorage.getItem(FONT_KEY) || f;
    } catch (e) {}
    if (FONT_ORDER.indexOf(f) < 0) f = 'm';
    html.setAttribute('data-theme', t);
    html.setAttribute('data-font', f);
    window.__font = f;
  })();

  /* ----------------------------------------------------------------------
     2. 顶栏 / 侧边栏 / 页内目录
     ---------------------------------------------------------------------- */
  const currentId = document.body.getAttribute('data-chapter');
  const isHome = document.body.hasAttribute('data-home');

  function buildTopbar() {
    if ($('.topbar')) return;
    const bar = document.createElement('header');
    bar.className = 'topbar';
    bar.innerHTML =
      '<button class="tbtn tbtn--menu" id="btn-menu" aria-label="目录">☰</button>' +
      '<a class="topbar__brand" href="' + ROOT + 'index.html">' +
        '<span class="topbar__mark">DL</span><span>' + BOOK.title + '</span>' +
      '</a>' +
      '<div class="topbar__spacer"></div>' +
      '<div class="topbar__tools">' +
        '<button class="tbtn" id="btn-font" title="调整正文字号">Aa 中</button>' +
        '<button class="tbtn" id="btn-theme" title="切换主题">☾</button>' +
      '</div>';
    document.body.insertBefore(bar, document.body.firstChild);

    $('#btn-menu').addEventListener('click', () => document.body.classList.toggle('nav-open'));
    $('#btn-font').addEventListener('click', () => {
      const cur = html.getAttribute('data-font') || 'm';
      const next = FONT_ORDER[(FONT_ORDER.indexOf(cur) + 1) % FONT_ORDER.length];
      applyFont(next);
      window.dispatchEvent(new CustomEvent('dl:resize'));
    });
    $('#btn-theme').addEventListener('click', () => {
      applyTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    });
    applyTheme(html.getAttribute('data-theme') || 'light');
    applyFont(html.getAttribute('data-font') || 'm');
  }

  function buildSidebar() {
    if (isHome || $('.sidebar')) return;
    const parts = [];
    BOOK.chapters.forEach((c) => {
      let p = parts.find((x) => x.name === c.part);
      if (!p) { p = { name: c.part, items: [] }; parts.push(p); }
      p.items.push(c);
    });

    const aside = document.createElement('aside');
    aside.className = 'sidebar';
    aside.id = 'sidebar';
    let h = '<p class="sidebar__label">目录</p>';
    parts.forEach((p) => {
      h += '<div class="part"><p class="part__name">' + p.name + '</p><ul class="toc-list">';
      p.items.forEach((c) => {
        const cur = c.id === currentId ? ' is-current' : '';
        h += '<li><a class="' + cur.trim() + '" href="' + ROOT + 'chapters/' + c.file + '">' +
             '<span class="n">' + c.id + '</span><span>' + c.title + '</span></a></li>';
      });
      h += '</ul></div>';
    });
    h += '<div class="pagemap" id="pagemap" hidden><p class="sidebar__label">本页</p><ul class="pagemap__list"></ul></div>';
    aside.innerHTML = h;

    const layout = $('.layout');
    if (layout) layout.insertBefore(aside, layout.firstChild);

    aside.addEventListener('click', (e) => {
      if (e.target.closest('a')) document.body.classList.remove('nav-open');
    });
  }

  function buildPageMap() {
    const article = $('.article');
    const map = $('#pagemap');
    if (!article || !map) return;
    const heads = $$('h2, h3', article);
    if (heads.length < 3) return;

    const ul = $('.pagemap__list', map);
    heads.forEach((hd, i) => {
      if (!hd.id) {
        hd.id = 'sec-' + (i + 1) + '-' + hd.textContent.trim().replace(/[\s\W]+/g, '-').slice(0, 24);
      }
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + hd.id;
      a.textContent = hd.textContent.replace(/^#+\s*/, '');
      if (hd.tagName === 'H3') a.className = 'lvl-3';
      li.appendChild(a);
      ul.appendChild(li);
    });
    map.hidden = false;

    // 目录高亮依赖 IntersectionObserver。没有它就只是少了滚动高亮，
    // 不该让整个启动流程挂掉（否则后面的公式渲染也会被连累）。
    if (typeof IntersectionObserver !== 'function') return;
    const links = $$('a', ul);
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        links.forEach((l) => l.classList.toggle('is-active', l.getAttribute('href') === '#' + en.target.id));
      });
    }, { rootMargin: '-70px 0px -72% 0px', threshold: 0 });
    heads.forEach((hd) => obs.observe(hd));
  }

  function buildChapterNav() {
    if (isHome) return;
    const idx = BOOK.chapters.findIndex((c) => c.id === currentId);
    if (idx < 0) return;
    const prev = BOOK.chapters[idx - 1];
    const next = BOOK.chapters[idx + 1];
    const nav = document.createElement('nav');
    nav.className = 'chapter-nav';
    const link = (c, dir) => c
      ? '<a class="chapter-nav__' + dir + '" href="' + c.file + '">' +
        '<div class="chapter-nav__dir">' + (dir === 'prev' ? '← 上一章' : '下一章 →') + '</div>' +
        '<div class="chapter-nav__title">' + c.title + '</div></a>'
      : '<span class="chapter-nav__empty"></span>';
    nav.innerHTML = link(prev, 'prev') + link(next, 'next');
    const article = $('.article');
    if (article) article.parentNode.insertBefore(nav, article.nextSibling);

    const foot = document.createElement('footer');
    foot.className = 'footer';
    foot.innerHTML = '<span>《' + BOOK.title + '》· 写给程序员的深度学习与大模型</span>' +
                     '<span>第 ' + currentId + ' / ' + BOOK.chapters.length + ' 章</span>';
    nav.parentNode.insertBefore(foot, nav.nextSibling);
  }

  /* ----------------------------------------------------------------------
     3. 阅读进度
     ---------------------------------------------------------------------- */
  function initProgress() {
    const bar = $('#progress');
    if (!bar) return;
    const update = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      const p = h > 0 ? Math.min(1, window.scrollY / h) : 0;
      bar.style.width = (p * 100).toFixed(2) + '%';
    };
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  /* ----------------------------------------------------------------------
     4. 代码块：注入工具条 + 语法高亮 + 复制
     ---------------------------------------------------------------------- */
  const PY_KEY = new Set(('False None True and as assert async await break class continue def del elif else ' +
    'except finally for from global if import in is lambda nonlocal not or pass raise return try while with ' +
    'yield match case').split(' '));
  const PY_BUILTIN = new Set(('abs all any bool bytes callable chr dict dir divmod enumerate eval filter float ' +
    'format frozenset getattr globals hasattr hash help hex id input int isinstance issubclass iter len list ' +
    'locals map max min next object oct open ord pow print property range repr reversed round set setattr ' +
    'slice sorted staticmethod str sum super tuple type vars zip self __init__ __name__ __main__ ' +
    'Exception ValueError TypeError RuntimeError KeyError IndexError super').split(' '));

  // Python 高亮：逐词判断关键字 / 内建 / 函数调用
  function hlPython(src) {
    let out = '', i = 0;
    const n = src.length;
    const emit = (cls, t) => { out += cls ? '<span class="tok-' + cls + '">' + esc(t) + '</span>' : esc(t); };
    while (i < n) {
      const rest = src.slice(i);
      let m;
      if ((m = /^("""[\s\S]*?"""|'''[\s\S]*?''')/.exec(rest))) { emit('str', m[0]); i += m[0].length; continue; }
      if ((m = /^(?:[rbfu]{0,2})("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')/.exec(rest))) { emit('str', m[0]); i += m[0].length; continue; }
      if ((m = /^#[^\n]*/.exec(rest))) { emit('com', m[0]); i += m[0].length; continue; }
      if ((m = /^@[A-Za-z_][\w.]*/.exec(rest))) { emit('dec', m[0]); i += m[0].length; continue; }
      if ((m = /^(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(rest))) { emit('num', m[0]); i += m[0].length; continue; }
      if ((m = /^[A-Za-z_]\w*/.exec(rest))) {
        const w = m[0];
        const after = rest.slice(w.length);
        if (PY_KEY.has(w)) emit('key', w);
        else if (PY_BUILTIN.has(w)) emit('bi', w);
        else if (/^\s*\(/.test(after)) emit('fn', w);
        else emit(null, w);
        i += w.length; continue;
      }
      if ((m = /^[(){}\[\],:;.+\-*/%=<>!&|^~]+/.exec(rest))) { emit('pun', m[0]); i += m[0].length; continue; }
      emit(null, src[i]); i++;
    }
    return out;
  }

  function hlBash(src) {
    let out = '', i = 0;
    const n = src.length;
    const emit = (cls, t) => { out += cls ? '<span class="tok-' + cls + '">' + esc(t) + '</span>' : esc(t); };
    while (i < n) {
      const rest = src.slice(i);
      let m;
      if ((m = /^#[^\n]*/.exec(rest))) { emit('com', m[0]); i += m[0].length; continue; }
      if ((m = /^("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/.exec(rest))) { emit('str', m[0]); i += m[0].length; continue; }
      if ((m = /^\s*[A-Za-z_][\w.\-/]*/.exec(rest)) && /^\s*$/.test(src.slice(Math.max(0, i - 60), i).split('\n').pop())) {
        emit('fn', m[0]); i += m[0].length; continue;
      }
      if ((m = /^--?[A-Za-z][\w-]*/.exec(rest))) { emit('num', m[0]); i += m[0].length; continue; }
      if ((m = /^\d+(?:\.\d+)?/.exec(rest))) { emit('num', m[0]); i += m[0].length; continue; }
      emit(null, src[i]); i++;
    }
    return out;
  }

  const HIGHLIGHTERS = { python: hlPython, py: hlPython, bash: hlBash, sh: hlBash, shell: hlBash };

  function buildCodeBlocks() {
    $$('.codeblock').forEach((blk) => {
      const src = blk.querySelector('script[type="text/dlcode"]');
      let code = '';
      if (src) {
        code = src.textContent.replace(/^\n/, '').replace(/\s+$/, '');
        src.remove();
      } else {
        const pre = blk.querySelector('pre code');
        if (!pre) return;
        code = pre.textContent.replace(/^\n/, '').replace(/\s+$/, '');
      }
      const lang = (blk.getAttribute('data-lang') || 'text').toLowerCase();
      const title = blk.getAttribute('data-title') || '';
      const hl = HIGHLIGHTERS[lang];
      const body = hl ? hl(code) : esc(code);

      blk.innerHTML =
        '<div class="codeblock__bar">' +
          '<span class="codeblock__lang">' + esc(lang) + '</span>' +
          (title ? '<span class="codeblock__title">' + esc(title) + '</span>' : '') +
          '<span class="codeblock__spacer"></span>' +
          '<button class="codeblock__copy" type="button">复制</button>' +
        '</div>' +
        '<pre><code>' + body + '</code></pre>';

      const btn = blk.querySelector('.codeblock__copy');
      btn.addEventListener('click', () => {
        const done = () => {
          btn.textContent = '已复制';
          btn.classList.add('done');
          setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('done'); }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(code).then(done, () => fallback());
        } else fallback();
        function fallback() {
          const ta = document.createElement('textarea');
          ta.value = code;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); done(); } catch (e) {}
          ta.remove();
        }
      });
    });
  }

  /* ----------------------------------------------------------------------
     5. 数学公式（KaTeX）

     加载顺序：本地 assets/vendor/katex → CDN → 纯文本降级。
     本地优先，是因为这本教材要能离线打开；CDN 兜底，是为了本地资源被误删时
     仍然可用；最后的降级路径把 $...$ 包成等宽样式，至少不会满屏乱码。
     ---------------------------------------------------------------------- */
  const KATEX_VER = '0.16.9';
  const KATEX_CDN = 'https://cdn.jsdelivr.net/npm/katex@' + KATEX_VER + '/dist/';
  const SRI = {
    css: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
    js: 'sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8',
    ar: 'sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05'
  };

  function addCss(href, onFail, sri) {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    if (sri) { l.integrity = sri; l.crossOrigin = 'anonymous'; }
    if (onFail) l.onerror = onFail;
    document.head.appendChild(l);
    return l;
  }

  function addScript(src, ok, fail, sri) {
    const s = document.createElement('script');
    s.src = src;
    if (sri) { s.integrity = sri; s.crossOrigin = 'anonymous'; }
    s.onload = ok || null;
    s.onerror = fail || null;
    document.head.appendChild(s);
    return s;
  }

  function loadKatex() {
    // 注意：首页用的是 .cover 而不是 .article，不能拿 .article 当守卫条件，
    // 否则封面页的公式永远不会被渲染。只要有 body 就加载。
    if (!document.body) return;
    const V = ROOT + 'assets/vendor/katex/';

    // 样式表：本地 → CDN（都不成功也不致命，公式会退化成衬线体）
    addCss(V + 'katex.min.css', () => addCss(KATEX_CDN + 'katex.min.css', null, SRI.css));

    const bootAutoRender = (base, sri) =>
      addScript(base + 'contrib/auto-render.min.js', renderMath, degradeMath, sri);

    // 主脚本：本地 → CDN → 降级
    addScript(V + 'katex.min.js',
      () => bootAutoRender(V, null),
      () => addScript(KATEX_CDN + 'katex.min.js',
        () => bootAutoRender(KATEX_CDN, SRI.ar),
        degradeMath,
        SRI.js
      )
    );
  }

  /* 彻底拿不到 KaTeX 时的体面降级：
     把 $...$ / $$...$$ 里的 LaTeX 源码包成等宽小块，
     读者看到的是"一段公式源码"，而不是莫名其妙的美元符号。 */
  function degradeMath() {
    document.documentElement.classList.add('no-katex');
    const SKIP = /^(SCRIPT|STYLE|CODE|PRE|TEXTAREA|OPTION|NOSCRIPT)$/;
    const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (SKIP.test(n.parentNode.nodeName) ? NodeFilter.FILTER_REJECT
        : (n.nodeValue.indexOf('$') >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT))
    });
    const targets = [];
    while (walk.nextNode()) targets.push(walk.currentNode);
    targets.forEach((node) => {
      const txt = node.nodeValue;
      if (!/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/.test(txt)) return;
      const frag = document.createDocumentFragment();
      let last = 0;
      const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
      let m;
      while ((m = re.exec(txt)) !== null) {
        if (m.index > last) frag.appendChild(document.createTextNode(txt.slice(last, m.index)));
        const span = document.createElement('span');
        span.className = 'mathsrc';
        span.textContent = (m[1] || m[2] || '').trim();
        frag.appendChild(span);
        last = m.index + m[0].length;
      }
      if (last < txt.length) frag.appendChild(document.createTextNode(txt.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  function renderMath() {
    if (typeof window.renderMathInElement !== 'function') {
      degradeMath();
      return;
    }
    try {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false }
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
        throwOnError: false,
        errorColor: '#b91c1c'
      });
    } catch (e) { /* 静默降级 */ }
  }

  /* ----------------------------------------------------------------------
     6. 表格包裹 / 小工具
     ---------------------------------------------------------------------- */
  function wrapTables() {
    $$('.article table').forEach((t) => {
      if (t.parentElement.classList.contains('table-wrap')) return;
      const w = document.createElement('div');
      w.className = 'table-wrap';
      t.parentNode.insertBefore(w, t);
      w.appendChild(t);
    });
  }

  /* ----------------------------------------------------------------------
     7. 启动
     ---------------------------------------------------------------------- */
  function boot() {
    // 每一步互相隔离：任何一个模块出问题，都不该连累公式渲染和其余功能。
    // 少一个滚动高亮可以忍，公式全变成源码不能忍。
    const safe = (label, fn) => {
      try { fn(); } catch (e) { console.warn('[book] ' + label + ' 初始化失败:', e); }
    };
    safe('顶栏', buildTopbar);
    safe('侧边栏', buildSidebar);
    safe('代码块', buildCodeBlocks);
    safe('表格包裹', wrapTables);
    safe('页内目录', buildPageMap);
    safe('章节导航', buildChapterNav);
    safe('阅读进度', initProgress);
    safe('公式渲染', loadKatex);
    safe('交互组件', () => { if (window.DLWidgets) window.DLWidgets.init(); });
    window.addEventListener('dl:theme', () => {
      if (window.DLWidgets) window.DLWidgets.redrawAll();
    });
    window.addEventListener('resize', () => {
      if (window.DLWidgets) window.DLWidgets.redrawAll();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();

  // 导出给 widgets.js 使用
  window.DL = { $, $$, esc, cssVar, BOOK, ROOT };
})();
