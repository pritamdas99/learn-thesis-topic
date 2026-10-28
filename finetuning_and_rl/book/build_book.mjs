// Build the course into one book: Markdown chapters (KaTeX math, highlighted code,
// as in VS Code's Markdown preview) followed by each chapter's companion script.
// Usage (from this folder): node build_book.mjs   ->  ../Finetuning_and_RL_Book.pdf
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import MarkdownIt from 'markdown-it';
import mdKatex from '@vscode/markdown-it-katex';
import hljs from 'highlight.js';
import puppeteer from 'puppeteer-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const outHtml = path.join(here, 'book.html');
const outPdf = path.join(root, 'Finetuning_and_RL_Book.pdf');

// ---------- Book structure ----------
const parts = [
  { title: 'Getting started', files: ['README.md', '00_start/LEARNING_MAP.md', '00_start/THEORY_INDEX.md',
    '00_start/RL_COVERAGE.md', '00_start/MATH_PRIMER.md', '00_start/TRICKS_AND_DEBUGGING.md', '00_start/GLOSSARY.md'] },
  { title: 'Foundations', dir: '01_foundations' },
  { title: 'Fine-tuning', dir: '02_finetuning' },
  { title: 'Reinforcement learning', dir: '03_reinforcement_learning' },
  { title: 'Alignment', dir: '04_alignment' },
  { title: 'Advanced topics', dir: '05_advanced' },
  { title: 'Projects and appendices', files: ['06_projects/PROJECTS.md', '00_start/SOURCES.md', 'VERIFICATION.md'] },
];
for (const p of parts) {
  if (p.dir) p.files = fs.readdirSync(path.join(root, p.dir)).filter(f => /^\d\d_.*\.md$/.test(f)).sort().map(f => `${p.dir}/${f}`);
}

const chapters = []; // {file, id, number, title, py}
parts.forEach((p, pi) => {
  p.number = pi;
  p.chapters = p.files.map(file => {
    const m = file.match(/^0(\d)_[^/]+\/(\d\d)_/);
    const number = m ? `${+m[1]}.${+m[2]}` : '';
    const pyFile = file.replace(/\.md$/, '.py');
    const ch = { file, id: 'ch-' + file.replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase(), number,
      py: fs.existsSync(path.join(root, pyFile)) ? pyFile : null };
    chapters.push(ch);
    return ch;
  });
});
const byFile = new Map(chapters.map(c => [c.file, c]));
const byPy = new Map(chapters.filter(c => c.py).map(c => [c.py, c]));

// ---------- Markdown ----------
const slugify = s => s.trim().toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const md = new MarkdownIt({
  html: true, linkify: true, typographer: false,
  highlight: (code, lang) => {
    const l = lang && hljs.getLanguage(lang) ? lang : (lang === 'text' || !lang ? null : 'plaintext');
    const body = l ? hljs.highlight(code, { language: l, ignoreIllegals: true }).value : esc(code);
    return `<pre class="hljs"><code>${body}</code></pre>`;
  },
});
md.use(mdKatex.default ?? mdKatex, { throwOnError: false, enableBareBlocks: true });

let current = null; // chapter being rendered
md.core.ruler.push('ids_and_links', state => {
  const seen = new Map();
  const toks = state.tokens;
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (t.type === 'heading_open') {
      const text = toks[i + 1].children.filter(c => c.type === 'text' || c.type === 'code_inline').map(c => c.content).join('');
      let s = slugify(text);
      const n = seen.get(s) ?? 0; seen.set(s, n + 1);
      if (n) s += '-' + n;
      t.attrSet('id', `${current.id}--${s}`);
      if (t.tag === 'h1') { // first h1 becomes the chapter title
        if (!current.title) { current.title = text; t.attrSet('id', current.id); t.attrSet('data-num', current.number); }
      } else if (t.tag === 'h2') t.attrSet('data-num', current.number);
    }
    for (const c of t.children ?? []) {
      if (c.type !== 'link_open') continue;
      const href = c.attrGet('href');
      if (/^[a-z]+:/i.test(href) || href.startsWith('#')) {
        if (href.startsWith('#')) c.attrSet('href', `#${current.id}--${href.slice(1)}`);
        continue;
      }
      const [p, frag] = href.split('#');
      const target = path.posix.normalize(path.posix.join(path.posix.dirname(current.file), decodeURIComponent(p)));
      if (byFile.has(target)) c.attrSet('href', '#' + byFile.get(target).id + (frag ? `--${frag}` : ''));
      else if (byPy.has(target)) c.attrSet('href', '#' + byPy.get(target).id + '--code');
      else c.attrSet('href', '#missing'), console.warn('unresolved link', current.file, href);
    }
  }
});

const renderPy = file => {
  const src = fs.readFileSync(path.join(root, file), 'utf8').replace(/\s+$/, '');
  const lines = hljs.highlight(src, { language: 'python' }).value.split('\n');
  // keep spans balanced per line so line numbers can wrap each line
  let open = [];
  const out = lines.map((ln, i) => {
    const prefix = open.join('');
    for (const m of ln.matchAll(/<span[^>]*>|<\/span>/g)) m[0].startsWith('</') ? open.pop() : open.push(m[0]);
    return `<span class="ln"><span class="no">${i + 1}</span>${prefix}${ln}${'</span>'.repeat(open.length)}</span>`;
  });
  return `<pre class="hljs listing"><code>${out.join('')}</code></pre>`;
};

// ---------- Assemble ----------
let body = '';
for (const p of parts) {
  body += `<section class="part"><div class="part-num">Part ${p.number}</div><h1 class="part-title">${p.title}</h1></section>`;
  for (const ch of p.chapters) {
    current = ch;
    const html = md.render(fs.readFileSync(path.join(root, ch.file), 'utf8'));
    body += `<section class="chapter"><div class="src">${ch.file}</div>${html}`;
    if (ch.py) {
      body += `<div class="code-appendix"><h2 id="${ch.id}--code" data-num="${ch.number}">Companion code: <code>${path.basename(ch.py)}</code></h2>
        <p class="src">${ch.py}</p>${renderPy(ch.py)}</div>`;
    }
    body += `</section>`;
  }
}

const toc = parts.map(p => `<li class="toc-part"><span>Part ${p.number} — ${p.title}</span><ol>` +
  p.chapters.map(c => `<li><a href="#${c.id}"><span class="toc-num">${c.number}</span>${esc(c.title)}</a>${c.py ? ` <span class="toc-py">+ ${path.basename(c.py)}</span>` : ''}</li>`).join('') +
  `</ol></li>`).join('');

const asset = rel => pathToFileURL(path.join(here, 'node_modules', rel)).href;
const today = new Date().toLocaleDateString('en-CA');
const doc = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Fine-tuning and Reinforcement Learning</title>
<link rel="stylesheet" href="${asset('@vscode/markdown-it-katex/node_modules/katex/dist/katex.min.css')}">
<link rel="stylesheet" href="${asset('highlight.js/styles/github.css')}">
<link rel="stylesheet" href="${pathToFileURL(path.join(here, 'book.css')).href}">
</head><body>
<section class="cover">
  <div class="cover-kicker">A self-study course</div>
  <h1>Fine-tuning and<br>Reinforcement Learning</h1>
  <p class="cover-sub">Theory, mathematics, and code — from tensors to RLHF, DPO and GRPO</p>
  <p class="cover-meta">${chapters.length} chapters · ${chapters.filter(c => c.py).length} companion scripts · built ${today}</p>
</section>
<nav class="toc"><h1>Contents</h1><ol>${toc}</ol></nav>
${body}
</body></html>`;
fs.writeFileSync(outHtml, doc);
console.log('wrote', outHtml);

// ---------- Print ----------
const browser = await puppeteer.launch({ executablePath: process.env.CHROME ?? '/usr/bin/google-chrome', args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.goto(pathToFileURL(outHtml).href, { waitUntil: 'networkidle0' });
await page.evaluateHandle('document.fonts.ready');
await page.pdf({
  path: outPdf, format: 'A4', printBackground: true, outline: true, tagged: true,
  margin: { top: '18mm', bottom: '18mm', left: '17mm', right: '17mm' },
  displayHeaderFooter: true,
  headerTemplate: '<span></span>',
  footerTemplate: `<div style="width:100%;font:8px sans-serif;color:#888;padding:0 17mm;display:flex;justify-content:space-between">
    <span>Fine-tuning and Reinforcement Learning</span><span class="pageNumber"></span></div>`,
});
await browser.close();
console.log('wrote', outPdf);
