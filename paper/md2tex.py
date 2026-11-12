#!/usr/bin/env python3
"""Convert paper/paper.md to LaTeX for tectonic (figures via paper/figures)."""
import re, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'paper/paper.md'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper/build/paper.tex'
FIGREL = os.path.relpath('paper/figures', os.path.dirname(OUT) or '.')
BS = '\\'

UNICODE = {
    '±': r'$\pm$', '—': '---', '–': '--', 'β': r'$\beta$', '†': r'$\dagger$', '‡': r'$\ddagger$',
    '·': r'$\cdot$', '×': r'$\times$', '§': r'\S', '¶': r'\P', 'Δ': r'$\Delta$',
    '≈': r'$\approx$', '≤': r'$\le$', '≥': r'$\ge$', '→': r'$\rightarrow$', '−': '-',
    '’': "'", '‘': "`", '“': '``', '”': "''", '…': r'\ldots{}', 'ℓ': r'$\ell$',
    'm̂': r'$\hat m$', 'ν': r'$\nu$', 'λ': r'$\lambda$', 'σ': r'$\sigma$', 'δ': r'$\delta$',
    'φ': r'$\phi$', '∈': r'$\in$', '⊙': r'$\odot$', '²': r'$^{2}$', 'ã': r'\~a',
    'いう': r'\texttt{U+3044U+3046}',
}
SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
           '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
KEYS = sorted(UNICODE, key=len, reverse=True)

def esc(s):
    out = []; i = 0
    while i < len(s):
        for k in KEYS:
            if s.startswith(k, i):
                out.append(UNICODE[k]); i += len(k); break
        else:
            out.append(SPECIAL.get(s[i], s[i])); i += 1
    return ''.join(out)

def esc_tt(s):
    out = []
    for c in s:
        if c in SPECIAL: out.append(SPECIAL[c])
        elif c == '|': out.append(r'\textbar{}')
        elif c in '_-./:': out.append(c + r'\allowbreak{}')
        else: out.append(c)
    return ''.join(out)

def text_render(s):
    out = []; pos = 0
    for m in re.finditer(r'https?://[^\s,;()]+', s):
        out.append(esc(s[pos:m.start()]))
        out.append(r'\url{' + m.group(0).rstrip('.,').replace('%', r'\%') + '}')
        pos = m.end()
    out.append(esc(s[pos:]))
    return ''.join(out)

def render_prose(v):
    out = []; pos = 0
    for m in re.finditer(r'\[(\d{1,2})\]', v):
        n = int(m.group(1))
        if not (1 <= n <= 27): continue
        out.append(text_render(v[pos:m.start()])); out.append(r'\cite{ref%d}' % n); pos = m.end()
    out.append(text_render(v[pos:]))
    return ''.join(out)

def inline(s):
    parts, pos = [], 0
    pattern = re.compile(r'(\$\$.*?\$\$|\$[^$\n]*\$|`[^`]*`|\*\*[^*]+\*\*|(?<!\*)\*[^*\n]+\*(?!\*))', re.S)
    for m in pattern.finditer(s):
        parts.append(('text', s[pos:m.start()])); tok = m.group(0)
        if tok.startswith('$$'): parts.append(('math', tok[2:-2].strip()))
        elif tok.startswith('$'): parts.append(('math', tok[1:-1]))
        elif tok.startswith('`'): parts.append(('code', tok[1:-1]))
        elif tok.startswith('**'): parts.append(('bold', tok[2:-2]))
        else: parts.append(('ital', tok[1:-1]))
        pos = m.end()
    parts.append(('text', s[pos:]))
    out = []
    for kind, val in parts:
        if kind == 'math': out.append('$' + val + '$')
        elif kind == 'code': out.append(r'\texttt{' + esc_tt(val) + '}')
        elif kind == 'bold': out.append(r'\textbf{' + inline(val) + '}')
        elif kind == 'ital': out.append(r'\emph{' + inline(val) + '}')
        else:
            v = re.sub(r'"([^"]{1,400}?)"', lambda m: '``' + m.group(1) + "''", val)
            out.append(render_prose(v))
    return ''.join(out)

def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]

def table_tex(rows, caption):
    header, body = rows[0], rows[2:]
    ncol = len(header)
    body = [(r + [''] * ncol)[:ncol] for r in body]
    avg = sum(len(c) for r in body for c in r) / max(1, len(body) * ncol)
    cap_tex = inline(caption) if caption else ''
    if avg > 40 and ncol <= 6:
        maxlens = [max([len(header[j])] + [len(r[j]) for r in body]) for j in range(ncol)]
        if ncol == 4:
            widths = [1.1, 3.3, 3.5, 7.3]
        else:
            widths = [max(1.3, min(4.2, 0.22 * ml + 1.0)) for ml in maxlens[:-1]]
            widths.append(max(3.0, 15.0 - sum(widths)))
        spec = '@{}' + ' '.join('>{' + BS + 'raggedright' + BS + 'arraybackslash}p{%.1fcm}' % w for w in widths) + '@{}'
        head = ' & '.join(inline(c) for c in header) + BS*2
        rows_tex = (BS + '\n').join(' & '.join(inline(c) for c in r) + BS*2 for r in body)
        L = []
        L.append(BS + 'begin{longtable}{' + spec + '}')
        L.append(BS + 'caption{' + cap_tex + '}' + BS*2)
        L.append(BS + 'toprule')
        L.append(head)
        L.append(BS + 'midrule' + BS + 'endfirsthead')
        L.append(BS + 'caption[]{}' + BS*2)
        L.append(BS + 'toprule')
        L.append(head)
        L.append(BS + 'midrule' + BS + 'endhead')
        L.append(BS + 'bottomrule' + BS + 'endlastfoot')
        L.append(rows_tex)
        L.append(BS + 'end{longtable}')
        return '\n'.join(L) + '\n'
    align = 'l' + 'c' * (ncol - 1)
    inner = ' & '.join(inline(c) for c in header) + r' \\ \midrule' + '\n'
    inner += (BS + '\n').join(' & '.join(inline(c) for c in r) + BS*2 for r in body) + '\n'
    tab = BS + 'begin{tabular}{' + align + '}\n' + BS + 'toprule\n' + inner + BS + 'bottomrule\n' + BS + 'end{tabular}'
    est = sum(max([len(c) for c in [header[j]] + [r[j] for r in body]] or [0]) for j in range(ncol))
    if ncol > 6 or est > 90:
        tab = BS + 'resizebox{' + BS + 'textwidth}{!}{%\n' + tab + '\n}'
    out = BS + 'begin{table}[t]\n' + BS + 'centering\n' + BS + 'small\n' + tab + '\n'
    if cap_tex: out += BS + 'caption{' + cap_tex + '}\n'
    out += BS + 'end{table}\n'
    return out

def parse(raw):
    raw = re.sub(r'<!--.*?-->', '', raw, flags=re.S)
    lines = raw.split('\n'); blocks = []; buf = []
    def flush():
        nonlocal buf
        if buf and any(x.strip() for x in buf):
            blocks.append(('para', '\n'.join(buf).strip()))
        buf = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            i += 1; blocks.append(('code', '\n'.join(code))); continue
        if re.match(r'^#{1,3} ', line):
            flush(); blocks.append(('head', line)); i += 1; continue
        if re.match(r'^\s*\|', line) and i + 1 < len(lines) and re.match(r'^\s*\|[-: |]+\|\s*$', lines[i + 1]):
            flush(); rows = []
            while i < len(lines) and re.match(r'^\s*\|', lines[i]):
                rows.append(split_row(lines[i])); i += 1
            blocks.append(('table', rows)); continue
        if re.match(r'^\s*\{\{', line):
            i += 1; continue
        if line.strip() == '':
            flush(); i += 1; continue
        if re.match(r'^(\s*[-*] |\s*\d+\. )', line):
            flush(); items = []; ordered = bool(re.match(r'^\s*\d+\. ', line))
            while i < len(lines) and re.match(r'^(\s*[-*] |\s*\d+\. )', lines[i]):
                m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', lines[i]); items.append(m.group(3)); i += 1
                while i < len(lines) and lines[i].startswith('  ') and lines[i].strip() and not re.match(r'^\s*([-*]|\d+\.) ', lines[i]):
                    items[-1] += ' ' + lines[i].strip(); i += 1
            blocks.append(('list', (ordered, items))); continue
        buf.append(line); i += 1
    flush()
    merged = []
    for blk in blocks:
        if merged and blk[0] == 'list' and merged[-1][0] == 'list':
            o1, i1 = merged[-1][1]; o2, i2 = blk[1]
            if o1 == o2:
                merged[-1] = ('list', (o1, i1 + i2)); continue
        merged.append(blk)
    return merged

FIGURES = [
    ('fig_methods_accuracy.png', r'Methods on AG News with 95\% bootstrap intervals: the w2 edits leave says-correct near 10\%, the output-head and gated patches move the decision.'),
    ('fig_task_vs_format.png', r'Label-restricted accuracy against says-correct. The w2 and dense w2 edits (left) move evidence without the decision; the readout and gated patches move the decision.'),
    ('fig_flip_layers.png', r'Per-layer true residual gap of the $\pm1$ patch against the continuous $\sigma$-space reference, and the dense $\Delta w_2$ RMS relative to weight RMS together with the post-FFN gain RMS.'),
    ('fig_gate_leak.png', r'Triggered accuracy against off-state first-token identity for the gated patches and the row patches (which leak fully by construction).'),
    ('fig_patch_size.png', r'Patch bytes (raw) against says-correct. A five-row gate costs 7,680 B; the one-row format skill 1,536 B; the w2 patch 12.6 MB compressed.'),
]

PREAMBLE = r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.1cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{listings}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,frame=single,framerule=0.2pt,rulecolor=\color{gray},backgroundcolor=\color{gray!8}}
\usepackage{xcolor}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.3em}
\setlength{\parindent}{0pt}
\sloppy
\title{\textbf{TITLE_PLACEHOLDER}}
\author{AUTHOR_PLACEHOLDER}
\date{12 November 2026}
\begin{document}
\maketitle
'''

def read_title(raw):
    for ln in raw.split('\n'):
        if ln.startswith('# '):
            return ln[2:].strip()
    return 'TITLE_PLACEHOLDER'


def read_author(raw):
    lines = [ln.strip() for ln in raw.split('\n')]
    for ln in lines[:8]:
        if not ln or ln.startswith('#') or ln.startswith('<!--'):
            continue
        if ln.startswith('{{'):
            continue
        return ln
    return 'Anonymous Author(s)'


def main():
    raw = open(SRC).read()
    author = read_author(raw)
    title = read_title(raw)
    raw = raw.replace(author, '', 1)          # the author line belongs to \maketitle only
    blocks = parse(raw)
    pre = PREAMBLE.replace('AUTHOR_PLACEHOLDER', author).replace('TITLE_PLACEHOLDER', title)
    if True:   # compact 8-page layout
        pre = pre.replace('[11pt]{article}', '[10pt]{article}').replace('margin=2.1cm', 'margin=1.9cm').replace('\\parskip}{0.3em', '\\parskip}{0.15em')
    tex = [pre]
    in_abstract = False; pending_caption = None; refs = []; in_refs = False
    for kind, val in blocks:
        if kind == 'head':
            m = re.match(r'^(#{1,3}) (.*)$', val); level, title = len(m.group(1)), m.group(2).strip()
            title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', title)
            if level == 1: continue
            if title.lower() == 'abstract':
                tex.append(r'\begin{abstract}'); in_abstract = True; continue
            if in_abstract: tex.append(r'\end{abstract}'); in_abstract = False
            if title.lower() == 'references':
                tex.append(BS + 'clearpage'); in_refs = True; continue
            if in_refs:
                in_refs = False
                tex.append('{' + BS + 'raggedright' + BS + 'begin{thebibliography}{99}\n' + '\n'.join(
                    BS + 'bibitem{ref%d} %s' % (n, t) for n, t in refs) + '\n' + BS + 'end{thebibliography}}\n')
                refs = []
            tex.append((BS + 'section{' if level == 2 else BS + 'subsection{') + inline(title) + '}')
            continue
        if kind == 'para':
            if in_refs:
                mref = re.match(r'^\[(\d+)\]\s*(.*)$', val, flags=re.S)
                if mref: refs.append((int(mref.group(1)), inline(mref.group(2))))
                else: refs.append((len(refs) + 1, inline(val)))
                continue
            m = re.match(r'^Table\s+(\S+)\.\s*(.*)$', val, flags=re.S)
            if m: pending_caption = ('Table ' + m.group(1), m.group(2).strip()); continue
            if in_abstract: tex.append(inline(val)); continue
            if val.strip().startswith('$$') and val.strip().endswith('$$'):
                tex.append('\\[\n' + val.strip()[2:-2].strip() + '\n\\]\n'); continue
            if 'fig_methods_accuracy' in val:
                tex.append(inline(val) + '\n')
                for fn, cap in FIGURES:
                    tex.append(BS + 'begin{figure}[t]\n' + BS + 'centering\n' + BS + 'includegraphics[width=0.78' + BS + 'textwidth]{' + FIGREL + '/' + fn + '}\n' + BS + 'caption{' + cap + '}\n' + BS + 'end{figure}\n')
                continue
            mfig = re.match(r'^!\[(.*?)\]\((\w+)\)$', val.strip(), flags=re.S)
            if mfig:
                cap = inline(mfig.group(1)); fname = {'FIG': 'fig_task_vs_format.png', 'FIG2': 'fig_gate_leak.png'}.get(mfig.group(2), mfig.group(2))
                tex.append(BS + 'begin{figure}[t]\n' + BS + 'centering\n' + BS + 'includegraphics[width=0.70' + BS + 'textwidth]{' + FIGREL + '/' + fname + '}\n' + BS + 'caption{' + cap + '}\n' + BS + 'end{figure}\n')
                continue
            tex.append(inline(val) + '\n'); continue
        if kind == 'list':
            ordered, items = val; env = 'enumerate' if ordered else 'itemize'
            tex.append(BS + 'begin{' + env + '}\n' + '\n'.join(BS + 'item ' + inline(x) for x in items) + '\n' + BS + 'end{' + env + '}\n'); continue
        if kind == 'code':
            tex.append(BS + 'begin{lstlisting}\n' + val + '\n' + BS + 'end{lstlisting}\n'); continue
        if kind == 'table':
            cap = pending_caption[1] if pending_caption else ''
            tex.append(table_tex(val, cap)); pending_caption = None; continue
    if in_abstract: tex.append(r'\end{abstract}')
    if refs:
        tex.append('{' + BS + 'raggedright' + BS + 'begin{thebibliography}{99}\n' + '\n'.join(BS + 'bibitem{ref%d} %s' % (n, t) for n, t in refs) + '\n' + BS + 'end{thebibliography}}\n')
    tex.append('\n' + BS + 'end{document}\n')
    os.makedirs('paper/build', exist_ok=True)
    open(OUT, 'w').write('\n'.join(tex))
    print('wrote', OUT)

if __name__ == '__main__': main()
