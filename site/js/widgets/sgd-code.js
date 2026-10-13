// widgets/sgd-code.js — short quotations of TorC's real source for the sgd-step lesson, and a small code panel that shows them
// with line numbers, a link to the pinned release on GitHub and optional highlighted lines.
//
// SOURCES holds the lines EXACTLY as they are in the library at the pinned ref (tools/trace/TORC_REF); `from`/`to` are 1-based file
// line numbers. site/test/sgd-step.test.mjs compares every quoted line with the fetched library source (build/trace/_deps/torc-src/)
// when that directory exists. Never edit a quoted line by hand: re-copy it from the library.
import { el } from '../core/dom.js';
import { REPO_URL } from '../core/chrome.js';
import { injectStyle } from './sgd-ui.js';

export const SOURCES = {
  "loop": {
    "file": "examples/linear_regression/linear_regression.cpp",
    "from": 66,
    "to": 68,
    "lines": [
      "            loss.backward();",
      "            optimizer.step();",
      "            optimizer.zero_grad();"
    ]
  },
  "step": {
    "file": "src/optim.cpp",
    "from": 17,
    "to": 30,
    "lines": [
      "void SGD::step() {",
      "    for (size_t i = 0; i < params_.size(); ++i) {",
      "        torc::Variable* param = params_[i];",
      "        if (!param->has_grad()) continue;",
      "",
      "        const torc::Tensor& grad = param->grad();",
      "        if (momentum_ != 0.0f) {",
      "            velocities_[i] = velocities_[i].mul(momentum_).add(grad);",
      "            param->data() = param->data().sub(velocities_[i].mul(lr_));",
      "        } else {",
      "            param->data() = param->data().sub(grad.mul(lr_));",
      "        }",
      "    }",
      "}"
    ]
  },
  "zero": {
    "file": "src/optim.cpp",
    "from": 32,
    "to": 36,
    "lines": [
      "void SGD::zero_grad() {",
      "    for (torc::Variable* param : params_) {",
      "        param->zero_grad();",
      "    }",
      "}"
    ]
  },
  "varzero": {
    "file": "src/autograd.cpp",
    "from": 110,
    "to": 113,
    "lines": [
      "void Variable::zero_grad() {",
      "    has_grad_ = false;",
      "    grad_ = Tensor(std::vector<int>{});",
      "}"
    ]
  },
  "accumulate": {
    "file": "include/torc/autograd.hpp",
    "from": 75,
    "to": 85,
    "lines": [
      "    void accumulate_grad(const Tensor& local_grad) {",
      "        if (!has_grad_) {",
      "            grad_ = Tensor(std::vector<int>(local_grad.shape().begin(), local_grad.shape().end()));",
      "            for (int i = 0; i < local_grad.numel(); ++i)",
      "                grad_.data()[i] = local_grad.data()[i];",
      "            has_grad_ = true;",
      "        } else {",
      "            for (int i = 0; i < grad_.numel(); ++i)",
      "                grad_.data()[i] += local_grad.data()[i];",
      "        }",
      "    }"
    ]
  }
};

const REPO = REPO_URL;
const CSS = `
.sgd-code{margin:.4rem 0;border:1px solid var(--border);border-radius:10px;background:#0b0f14;overflow:hidden}
.sgd-code figcaption{padding:.25rem .7rem;font-size:.8rem;color:var(--muted);border-bottom:1px solid var(--border);background:var(--panel2)}
.sgd-code figcaption a{color:var(--fwd)}
.sgd-code figcaption code{word-break:break-all}
.sgd-code pre{margin:0;padding:.35rem 0;overflow-x:auto;font-size:.82rem;line-height:1.5}
.sgd-code pre code{display:block;background:none;padding:0;font-size:1em;min-width:max-content}
.sgd-code .ln{display:block;padding:0 .8rem 0 0;white-space:pre;color:#c9d1d9}
.sgd-code .ln i{display:inline-block;width:3.2em;padding-right:.9em;text-align:right;color:#8b949e;font-style:normal;user-select:none}
.sgd-code .ln.hl{background:rgba(88,196,221,.18);box-shadow:inset 3px 0 0 var(--fwd);color:#fff}
.sgd-code .ln.off{text-decoration:line-through;color:#8b949e}
.sgd-code .ln b{color:var(--warn);font-weight:600;margin-left:1em;font-family:system-ui,sans-serif;font-size:.78rem;text-decoration:none;display:inline-block}
`;

// codePanel({key, ref, highlight: [fileLineNumbers], off: [fileLineNumbers], notes: {lineNo: 'text'}})
export function codePanel({ key, ref = 'main', highlight = [], off = [], notes = {} }) {
  injectStyle('sgd-code-css', CSS);
  const s = SOURCES[key];
  const indent = Math.min(...s.lines.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length));
  const fig = el('figure', { class: 'sgd-code' },
    el('figcaption', {}, 'TorC ', el('code', {}, s.file), ` lines ${s.from}–${s.to}, `,
      el('a', { href: `${REPO}/blob/${ref}/${s.file}#L${s.from}-L${s.to}` }, `release ${ref}`)),
    el('pre', {}, el('code', {}, s.lines.map((text, i) => {
      const n = s.from + i;
      return el('span', { class: `ln ${highlight.includes(n) ? 'hl' : ''} ${off.includes(n) ? 'off' : ''}`.trim(), 'data-line': n },
        el('i', {}, String(n)), text.slice(indent) || ' ', notes[n] ? el('b', {}, notes[n]) : null);
    }))));
  return fig;
}
