#!/usr/bin/env node
// tools/snippets/check.mjs — compiles every code snippet of site/course.json against a TorC checkout and runs the ones that can run,
// so the code shown on the home page is real API at the pinned ref, and the values its comments state are what TorC computes.
//
//   node tools/snippets/check.mjs --torc <TorC source dir> [--build <dir>] [--cxx clang++-18]
//
// The TorC checkout must be at the ref in tools/trace/TORC_REF (tools/trace/run.sh fetches it to build/trace/_deps/torc-src); a git
// checkout at another commit is refused. The library is (re)built incrementally in <build>/torc every run.
// Each snippet runs inside its own function after the lesson's PRELUDE (the variables it assumes) and before its CHECK (assertions
// on the values its comments state). A lesson in course.json without an entry here fails the check, so a new snippet cannot skip it.
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const TORC = arg('--torc', path.join(ROOT, 'build/trace/_deps/torc-src'));
const BUILD = path.resolve(arg('--build', path.join(ROOT, 'build/snippets')));
const CXX = arg('--cxx', 'clang++-18');

const shape = (...d) => `Tensor(std::vector<int>{${d.join(', ')}})`;
// run: false = compile only (needs files or a long training run)
export const SPEC = {
  'why-gradients': {
    prelude: 'Variable x(Tensor({2.0f}, {1}), false);\nVariable t(Tensor({6.0f}, {1}), false);',
    check: 'CHECK(w.grad().data()[0] == -24.0f);',
  },
  'sgd-step': {
    prelude: `Variable W(Tensor({0.0f}, {1, 1}), true);\nVariable b(Tensor({0.0f}, {1}), true);\nVariable xin(Tensor({1.0f, 2.0f, 3.0f}, {3, 1}), false);\nVariable tgt(Tensor({3.0f, 5.0f, 7.0f}, {3, 1}), false);\nVariable Wt = transpose(W);\nVariable xW = matmul(xin, Wt);\nVariable yhat = add(xW, b);\nnn::MSELoss mse;\nVariable loss = mse(yhat, tgt);`,
    check: 'CHECK(!W.has_grad() && !b.has_grad());\nCHECK(std::abs(W.data().data()[0] - 1.1333334f) < 1e-5f);',
  },
  tensors: { prelude: '', check: 'CHECK((c.shape() == std::vector<int>{2, 3}));\nCHECK(c.data()[3] == 14.0f);\nCHECK((g.shape() == std::vector<int>{2, 2}));\nCHECK(g.data()[0] == 14.0f);' },
  'linear-layer': {
    prelude: `Variable x(${shape(8, 784)}, false);`,
    check: 'CHECK((y.data().shape() == std::vector<int>{8, 32}));\nconst auto& np = layer.named_parameters();\nCHECK((np.at("weight").data().shape() == std::vector<int>{32, 784}));\nCHECK((np.at("bias").data().shape() == std::vector<int>{32}));',
  },
  nonlinearity: {
    prelude: `Variable z(Tensor({-1.0f, 0.0f, 2.0f, 1.0f, 1.0f, 1.0f}, {2, 3}), true);`,
    check: 'CHECK(h.data().data()[0] == 0.0f && h.data().data()[2] == 2.0f);\nCHECK(std::abs(s.data().data()[1] - 0.5f) < 1e-6f);\nCHECK(std::abs(p.data().data()[3] + p.data().data()[4] + p.data().data()[5] - 1.0f) < 1e-6f);',
  },
  modules: { prelude: '', check: 'CHECK(params.size() == 4);\nint n = 0; for (auto* q : params) n += q->data().numel();\nCHECK(n == 784 * 32 + 32 + 32 * 10 + 10);' },
  losses: {
    prelude: `Variable logits(${shape(4, 10)}, true);\nVariable labels(Tensor({0.0f, 3.0f, 9.0f, 1.0f}, {4}), false);`,
    check: 'CHECK(std::abs(loss.data().data()[0] - std::log(10.0f)) < 1e-5f);   // equal logits: -log(1/10)\n(void)mse;',
  },
  'training-loop': {
    prelude: `data::TensorDataset dataset(${shape(100, 1)}, ${shape(100, 1)});\nint batches = 0;`,
    after: '++batches;',
    check: 'CHECK(batches == 4);   // 32 + 32 + 32 + 4',
  },
  // each optimizer takes one step on L = w² from w = 1 (gradient 2): every one must move w down
  optimizers: {
    prelude: 'Variable w(Tensor({1.0f}, {1}), true);\nstd::vector<Variable*> params{&w};',
    check: 'auto down = [&](auto& opt) { float before = w.data().data()[0]; Variable sq = mul(w, w); Variable L = sum(sq); L.backward(); opt.step(); opt.zero_grad(); return w.data().data()[0] < before; };\nCHECK(down(sgd));\nCHECK(down(adam));\nCHECK(down(adamw));',
  },
  generalization: {
    prelude: `Variable w(${shape(3)}, true);\nstd::vector<Variable*> params{&w};\ndata::TensorDataset train_set(${shape(128, 1)}, ${shape(128, 1)});\ndata::TensorDataset test_set(${shape(32, 1)}, ${shape(32, 1)});`,
    check: 'int nt = 0, ns = 0;\nwhile (train.has_next()) { train.next_batch(); ++nt; }\nwhile (test.has_next()) { test.next_batch(); ++ns; }\nCHECK(nt == 2 && ns == 1);   // 128 / 64 and 32 / 256\n(void)opt;',
  },
  'mnist-mlp': {
    run: false,
    prelude: 'std::string path = "mnist_train.csv";\nnn::Sequential model;\nmodel.add(std::make_unique<nn::Linear>(784, 32));\nmodel.add(std::make_unique<nn::ReLU>());\nmodel.add(std::make_unique<nn::Linear>(32, 32));\nmodel.add(std::make_unique<nn::ReLU>());\nmodel.add(std::make_unique<nn::Linear>(32, 10));',
    check: 'int n = 0; for (auto* q : model.parameters()) n += q->data().numel();\nCHECK(n == 26506);\n(void)loader; (void)opt; (void)loss_fn;',
  },
  convolution: { prelude: `Variable x(${shape(2, 1, 28, 28)}, false);`, check: 'CHECK((y.data().shape() == std::vector<int>{2, 8, 14, 14}));\nVariable f = flat(y);\nCHECK((f.data().shape() == std::vector<int>{2, 8 * 14 * 14}));' },
  'mnist-cnn': {
    prelude: 'nn::Sequential model;',
    check: `int n = 0; for (auto* q : model.parameters()) n += q->data().numel();\nCHECK(n == 9098);\nVariable x(${shape(2, 1, 28, 28)}, false);\nVariable out = model(x);\nCHECK((out.data().shape() == std::vector<int>{2, 10}));`,
  },
  'backprop-lab': { prelude: '', check: 'CHECK(x.grad().data()[0] == 7.0f);' },
};

function program(course) {
  const lessons = course.parts.flatMap((p) => p.lessons).filter((l) => l.code);
  const missing = lessons.filter((l) => !SPEC[l.id]).map((l) => l.id);
  if (missing.length) throw new Error(`no SPEC entry for: ${missing.join(', ')} (add the prelude and checks in tools/snippets/check.mjs)`);
  const indent = (s, n = 4) => s.split('\n').map((x) => (x ? ' '.repeat(n) + x : x)).join('\n');
  const fn = (l) => {
    const s = SPEC[l.id];
    // the training-loop snippet's body is a comment; `after` adds a statement inside its loop so the loop can be counted
    const code = s.after ? l.code.replace(/\n\}$/, `\n    ${s.after}\n}`) : l.code;
    return `// ${l.id}\nstatic void snippet_${l.id.replace(/-/g, '_')}() {\n${indent(s.prelude)}\n${indent(code)}\n${indent(s.check)}\n}`;
  };
  const calls = lessons.filter((l) => SPEC[l.id].run !== false).map((l) => `    snippet_${l.id.replace(/-/g, '_')}(); std::puts("ok  ${l.id}");`);
  const compileOnly = lessons.filter((l) => SPEC[l.id].run === false).map((l) => `    (void)&snippet_${l.id.replace(/-/g, '_')}; std::puts("compiled only  ${l.id}");`);
  return `// generated by tools/snippets/check.mjs from site/course.json — do not edit
#include <torc/autograd.hpp>
#include <torc/nn.hpp>
#include <torc/nn/linear.hpp>
#include <torc/nn/activations.hpp>
#include <torc/nn/losses.hpp>
#include <torc/nn/conv.hpp>
#include <torc/optim.hpp>
#include <torc/data.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
using namespace torc;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

${lessons.map(fn).join('\n\n')}

int main() {
${[...calls, ...compileOnly].join('\n')}
}
`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const course = JSON.parse(readFileSync(path.join(ROOT, 'site/course.json'), 'utf8'));
  if (!existsSync(path.join(TORC, 'include/torc/autograd.hpp'))) { console.error(`no TorC checkout at ${TORC} (pass --torc <dir>)`); process.exit(2); }
  // the checkout must be the pinned release
  const ref = readFileSync(path.join(ROOT, 'tools/trace/TORC_REF'), 'utf8').split('\n').map((l) => l.trim()).find((l) => l && !l.startsWith('#'));
  if (existsSync(path.join(TORC, '.git'))) {
    const git = (...a) => execFileSync('git', ['-C', TORC, ...a], { encoding: 'utf8' }).trim();
    const have = git('rev-parse', 'HEAD');
    let want = null;
    try { want = git('rev-parse', `${ref}^{commit}`); } catch { /* the ref is unknown to this checkout */ }
    if (have !== want) { console.error(`${TORC} is at ${have}, not at the pinned TorC ref ${ref} (tools/trace/TORC_REF)`); process.exit(2); }
  } else console.warn(`warning: ${TORC} is not a git checkout; cannot confirm it is TorC ${ref}`);
  mkdirSync(BUILD, { recursive: true });
  const libDir = path.join(BUILD, 'torc'), lib = path.join(libDir, 'libtorc.a');
  // a build directory configured for another source tree is discarded; otherwise cmake rebuilds whatever changed
  const cache = path.join(libDir, 'CMakeCache.txt');
  if (existsSync(cache) && !readFileSync(cache, 'utf8').includes(`CMAKE_HOME_DIRECTORY:INTERNAL=${path.resolve(TORC)}`)) rmSync(libDir, { recursive: true, force: true });
  execFileSync('cmake', ['-S', TORC, '-B', libDir, '-DCMAKE_BUILD_TYPE=Release', `-DCMAKE_CXX_COMPILER=${CXX}`, '-DTORC_BUILD_TESTS=OFF', '-DTORC_BUILD_EXAMPLES=OFF', '-DTORC_NATIVE_ARCH=OFF', '-DTORC_INSTALL=OFF'], { stdio: ['ignore', 'ignore', 'inherit'] });
  execFileSync('cmake', ['--build', libDir, '-j8'], { stdio: ['ignore', 'ignore', 'inherit'] });
  const cpp = path.join(BUILD, 'snippets.cpp'), exe = path.join(BUILD, 'snippets');
  writeFileSync(cpp, program(course));
  execFileSync(CXX, ['-std=c++23', '-O1', '-I', path.join(TORC, 'include'), cpp, lib, '-o', exe], { stdio: 'inherit' });
  execFileSync(exe, { stdio: 'inherit' });
  console.log('snippets: ok');
}
