# Lesson `from-digits-to-chatgpt` ("From digits to ChatGPT") — internal note

Not shipped. New page, written 2026-11-01 as a `template: 3` lesson with `conceptual: true` (brief §6 Part 5, decision 18). It is the
last page of the main path (Part 6 `next`), `loop: 'all'`, about 10 minutes, no trace file.

## What the page teaches
1. Text becomes tokens (a tokenizer, the vocabulary), and each token has an embedding, a vector of parameters that training learns.
2. Next-token prediction: one sentence gives one training example per token after the first, and the target is the token that
   really comes next. The text gives its own targets ("self-supervised learning" is named once). The loss is cross-entropy over
   the vocabulary, the same as over the ten digits of `losses`.
3. Attention: each token gives a weight to itself and to every earlier token, takes the weighted sum of vectors that stand for
   them, and the weights come from dot products followed by softmax, so the sum is a weighted average. Training learns how to
   make the vectors (the weights themselves are computed anew for each text). A token may look only back (the mask).
4. Pretraining runs this course's loop (data, model, loss, adjust; backpropagation, an optimizer). The first GPT model trained
   with Adam and AdamW's decoupled weight decay. Large language models have far more parameters and text (no count quoted).
5. After pretraining: fine-tuning on answers written by people (the same loss, targets from people) and learning from human
   feedback (rankings, a second model that scores answers, reinforcement learning). The last goal is not a next token.
6. Where TorC stops, and the curated list (3Blue1Brown, Karpathy's Zero to Hero, Nielsen, d2l.ai).

## Observed (TorC `38785db`, read in the pinned checkout)
- `include/torc/nn/*.hpp`: `Linear`, `Conv2d`, `Flatten`, `ReLU`, `Sigmoid`, `Softmax`, `MSELoss`, `CrossEntropyLoss`;
  `include/torc/optim.hpp`: `SGD`, `Adam`, `AdamW`; `include/torc/nn.hpp`: `Module`, `Sequential`. There is no embedding module,
  no attention module and no tokenizer (`grep -i "embedding|attention|transformer"` over `include/` and `src/` finds nothing).
  The page says "It has no layer for embeddings or attention, so this course stops at networks for images." It does not say that
  attention cannot be written from TorC's `matmul` and `softmax` ops (it could be, by hand); it says only that there is no layer.
- The digit network of `mnist-mlp` trains with `optim::AdamW` (its `course.json` `code`, checked by `tools/snippets/check.mjs`).
- The two numbers on the page are TorC anchors: `pixels` (784) and `mlp_parameters` (26,506), both from `mnist-mlp.json`.
- The cumulative quiz relies on `losses.json` `ten`: ten equal scores give each class 0.1 and the loss ln 10. The test checks it.

## Expected mathematics
- Next-token cross-entropy for one position: `−ln softmax(s)[next token]`, the same function as `nn::CrossEntropyLoss` for one row.
- With `n` equal scores, each probability is `1/n` and the loss is `ln n`, which grows with `n` (the cumulative quiz).
- Scaled dot-product attention (Vaswani et al. 2017, §3.2.1): weights `softmax(q·k / √d)` over the allowed positions, output the
  weighted sum of the value vectors. The weights are positive and add to one, so the output is a weighted average. The page
  leaves out the scaling, the separate query, key and value vectors (it says "vectors that it makes from the embeddings") and
  multiple heads.

## Suspected discrepancy
None in TorC (the page makes no claim about TorC beyond what it has).

## Sources (not fetched in this session: verify before quoting)
The session that wrote the page had no network access (arxiv.org and the other hosts did not resolve, and WebFetch was refused),
so every source below is cited from its known content and was not opened. Verify each one before quoting it elsewhere.
- Sennrich, Haddow, Birch (2016), arXiv 1508.07909: subword units (tokens that are parts of words).
- Vaswani et al. (2017), arXiv 1706.03762: §3.2.1 attention as a softmax-weighted sum, §3.2.3 the mask in the decoder (a position
  sees only earlier positions), §3.3 the feed-forward layers (two linear layers with ReLU between them), §3.4 learned embeddings.
- Radford et al. (2018), GPT, OpenAI PDF: §3.1 unsupervised pretraining by maximizing the likelihood of the next token on unlabeled
  text (the minus log-likelihood is the cross-entropy); §4.1 Adam and "a modified version of L2 regularization" from Loshchilov
  and Hutter, i.e. decoupled weight decay. The page says "the first GPT model trained with Adam and the weight decay of AdamW".
  Note: that model used GELU, not ReLU; the page says "an activation function such as ReLU", which the Vaswani design supports.
- Loshchilov and Hutter, arXiv 1711.05101: AdamW. The footer gives the year 2019 (the ICLR version); the arXiv abstract page is
  from 2017. Check.
- Brown et al. (2020), arXiv 2005.14165: much larger models trained on much more text ("far more parameters and read far more
  text"; no count is quoted on the page).
- Ouyang et al. (2022), arXiv 2203.02155: supervised fine-tuning on demonstrations written by labelers, a reward model from
  labelers' rankings, then reinforcement learning (PPO) against it.
- OpenAI (2022), "Introducing ChatGPT", `https://openai.com/index/chatgpt/`: ChatGPT was trained with RLHF with the methods of
  InstructGPT (Ouyang et al.). The URL is the least certain of the list (the blog moved from `/blog/` to `/index/`). Check it.
- Resources (home pages): 3Blue1Brown `https://www.3blue1brown.com/topics/neural-networks`; Karpathy
  `https://karpathy.ai/zero-to-hero.html`; Nielsen `https://neuralnetworksanddeeplearning.com/` (the site may serve only http;
  the footer requires https, so check that the https address loads); d2l.ai `https://d2l.ai/`. The one-line descriptions on the
  page (videos with moving pictures; build backpropagation and later a small GPT in Python; a book about a network for
  handwritten digits; a book with Python code from linear layers to attention and transformers) are from the authors' knowledge
  of these resources and the brief §13.

## Page choices
- One name: the course says **target** (glossary), so the page never says "label". "Piece of text" is used before "token" is
  defined, then "token" only.
- Deviation from the brief, for accuracy: the brief says that the later stages "use other objectives". Supervised fine-tuning
  uses the same next-token loss on answers written by people, so the page says that, and says that the human-feedback goal is the
  one that is not a next token.
- Attention "with weights it learns" (brief) is stated precisely: the weights are computed anew for every text, and training
  learns how to compute the vectors behind them. This keeps the contrast with the neuron of `linear-layer`, whose weights are
  parameters.
- The mask is taught (a token looks only back), because without it the reader could think that the model reads its own target.
  The check quiz `look-back` asks for the reason, which the text does not state; the widget's dashed tokens show it.
- Widget `widgets/next-token.js`: a made-up illustration (it says so), no numbers except "Example k of n" for the reader's own
  sentence. It cuts text at spaces and punctuation and says that a real tokenizer also cuts rare words into pieces. No TorC run.
- Figure: hand-chosen attention weights for "it" in "The cat sat on the mat because it was tired." as bar heights, no numbers
  shown, labelled as made up and not from a trained model or TorC. Kept in its own text section so the attention section stays
  under 150 words.
- Quizzes: predict (where the targets come from), cumulative from `losses` (equal scores over a vocabulary give a larger surprise
  than over ten digits), check (why look only back), explain. No transfer quiz (a conceptual lesson needs none, and the 1,400-word
  cap left no room; a melody transfer was drafted and cut). No refresher (the dot product is from `linear-layer`; the refresher
  would have pushed the page over the word cap).
- The prose is at 1,399 of 1,400 words (`checkV3`). Any addition needs a cut.
- Book titles with a colon or a phrasal verb are avoided in the prose (the STE linter flags "Dive into"): Karpathy's series is
  "Zero to Hero", and d2l.ai is named by its address. The footer carries the full titles.
