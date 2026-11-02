# Lesson `from-digits-to-chatgpt` ("From digits to ChatGPT") — internal note

Not shipped. New page, written 2026-11-01 as a `template: 3` lesson with `conceptual: true` (brief §6 Part 5, decision 18). It is the
last page of the main path (Part 6 `next`), `loop: 'all'`, about 10 minutes, no trace file.

## What the page teaches
1. Text becomes tokens (a tokenizer, the vocabulary), and each token has an embedding, a vector of parameters that training learns.
2. Next-token prediction: one sentence gives one sample per token after the first, and the target is the token that really
   comes next. The text gives its own targets ("is also called self-supervised learning", the name from d2l.ai). The loss is
   cross-entropy over the vocabulary, the same as over the ten digits of `losses`. The reader counts the samples of "I like green
   tea" (four tokens, no full stop, three samples).
   Writing (opening question 1, review H1): the trained model reads the text so far, softmax gives every token a probability, it
   picks a likely token, often at random with these probabilities (Holtzman et al. 2020), adds it and predicts again. A widget task
   shows the same move: going one sample on, the target joins the tokens that the model reads.
3. Attention: each token gives an attention weight to itself and to every earlier token, takes the weighted average of vectors
   that stand for them, and the attention weights come from dot products followed by softmax. A dot product is large when two
   vectors have large numbers in the same places, and training makes the vectors of "it" and "cat" match. The attention weights
   are computed anew for each text, unlike the weights of a neuron, which are parameters. A token may look only back (the mask).
   "Attention weights" is the one name after the contrast sentence (review M2; tested).
4. Pretraining runs this course's loop (data, model, loss, adjust; backpropagation, an optimizer). The first GPT model trained
   with Adam and AdamW's decoupled weight decay. "Large language models are far larger than any network of this course" (no count
   quoted; the 26,506 anchor sentence was dropped, review M7).
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
- The page binds no number. The only digits are the section numbers and the year 2022 of the first ChatGPT, cited to OpenAI (2022).
  The anchors `pixels` and `mlp_parameters` were dropped in the review round (word budget, M7).
- The cumulative quiz relies on `losses.json` `ten`: ten equal scores give each class 0.1 and the loss ln 10. The test checks it.

## Expected mathematics
- Next-token cross-entropy for one position: `−ln softmax(s)[next token]`, the same function as `nn::CrossEntropyLoss` for one row.
- With `n` equal scores, each probability is `1/n` and the loss is `ln n`, which grows with `n` (the cumulative quiz).
- Scaled dot-product attention (Vaswani et al. 2017, §3.2.1): weights `softmax(q·k / √d)` over the allowed positions, output the
  weighted sum of the value vectors. The weights are positive and add to one, so the output is a weighted average. The page
  leaves out the scaling, the separate query, key and value vectors (it says "vectors that it makes from the embeddings") and
  multiple heads.
- **Simplifications on the page (review M1).** The page describes one attention step on the embeddings. A real transformer stacks
  many layers, so later attention layers work on the outputs of earlier layers, not on the embeddings; the page says only that
  attention and layers of neurons "take turns many times". It names the use of token positions in one clause ("It also uses token
  positions", Vaswani et al. §3.5) and does not explain how. The vectors of "it" and "cat" that "match" are the query of "it" and
  the key of "cat"; the page does not name queries, keys and values.

## Suspected discrepancy
None in TorC (the page makes no claim about TorC beyond what it has).

## Sources (not fetched in this session: verify before quoting)
The session that wrote the page had no network access (arxiv.org and the other hosts did not resolve, and WebFetch was refused),
so every source below is cited from its known content and was not opened. Verify each one before quoting it elsewhere.
- Sennrich, Haddow, Birch (2016), arXiv 1508.07909: subword units (tokens that are parts of words).
- Vaswani et al. (2017), arXiv 1706.03762: §3.2.1 attention as a softmax-weighted sum, §3.2.3 the mask in the decoder (a position
  sees only earlier positions), §3.3 the feed-forward layers (two linear layers with ReLU between them), §3.4 learned embeddings,
  §3.5 positional encoding. (The first source title had the feed-forward layers under 3.2/3.4; fixed, review M10.)
- Holtzman et al. (2020), arXiv 1904.09751: generating text by sampling each next token from the predicted probabilities (and
  its variants); supports "often at random with these probabilities". Added for review H1.
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
- OpenAI (2022), "Introducing ChatGPT", `https://openai.com/index/chatgpt/`: the first ChatGPT (2022, the year on the page) was
  trained with RLHF with the methods of InstructGPT (Ouyang et al.). The URL is the least certain of the list (the blog moved from
  `/blog/` to `/index/`). Check it.
- d2l.ai (Zhang, Lipton, Li, Smola) is also cited for the name "self-supervised learning" (Radford et al. say "unsupervised
  pre-training"). Check which chapter uses the name before quoting it.
- Resources (home pages): 3Blue1Brown `https://www.3blue1brown.com/topics/neural-networks`; Karpathy
  `https://karpathy.ai/zero-to-hero.html`; Nielsen `https://neuralnetworksanddeeplearning.com/` (the site may serve only http;
  the footer requires https, so check that the https address loads); d2l.ai `https://d2l.ai/`. The one-line descriptions on the
  page (videos with moving pictures; build backpropagation and later a small GPT in Python; a book about a network for
  handwritten digits; a book with Python code from linear layers to attention and transformers) are from the authors' knowledge
  of these resources and the brief §13.

## Page choices
- One name: the course says **target** and **sample** (glossary), so the page never says "label" or "training example"
  (review M3, tested; the widget says "Next sample" and "Sample k of n"). "Piece of text" is used before "token" is defined, then
  "token" only.
- The setup does not define "language model" or say "continue" (it hinted at the predict, review M4); the definition comes in
  section 1, after the quiz. The three predict options have similar lengths, and each wrong one names its belief.
- Deviation from the brief, for accuracy: the brief says that the later stages "use other objectives". Supervised fine-tuning
  uses the same next-token loss on answers written by people, so the page says that, and says that the human-feedback goal is the
  one that is not a next token.
- Attention "with weights it learns" (brief) is stated precisely: the weights are computed anew for every text, and training
  learns how to compute the vectors behind them. This keeps the contrast with the neuron of `linear-layer`, whose weights are
  parameters.
- The mask is taught (a token looks only back), because without it the reader could think that the model reads its own target.
  The widget's key shows it ("the target, hidden from the model", "later tokens, hidden too"). The check quiz `look-back`, which
  asked for the reason, was cut in the review round to make room for the learner step (M6) and the writing paragraph (H1).
- Widget `widgets/next-token.js`: a made-up illustration (it says so), no numbers except "Sample k of n" for the reader's own
  sentence. It cuts text at spaces and punctuation and says that a real tokenizer also cuts rare words into pieces. No TorC run.
  The visible readout is not a live region. A hidden live region announces a step at once and typing after a pause of 0.7 s.
  The buttons use `aria-disabled` at the ends, so the focus stays on them. A fake-DOM test runs the widget in Node.
- Figure: hand-chosen attention weights for "it" in "The cat sat on the mat because it was tired." as bar heights, no numbers
  shown, labelled as made up and not from a trained model or TorC. Kept in its own text section so the attention section stays
  under 150 words.
- Quizzes: predict (where the targets come from), check "Your turn" (count the samples of a sentence, the learner step),
  cumulative from `losses` ("Suppose that a language model gives every token the same score": a supposition, since a random start
  gives scores that are only near-equal, review M8), explain. No transfer quiz (a conceptual lesson needs none, and the 1,400-word
  cap left no room; a melody transfer was drafted and cut). No refresher (the dot product is from `linear-layer`; the refresher
  would have pushed the page over the word cap).
- The prose is at 1,399 of 1,400 words (`checkV3`). Any addition needs a cut.
- The last line points to a next step (review M11): Karpathy's videos, which build a small GPT, or the appendix "Using TorC".
- Book titles with a colon or a phrasal verb are avoided in the prose (the STE linter flags "Dive into"): Karpathy's series is
  "Zero to Hero", and d2l.ai is named by its address. The footer carries the full titles.
