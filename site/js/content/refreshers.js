// content/refreshers.js — the maths refreshers (brief §8, R9; decision 18): short, collapsible explanations that a lesson opens just
// in time with refresher(key) from core/blocks.js. Each is { title, html, source }: title is plain text for the <summary>, html is
// the body (100-200 words, with a picture), source is { title, url } and is cited in the last line of the body.
// The html is static markup, so it must not call KaTeX or T(). It is school mathematics, not a TorC run: the numbers are small
// arithmetic examples that the reader can check by hand, and they are written inside <code> (never a trace value).
//
// Rules for the html (tested in site/test/reference.test.mjs):
//   - one <figure> with an inline SVG (role="img", an aria-label that describes the picture), drawn with the colour tokens of
//     base.css (var(--fg), var(--muted), var(--fwd), …), so it works in the dark theme and at 390 px;
//   - few glossary words, because the first use of a glossary term in a lesson must be a term() link (site/test/v3.mjs): a lesson
//     places a refresher after its own term() link of the words listed in REFRESHER_TERMS of the test;
//   - a symbol of content/notation.js only where the lesson that uses the refresher already introduced it (sum, dot-product and
//     matrix-product use Σ and subscripts, so linear-layer or later; exp-ln uses exp and ln, so losses or later).
const W = 300;

// an SVG figure that scales down on a phone and keeps its text readable
const fig = (h, label, body) => `<figure class="refresher-fig" style="margin:.7rem 0">`
  + `<svg viewBox="0 0 ${W} ${h}" width="${W}" height="${h}" role="img" aria-label="${label}" focusable="false" `
  + `style="display:block;width:100%;max-width:${W}px;height:auto;font:13px system-ui,-apple-system,'Segoe UI',Roboto,sans-serif">${body}</svg></figure>`;
const line = (x1, y1, x2, y2, color = 'var(--muted)', extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:${color};stroke-width:2${extra}"/>`;
const txt = (x, y, s, color = 'var(--muted)', anchor = 'middle', extra = '') => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="fill:${color}${extra}">${s}</text>`;
const dot = (x, y, color = 'var(--fwd)', r = 4) => `<circle cx="${x}" cy="${y}" r="${r}" style="fill:${color}"/>`;
const poly = (pts, color = 'var(--fwd)', extra = '') => `<polyline points="${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')}" style="fill:none;stroke:${color};stroke-width:2.5${extra}"/>`;
const curve = (f, x0, x1, n = 60) => Array.from({ length: n + 1 }, (_, i) => { const x = x0 + ((x1 - x0) * i) / n; return [x, f(x)]; });
const box = (x, y, w, h, s, color = 'var(--border)', fill = 'var(--panel2)') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" style="fill:${fill};stroke:${color};stroke-width:1.5"/>`
  + txt(x + w / 2, y + h / 2 + 5, s, 'var(--fg)');
const source = (s) => `<p class="hint">This is school mathematics, not a TorC run. A textbook with more examples is <a href="${s.url}">${s.title}</a>.</p>`;
const OPENSTAX = (book, title) => ({ title: `OpenStax, ${title}`, url: `https://openstax.org/details/books/${book}` });

// ---- the pictures ---------------------------------------------------------------------------------------------------------
// y = x² from x = 0 to 4, with a nudge from x = 3 (drawn wider than a real nudge, so that it is visible)
const slopeFig = (() => {
  const X = (x) => 30 + x * 60, Y = (y) => 170 - y * 9;
  const [x0, x1] = [3, 3.6];
  return fig(185, 'The curve y equals x squared. A small step to the right from one point gives a small change in height. The slope is the change in height divided by the step.',
    line(30, 170, 290, 170) + line(30, 170, 30, 10)
    + poly(curve((x) => x * x, 0, 4.2).map(([x, y]) => [X(x), Y(y)]))
    + line(X(x0), Y(x0 * x0), X(x1), Y(x0 * x0), 'var(--warn)', ';stroke-dasharray:4 3')
    + line(X(x1), Y(x0 * x0), X(x1), Y(x1 * x1), 'var(--warn)', ';stroke-dasharray:4 3')
    + dot(X(x0), Y(x0 * x0)) + dot(X(x1), Y(x1 * x1))
    + txt((X(x0) + X(x1)) / 2, Y(x0 * x0) + 18, 'nudge', 'var(--warn)')
    + txt(X(x1) + 6, (Y(x0 * x0) + Y(x1 * x1)) / 2 + 4, 'change', 'var(--warn)', 'start')
    + txt(X(1.4), Y(1.4 * 1.4) - 12, 'y = x²', 'var(--fwd)'));
})();

const sumFig = fig(110, 'The sum sign with the index i from 0 to 2 over x sub i. It means x sub 0 plus x sub 1 plus x sub 2, which is 3 plus 1 plus 4, which is 8.',
  txt(40, 62, 'Σ', 'var(--fg)', 'middle', ';font-size:44px')
  + txt(40, 92, 'i = 0', 'var(--muted)') + txt(40, 22, '2', 'var(--muted)') + txt(78, 60, 'xᵢ', 'var(--fg)', 'middle', ';font-size:20px')
  + line(100, 55, 130, 55, 'var(--muted)') + `<polygon points="136,55 128,51 128,59" style="fill:var(--muted)"/>`
  + txt(215, 50, 'x₀ + x₁ + x₂', 'var(--fg)', 'middle', ';font-size:16px') + txt(215, 78, '3 + 1 + 4 = 8', 'var(--fwd)', 'middle', ';font-size:16px'));

const dotFig = fig(150, 'Two lists of two numbers, 2 and 3, and 4 and 1. The numbers in the same position are multiplied, 2 times 4 is 8 and 3 times 1 is 3. The products are added to 11.',
  box(40, 10, 50, 32, '2') + box(100, 10, 50, 32, '3') + txt(20, 32, 'a', 'var(--muted)')
  + box(40, 56, 50, 32, '4', 'var(--fwd)') + box(100, 56, 50, 32, '1', 'var(--fwd)') + txt(20, 78, 'b', 'var(--muted)')
  + txt(65, 112, '2 × 4', 'var(--fg)') + txt(125, 112, '3 × 1', 'var(--fg)')
  + txt(65, 136, '= 8', 'var(--fg)') + txt(125, 136, '= 3', 'var(--fg)')
  + line(165, 72, 200, 72) + `<polygon points="206,72 198,68 198,76" style="fill:var(--muted)"/>`
  + box(212, 54, 80, 36, '8 + 3 = 11', 'var(--ok)'));

const matFig = (() => {
  const cell = (x, y, s, on) => box(x, y, 30, 26, s, on ? 'var(--fwd)' : 'var(--border)', on ? 'var(--panel)' : 'var(--panel2)');
  const grid = (x, y, v, on) => v.map((r, i) => r.map((s, j) => cell(x + j * 32, y + i * 28, s, on(i, j))).join('')).join('');
  return fig(110, 'The matrix with rows 1, 2 and 3, 4 times the matrix with rows 5, 6 and 7, 8. Row 0 of the first matrix and column 0 of the second give the entry 19 in row 0, column 0 of the result.',
    grid(6, 30, [['1', '2'], ['3', '4']], (i) => i === 0) + txt(80, 64, '×', 'var(--fg)', 'middle', ';font-size:18px')
    + grid(96, 30, [['5', '6'], ['7', '8']], (i, j) => j === 0) + txt(170, 64, '=', 'var(--fg)', 'middle', ';font-size:18px')
    + grid(186, 30, [['19', '22'], ['43', '50']], (i, j) => i === 0 && j === 0)
    + txt(38, 20, 'A', 'var(--muted)') + txt(128, 20, 'B', 'var(--muted)') + txt(218, 20, 'AB', 'var(--muted)')
    + txt(150, 102, '1 × 5 + 2 × 7 = 19', 'var(--fwd)'));
})();

const expFig = (() => {
  const X = (x) => 150 + x * 40, Y = (y) => 100 - y * 30;
  return fig(200, 'The curve of exp rises from just above zero on the left and grows fast on the right. The curve of ln is its mirror image across the line y equals x.',
    line(10, 100, 290, 100) + line(150, 195, 150, 5)
    + poly(curve((x) => x, -1.6, 3.3).map(([x, y]) => [X(x), Y(y)]), 'var(--border)', ';stroke-dasharray:4 4;stroke-width:1.5')
    + poly(curve(Math.exp, -3.4, 1.15).map(([x, y]) => [X(x), Y(y)]))
    + poly(curve(Math.log, 0.06, 3.4).map(([x, y]) => [X(x), Y(y)]), 'var(--warn)')
    + txt(X(-1.8), Y(Math.exp(-1.8)) - 10, 'exp', 'var(--fwd)') + txt(X(3.1), Y(Math.log(3.1)) + 22, 'ln', 'var(--warn)')
    + dot(X(0), Y(1), 'var(--fwd)', 3.5) + dot(X(1), Y(0), 'var(--warn)', 3.5) + txt(X(0) + 7, Y(1) + 16, '1', 'var(--muted)', 'start')
    + txt(X(1), Y(0) + 18, '1', 'var(--muted)'));
})();

const probFig = fig(90, 'A bar of length 1 cut into three parts, 0.5, 0.3 and 0.2. Together the three parts fill the whole bar.',
  box(10, 20, 140, 36, '0.5', 'var(--fwd)') + box(150, 20, 84, 36, '0.3', 'var(--warn)') + box(234, 20, 56, 36, '0.2', 'var(--ok)')
  + txt(150, 80, '0.5 + 0.3 + 0.2 = 1', 'var(--fg)'));

const meanFig = (() => {
  const X = (v) => 30 + (v - 1) * 34;
  return fig(110, 'The numbers 2, 4, 6 and 8 on a number line, with the mean 5 in the middle. Arrows show the distance of each number from the mean.',
    line(20, 70, 290, 70) + [1, 2, 3, 4, 5, 6, 7, 8].map((v) => line(X(v), 66, X(v), 74) + txt(X(v), 92, String(v))).join('')
    + line(X(5), 20, X(5), 80, 'var(--warn)', ';stroke-dasharray:4 3') + txt(X(5), 14, 'mean', 'var(--warn)')
    + [2, 4, 6, 8].map((v, i) => dot(X(v), 70) + line(X(5), 32 + i * 9, X(v), 32 + i * 9, 'var(--fwd)', ';stroke-width:1.5')).join(''));
})();

const normalFig = (() => {
  const X = (x) => 150 + x * 32, Y = (y) => 150 - y * 250;
  const g = (s) => (x) => Math.exp(-(x * x) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
  return fig(165, 'Two bell curves with the same mean in the middle. The curve with the small standard deviation is narrow and tall. The curve with the large standard deviation is wide and low.',
    line(10, 150, 290, 150)
    + poly(curve(g(0.75), -4.3, 4.3, 90).map(([x, y]) => [X(x), Y(y)]))
    + poly(curve(g(1.7), -4.3, 4.3, 90).map(([x, y]) => [X(x), Y(y)]), 'var(--warn)')
    + line(150, 150, 150, 154) + txt(150, 164, 'mean')
    + txt(X(0.7), Y(g(0.75)(0)) + 14, 'narrow', 'var(--fwd)', 'start') + txt(X(2.6), Y(g(1.7)(2.6)) - 8, 'wide', 'var(--warn)', 'start'));
})();

const contourFig = (() => {
  const rings = [1, 2, 3, 4, 5].map((k) => `<ellipse cx="150" cy="80" rx="${k * 26}" ry="${k * 13}" style="fill:none;stroke:${k === 1 ? 'var(--fwd)' : 'var(--muted)'};stroke-width:1.5"/>`).join('');
  return fig(165, 'A contour map of a valley. Closed rings lie around the lowest point. The rings lie close together where the ground is steep and far apart where it is flat. An arrow crosses the rings at a right angle.',
    rings + txt(150, 85, 'low', 'var(--fwd)') + txt(150, 160, 'high')
    + line(150, 15, 150, 52, 'var(--warn)') + `<polygon points="150,58 146,50 154,50" style="fill:var(--warn)"/>`
    + txt(158, 30, 'steepest way down', 'var(--warn)', 'start'));
})();

const logFig = (() => {
  const X = (v) => 20 + Math.log10(v) * 85;
  const minor = [1, 10, 100].flatMap((d) => [2, 3, 4, 5, 6, 7, 8, 9].map((k) => line(X(d * k), 52, X(d * k), 60, 'var(--muted)', ';stroke-width:1')));
  return fig(110, 'A log axis with the marks 1, 10, 100 and 1,000 at equal distances. Between two of these marks, the small marks for 2 to 9 lie closer together on the right.',
    line(20, 60, 280, 60) + minor.join('')
    + [1, 10, 100, 1000].map((v) => line(X(v), 46, X(v), 66, 'var(--fg)') + txt(X(v), 84, v === 1000 ? '1,000' : String(v), 'var(--fg)')).join('')
    + [0, 1, 2].map((i) => txt((X(10 ** i) + X(10 ** (i + 1))) / 2, 34, '× 10', 'var(--fwd)')).join(''));
})();

// ---- the refreshers -------------------------------------------------------------------------------------------------------
const R = {
  slope: {
    title: 'The slope of a curve',
    source: OPENSTAX('calculus-volume-1', 'Calculus Volume 1, section 3.1'),
    body: `<p>The slope of a straight line tells how far the line goes up for each unit that it goes to the right.</p>
<p>A curve has no single slope, because its steepness changes from point to point. To find the slope at one point, move a very small distance to the right. This small move is a nudge. Then divide the change in height by the size of the nudge.</p>
${slopeFig}
<p>Take the curve <code>y = x²</code> at <code>x = 3</code>, where the height is <code>9</code>. A nudge of <code>0.1</code> gives the height <code>3.1² = 9.61</code>, so the ratio is <code>0.61 / 0.1 = 6.1</code>. A nudge of <code>0.01</code> gives the ratio <code>0.0601 / 0.01 = 6.01</code>.</p>
<p>When the nudge gets smaller, the ratio comes closer to one number. Here that number is <code>6</code>. Mathematicians call it the derivative of the curve at that point.</p>`,
  },
  sum: {
    title: 'The sum sign Σ',
    source: OPENSTAX('calculus-volume-1', 'Calculus Volume 1, section 5.1'),
    body: `<p>The sign Σ is the Greek capital letter sigma. It tells you to add a list of terms. Under the sign is an index with its first value, and above the sign is its last value.</p>
<p>For each value of the index, you write the term that follows the sign. Then you add all the terms.</p>
${sumFig}
<p>Take a list <code>x</code> with three entries, <code>x₀ = 3</code>, <code>x₁ = 1</code> and <code>x₂ = 4</code>. The small number under each <code>x</code> is its position in the list. On this site, positions count from <code>0</code>. The sum of <code>xᵢ</code> for <code>i</code> from <code>0</code> to <code>2</code> is <code>3 + 1 + 4 = 8</code>.</p>
<p>The letter <code>i</code> is only a name for the position. If you write <code>j</code> in its place, you get the same sum.</p>`,
  },
  'dot-product': {
    title: 'Vectors and the dot product',
    source: OPENSTAX('calculus-volume-3', 'Calculus Volume 3, section 2.3'),
    body: `<p>A vector is a list of numbers in a fixed order. The vector <code>a = (2, 3)</code> has two entries. Its first entry is <code>2</code>, and its second entry is <code>3</code>.</p>
<p>The dot product takes two vectors of the same length and gives one number. Multiply the entries that stand in the same position. Then add the products.</p>
${dotFig}
<p>Take <code>a = (2, 3)</code> and <code>b = (4, 1)</code>. The dot product is <code>a · b = 2 × 4 + 3 × 1 = 8 + 3 = 11</code>.</p>
<p>With the sum sign, the dot product is the sum of <code>aᵢ × bᵢ</code> over all positions <code>i</code>. A weighted sum is a dot product. One vector holds the inputs, and the other vector holds the weights.</p>`,
  },
  'matrix-product': {
    title: 'The matrix product',
    source: OPENSTAX('college-algebra-2e', 'College Algebra 2e, section 7.5'),
    body: `<p>A matrix is a grid of numbers in rows and columns. This site counts rows and columns from <code>0</code>.</p>
<p>To multiply a matrix <code>A</code> by a matrix <code>B</code>, take one row of <code>A</code> and one column of <code>B</code>. Their dot product gives one entry of the result. The row and the column tell where the entry goes.</p>
${matFig}
<p>Take <code>A</code> with the rows <code>(1, 2)</code> and <code>(3, 4)</code>, and <code>B</code> with the rows <code>(5, 6)</code> and <code>(7, 8)</code>. Row <code>0</code> of <code>A</code> and column <code>0</code> of <code>B</code> give <code>1 × 5 + 2 × 7 = 19</code>. In the same way, the other three entries are <code>22</code>, <code>43</code> and <code>50</code>.</p>
<p>Each row of <code>A</code> must have as many entries as each column of <code>B</code>. Otherwise the dot product does not exist. The order also matters, because <code>AB</code> and <code>BA</code> are usually different.</p>`,
  },
  'exp-ln': {
    title: 'The functions exp and ln',
    source: OPENSTAX('algebra-and-trigonometry-2e', 'Algebra and Trigonometry 2e, chapter 6'),
    body: `<p>The number <code>e</code> is about <code>2.718</code>. The function exp raises <code>e</code> to a power, so <code>exp(x)</code> is the same as <code>eˣ</code>. For example, <code>exp(0) = 1</code> and <code>exp(1) = e</code>.</p>
<p>The value of <code>exp(x)</code> is always above zero, also for a negative <code>x</code>. It grows fast when <code>x</code> grows.</p>
${expFig}
<p>The function ln reverses exp. The value <code>ln(y)</code> is the power that you must give to <code>e</code> to get <code>y</code>. So <code>ln(e) = 1</code> and <code>ln(1) = 0</code>. The function ln exists only for numbers above zero. It also changes a product into a sum, because <code>ln(a × b) = ln(a) + ln(b)</code>.</p>
<p>At school, log often means the logarithm with base <code>10</code>, so <code>log(100) = 2</code>. On this site, log always means ln, the logarithm with base <code>e</code>. The log function of TorC also uses the base <code>e</code>.</p>`,
  },
  probability: {
    title: 'Probability, and why the shares add to 1',
    source: OPENSTAX('introductory-statistics-2e', 'Introductory Statistics 2e, chapter 3'),
    body: `<p>A probability is a number from <code>0</code> to <code>1</code> that tells how likely an event is. A probability of <code>0</code> means that the event never happens. A probability of <code>1</code> means that it always happens. People often write a probability as a percentage, so <code>0.25</code> is <code>25 %</code>.</p>
<p>A fair die has six faces, and each face has the probability <code>1/6</code>. One face always comes up, and two faces never come up together. So the six probabilities add to <code>6 × 1/6 = 1</code>.</p>
${probFig}
<p>This rule holds for every complete list of outcomes that exclude each other. Their probabilities always add to <code>1</code>. Suppose that a question has three possible answers. If you give two answers <code>0.5</code> and <code>0.3</code>, then the third answer must get <code>0.2</code>.</p>`,
  },
  'mean-variance': {
    title: 'Mean, variance and standard deviation',
    source: OPENSTAX('introductory-statistics-2e', 'Introductory Statistics 2e, chapter 2'),
    body: `<p>The mean of a list of numbers is their sum divided by the count of numbers. For the list <code>2, 4, 6, 8</code>, the mean is <code>20 / 4 = 5</code>.</p>
<p>The variance tells how far the numbers spread around the mean. Subtract the mean from each number, and square each difference. The mean of these squares is the variance.</p>
${meanFig}
<p>Here the differences are <code>−3, −1, 1, 3</code>, and their squares are <code>9, 1, 1, 9</code>. The variance is <code>20 / 4 = 5</code>. The square makes every difference positive, and it makes large differences count much more than small ones.</p>
<p>The standard deviation is the square root of the variance. Here it is <code>√5</code>, which is about <code>2.24</code>. It has the same unit as the numbers. So it tells, in that unit, how far the numbers usually are from the mean.</p>`,
  },
  'normal-distribution': {
    title: 'The normal distribution',
    source: OPENSTAX('introductory-statistics-2e', 'Introductory Statistics 2e, chapter 6'),
    body: `<p>Some random numbers are often near one middle value and rare far from it. The normal distribution describes such numbers. Its curve is a hill with one top. People call it the bell curve.</p>
<p>Two numbers set the curve. The mean sets the position of the top. The standard deviation sets the width. A large standard deviation gives a wide, low hill, and a small one gives a narrow, tall hill.</p>
${normalFig}
<p>The area under the whole curve is <code>1</code>. The area between two values tells how often a random number lands between them. Most of the area lies within two standard deviations of the mean.</p>
<p>A computer can make random numbers that follow a normal distribution with a given mean and standard deviation.</p>`,
  },
  'contour-map': {
    title: 'How to read a contour map',
    source: OPENSTAX('calculus-volume-3', 'Calculus Volume 3, section 4.1'),
    body: `<p>A hiking map shows the ground from above. Each closed line on the map joins points that have the same height. These lines are contour lines.</p>
<p>Where the lines lie close together, the ground is steep. Where they lie far apart, the ground is almost flat. The smallest ring marks the top of a hill or the bottom of a valley.</p>
${contourFig}
<p>A colour map does the same with colours. Each colour stands for one height, and the legend of the map tells which colour is low and which is high.</p>
<p>From any point, the steepest way down crosses the contour lines at a right angle. In this course, the two directions are two numbers that you can change. The height is the result.</p>`,
  },
  'log-axis': {
    title: 'How to read a log axis',
    source: OPENSTAX('algebra-and-trigonometry-2e', 'Algebra and Trigonometry 2e, chapter 6'),
    body: `<p>On an ordinary axis, equal distances mean equal differences. The step from <code>1</code> to <code>2</code> has the same length as the step from <code>101</code> to <code>102</code>.</p>
<p>On a log axis, equal distances mean equal factors. The step from <code>1</code> to <code>10</code> has the same length as the step from <code>10</code> to <code>100</code>.</p>
${logFig}
<p>A log axis helps when a value changes by many factors of ten. On an ordinary axis, all values below <code>10</code> would fill a thin strip next to the values near <code>1,000</code>. On a log axis, each factor of ten gets the same room.</p>
<p>Halfway between <code>1</code> and <code>100</code> on a log axis is <code>10</code>, because <code>1 × 10 = 10</code> and <code>10 × 10 = 100</code>. A straight falling line on a log axis means that the value shrinks by the same factor in each equal step.</p>`,
  },
};

export const REFRESHERS = Object.fromEntries(Object.entries(R).map(([k, r]) => [k, { title: r.title, source: r.source, html: r.body + '\n' + source(r.source) }]));
