// content/refreshers.js — the maths refreshers (brief §8, R9): short, collapsible explanations that a lesson opens just in time with
// refresher(key) from core/blocks.js. Each is { title, html }: title is plain text for the <summary>, html is the body (100-200
// words, ideally with a picture). The html is static markup, so it must not call KaTeX or T(); write numbers as words or inside
// <code>. Only sample entries live here for now; the content agent adds the rest (see the list in docs/pedagogy-review.md §8).
export const REFRESHERS = {
  slope: {
    title: 'The slope of a curve',
    html: `<p>The slope of a straight line tells how far the line goes up for each unit that it goes to the right. A line that goes down has a negative slope.</p>
<p>A curve has no single slope. To find its slope at one point, move a very small distance to the right. Then divide the change in height by that distance. This ratio is the slope of the curve at that point. Mathematicians call it the derivative.</p>
<p>When the distance gets smaller, the ratio comes closer to the true slope.</p>`,
  },
  sum: {
    title: 'The sum sign Σ',
    html: `<p>The sign Σ is the Greek capital letter sigma. It tells you to add a list of terms. Under the sign is an index with its first value, and above the sign is its last value.</p>
<p>For each value of the index, you write the term that follows the sign. Then you add all the terms. For an index <code>i</code> from <code>0</code> to <code>2</code>, the sum of <code>x[i]</code> is <code>x[0] + x[1] + x[2]</code>.</p>`,
  },
};
