// core/popover.js — the glossary definition popover for term() links (core/blocks.js, R9). lesson-boot.js loads site/glossary.json
// ({schema: "torc-glossary/1", terms: {<key>: {term, forms, short, lesson}}}) and calls wireTerms(main, glossary).
//
// Each <a class="term" data-term="key"> keeps its link to ../glossary/#key and gets a tooltip (role="tooltip", referenced by
// aria-describedby, so a screen reader reads the definition when the link gets focus). It opens on hover (and stays while the
// pointer is over the link or the tooltip), on keyboard focus, and on the first tap of a touch screen (the second tap follows
// the link). Escape closes it (WCAG 1.4.13). The tooltip holds text only, never a control. It is placed under the term and kept
// inside the viewport, so it works at 390 px width. An unknown key leaves the plain link (the tests reject unknown keys).

let count = 0;

export function wireTerms(root, glossary, doc = document) {
  const terms = glossary?.terms ?? {};
  const links = [...root.querySelectorAll('a.term[data-term]')].filter((a) => Object.hasOwn(terms, a.dataset.term));
  if (!links.length) return { destroy() {} };
  const ac = new AbortController(), on = (el, ev, fn, opts = {}) => el.addEventListener(ev, fn, { ...opts, signal: ac.signal });
  const tips = [];
  let openLink = null, hideTimer = null;

  const place = (a, tip) => {
    const r = a.getBoundingClientRect(), vw = doc.documentElement.clientWidth;
    tip.style.maxWidth = `${Math.min(320, vw - 16)}px`;
    const w = tip.offsetWidth, left = Math.max(8, Math.min(r.left + window.scrollX, window.scrollX + vw - w - 8));
    tip.style.left = `${left}px`;
    tip.style.top = `${r.bottom + window.scrollY + 6}px`;
  };
  const show = (a) => {
    clearTimeout(hideTimer);
    if (openLink && openLink !== a) hide(openLink);
    const tip = doc.getElementById(a.getAttribute('aria-describedby'));
    tip.hidden = false;
    place(a, tip);
    a.classList.add('open');
    openLink = a;
  };
  function hide(a) {
    const tip = doc.getElementById(a.getAttribute('aria-describedby'));
    if (tip) tip.hidden = true;
    a.classList.remove('open');
    if (openLink === a) openLink = null;
  }
  const hideSoon = (a) => { clearTimeout(hideTimer); hideTimer = setTimeout(() => hide(a), 200); };

  for (const a of links) {
    const t = terms[a.dataset.term];
    const tip = doc.createElement('span');
    tip.className = 'term-tip';
    tip.id = `term-tip-${++count}`;
    tip.setAttribute('role', 'tooltip');
    tip.hidden = true;
    const head = doc.createElement('strong');
    head.textContent = t.term.charAt(0).toUpperCase() + t.term.slice(1) + '.';
    tip.append(head, ' ', t.short);
    doc.body.append(tip);
    tips.push(tip);
    a.setAttribute('aria-describedby', tip.id);
    let touched = false;
    on(a, 'pointerenter', (e) => { if (e.pointerType === 'mouse') show(a); });
    on(a, 'pointerleave', (e) => { if (e.pointerType === 'mouse') hideSoon(a); });
    on(tip, 'pointerenter', () => clearTimeout(hideTimer));
    on(tip, 'pointerleave', () => hideSoon(a));
    on(a, 'focus', () => { if (!touched) show(a); });   // a tap also focuses the link; the click handler decides then
    on(a, 'blur', () => { if (!touched) hide(a); });
    on(a, 'pointerdown', (e) => { touched = e.pointerType !== 'mouse'; });
    on(a, 'click', (e) => {
      if (!touched) return;               // mouse and keyboard follow the link at once
      touched = false;
      if (openLink !== a) { e.preventDefault(); show(a); }   // first tap: the definition; second tap: the glossary page
    });
  }
  on(doc, 'keydown', (e) => { if (e.key === 'Escape' && openLink) hide(openLink); });
  on(doc, 'pointerdown', (e) => { if (openLink && !openLink.contains(e.target) && !e.target.closest?.('.term-tip')) hide(openLink); });
  on(window, 'resize', () => { if (openLink) place(openLink, doc.getElementById(openLink.getAttribute('aria-describedby'))); });
  return { destroy() { ac.abort(); tips.forEach((t) => t.remove()); } };
}
