// core/quiz.js — behaviour of the quiz and explain sections that core/sections.js renders (R10, brief §9 "Checks").
// lesson-boot.js calls wireQuizzes(main, {lessonId, store}) and wireExplains(main, {lessonId, store}) after rendering.
//
// quiz     The learner answers first: a radio choice or a number, then Check (a real <button type="submit">, so Enter works).
//          The feedback region (role="status", aria-live="polite") says whether the answer is right; for a choice it adds the
//          feedback of the option that was chosen (a <template data-fb="i"> rendered in Node). The reveal opens after the first
//          attempt and stays open. The learner can change the answer and check again at any time. Every attempt is recorded in the
//          store as {tries, correct}; a page reload shows the reveal again for a quiz that was tried before.
// explain  The learner writes a sentence, then "Compare with a good answer" shows the model and stores explained[id]. The sentence
//          itself is never stored.
// Everything works with the keyboard only and with storage blocked (the store keeps the record in memory, core/store.js).
// parseAnswer / judgeNumeric / judgeChoice are pure and unit-tested in Node.

// "1.92", " -14.4 ", "−14.4" (U+2212), "26,506" (thousands), "1,92" (a decimal comma: one comma, not followed by exactly three
// digits) -> number, or null when it is not a number.
export function parseAnswer(s) {
  let t = String(s ?? '').trim().replace(/−/g, '-').replace(/\s+/g, '');
  if (!t) return null;
  if (/^-?\d+,(\d{1,2}|\d{4,})$/.test(t)) t = t.replace(',', '.');
  t = t.replace(/,(?=\d{3}(\D|$))/g, '');                       // 26,506 -> 26506
  if (!/^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(t)) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

// |value - answer| <= tolerance, with a little room for the float32 values of the traces
export const judgeNumeric = (answer, tolerance, value) => value !== null && Math.abs(value - answer) <= tolerance * (1 + 1e-6);
export const judgeChoice = (correctIndex, chosen) => chosen !== null && Number(chosen) === Number(correctIndex);

// Sets the text of a live region so that the same message is announced again on a second attempt.
function announce(region, html) {
  region.innerHTML = '';
  setTimeout(() => { region.innerHTML = html; }, 30);
}

const msg = (cls, text) => `<p class="verdict ${cls}"><strong>${text}</strong></p>`;

export function wireQuizzes(root, { lessonId, store }) {
  for (const sec of root.querySelectorAll('section.quiz[data-quiz]')) {
    const id = sec.dataset.quiz, type = sec.dataset.type;
    const form = sec.querySelector('form'), fb = sec.querySelector('.feedback'), reveal = sec.querySelector('.reveal');
    const open = () => { reveal.hidden = false; sec.classList.add('tried'); };
    try { if (store.quiz(lessonId, id).tries > 0) open(); } catch { /* the page works without the record */ }
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      let correct, extra = '';
      if (type === 'numeric') {
        const input = form.querySelector('input');
        const v = parseAnswer(input.value);
        if (v === null) { announce(fb, msg('empty', 'Type a number first.')); input.focus(); return; }
        const answer = Number(sec.dataset.answer);
        correct = judgeNumeric(answer, Number(sec.dataset.tolerance), v);
        if (!correct) extra = `<p>${v > answer ? 'Your answer is too high.' : 'Your answer is too low.'}</p>`;
        input.setAttribute('aria-invalid', String(!correct));
      } else {
        const chosen = form.querySelector('input[type="radio"]:checked');
        if (!chosen) { announce(fb, msg('empty', 'Choose an answer first.')); form.querySelector('input[type="radio"]')?.focus(); return; }
        correct = judgeChoice(sec.dataset.correct, chosen.value);
        extra = sec.querySelector(`template[data-fb="${CSS.escape(chosen.value)}"]`)?.innerHTML ?? '';
        for (const o of sec.querySelectorAll('.option')) o.classList.remove('chosen', 'right', 'wrong');
        chosen.closest('.option')?.classList.add('chosen', correct ? 'right' : 'wrong');
      }
      try { store.recordQuiz(lessonId, id, correct); } catch { /* storage problems never block the quiz */ }
      sec.dataset.state = correct ? 'right' : 'wrong';
      const tail = reveal.hidden ? '<p>The explanation is now open below.</p>' : '';
      announce(fb, (correct ? msg('right', 'Correct.') : msg('wrong', 'Not quite. You can try again.')) + extra + tail);
      open();
    });
  }
}

export function wireExplains(root, { lessonId, store }) {
  for (const sec of root.querySelectorAll('section.explain[data-explain]')) {
    const id = sec.dataset.explain, box = sec.querySelector('textarea'), fb = sec.querySelector('.feedback'), model = sec.querySelector('.model');
    sec.querySelector('[data-explain-compare]').addEventListener('click', () => {
      if (!box.value.trim()) { announce(fb, msg('empty', 'Write your sentence first.')); box.focus(); return; }
      model.hidden = false;
      announce(fb, '<p>Compare your sentence with the answer below.</p>');
      try { store.setExplained(lessonId, id); } catch { /* in memory only */ }
    });
  }
}
