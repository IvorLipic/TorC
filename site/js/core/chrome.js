// core/chrome.js — page furniture shared by every page: the site header (brand + Course link) and URL helpers.
// All URLs are derived from where this module lives (import.meta.url), so the site works under any prefix
// (/ on localhost, /TorC/ on GitHub Pages) with no <base> element and no root-absolute path in the source.
import { escapeHtml } from './dom.js';

export const SITE = new URL('../../', import.meta.url);     // site root: this file is site/js/core/chrome.js
export const siteUrl = (rel) => new URL(rel, SITE).href;
export const REPO_URL = 'https://github.com/IvorLipic/TorC';

// current: 'home' | 'lesson' (marks the Course link when you are on the course map)
export function headerHtml(current = 'lesson') {
  return '<a class="skip" href="#top">Skip to content</a>'
    + '<header class="site"><nav aria-label="Site">'
    + `<a class="brand" href="${escapeHtml(siteUrl(''))}">TorC Learn</a>`
    + `<a class="nav" href="${escapeHtml(siteUrl(''))}"${current === 'home' ? ' aria-current="page"' : ''}>Course</a>`
    + `<a class="nav" href="${REPO_URL}">GitHub</a>`
    + '</nav></header>';
}
