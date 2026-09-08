import { DARK, LIGHT, type Palette } from "./mark";

// Shared page chrome for the site's own pages (home, brand): tokens, the theme toggle, the outbound
// links, and the half-block `pre` rules. Dark is the brand; light is the same assets on paper behind
// a toggle. The top row holds only the outbound icons (and, on a subpage, the back arrow): brand is
// a detail reached from the foot of home, not a top-level destination.

/** A palette whose values are CSS custom properties, so inline SVG follows the page theme. */
export const CSS_PALETTE: Palette = {
    gold: "var(--gold)",
    dim: "var(--dim)",
    ink: "var(--ink)",
    bg: "var(--bg)",
};

/** Head link to the agents entry, invisible on the page; `/llms.txt` is found by path. */
export const AGENTS_LINK = (here: "home" | "brand") =>
    `<link rel="alternate" type="text/plain" href="${here === "home" ? "./" : "../"}llms.txt" title="llms.txt">`;

export const FONTS =
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">';

export const STYLE = `
:root { --bg: ${DARK.bg}; --bg2: #1c1917; --ink: ${DARK.ink}; --muted: #a08c78; --gold: ${DARK.gold}; --dim: ${DARK.dim}; --line: #2a2420; }
:root[data-theme="light"] { --bg: ${LIGHT.bg}; --bg2: #efe9df; --ink: ${LIGHT.ink}; --muted: #6e655c; --gold: ${LIGHT.gold}; --dim: ${LIGHT.dim}; --line: #e3dbcf; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-text-size-adjust: 100%; scrollbar-gutter: stable; }
body { background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 15px; line-height: 1.55; }
main { max-width: 880px; margin: 0 auto; padding: 40px 24px 96px; display: grid; gap: 40px; }
a { color: inherit; text-decoration: none; }
a:hover { color: var(--gold); }
h2 { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; font-weight: 400; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
section { display: grid; gap: 14px; }
.top { display: flex; gap: 18px; align-items: center; color: var(--muted); padding-right: 44px; }
.top .sp { flex: 1; }
.top a { display: inline-flex; align-items: center; }
.top svg { width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 1.75; stroke-linecap: round; stroke-linejoin: round; }
footer { display: grid; gap: 6px; font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; color: var(--muted); }
pre, code { font-family: "JetBrains Mono", ui-monospace, monospace; }
pre { font-size: 13px; line-height: 1.2; white-space: pre; overflow-x: auto; }
pre.blocks { color: var(--gold); }
code { font-size: 13px; }
.block { background: var(--bg2); padding: 20px 24px; border-radius: 2px; }
.muted { color: var(--muted); }
svg { display: block; max-width: 100%; height: auto; }
canvas { display: block; max-width: 100%; height: auto; }
.toggle { position: fixed; top: 14px; right: 14px; width: 30px; height: 30px; border: 1px solid var(--line); background: var(--bg); color: var(--muted); border-radius: 2px; cursor: pointer; display: grid; place-items: center; font-family: system-ui, sans-serif; font-size: 14px; line-height: 1; }
.toggle:hover { color: var(--gold); }
.toggle:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
@media (max-width: 480px) { main { padding: 28px 16px 64px; } }
`;

/** Applies the saved theme before first paint; the toggle flips and stores it. */
export const THEME_SCRIPT = `<script>(function(){try{var t=localStorage.getItem("shallot-theme");if(t==="light")document.documentElement.dataset.theme="light";}catch(e){}})();</script>`;

export const TOGGLE = `<button class="toggle" type="button" aria-label="Toggle light and dark" data-toggle>◐</button>
<script>document.querySelector("[data-toggle]").addEventListener("click",function(){var r=document.documentElement;var next=r.dataset.theme==="light"?"":"light";if(next)r.dataset.theme=next;else delete r.dataset.theme;try{localStorage.setItem("shallot-theme",next||"dark");}catch(e){}});</script>`;

const GITHUB =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>';
const PACKAGE =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="M3.3 7 12 12l8.7-5"/><path d="M12 22V12"/></svg>';

const ARROW =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>';

const LINKS = `<a href="https://github.com/dylanebert/shallot" target="_blank" rel="noopener" aria-label="GitHub">${GITHUB}</a><a href="https://www.npmjs.com/package/@dylanebert/shallot" target="_blank" rel="noopener" aria-label="npm">${PACKAGE}</a>`;

/** The top row: GitHub and npm icons on the right; on a subpage, a back arrow to home on the left. */
export function top(here: "home" | "brand"): string {
    const back = here === "home" ? "" : `<a href="../" aria-label="Back to shallot">${ARROW}</a>`;
    return `<div class="top">${back}<span class="sp"></span>${LINKS}</div>`;
}
