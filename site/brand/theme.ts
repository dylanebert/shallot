import { DARK, LIGHT, type Palette } from "./mark";

// Shared page chrome for the site's own pages (home, brand): tokens, the theme toggle, and the
// half-block `pre` rules. Dark is the brand; light is the same assets on paper behind a toggle.

/** A palette whose values are CSS custom properties, so inline SVG follows the page theme. */
export const CSS_PALETTE: Palette = {
    gold: "var(--gold)",
    dim: "var(--dim)",
    ink: "var(--ink)",
    bg: "var(--bg)",
};

export const FONTS =
    '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap">';

export const STYLE = `
:root { --bg: ${DARK.bg}; --bg2: #1c1917; --ink: ${DARK.ink}; --muted: #a08c78; --gold: ${DARK.gold}; --dim: ${DARK.dim}; --line: #2a2420; }
:root[data-theme="light"] { --bg: ${LIGHT.bg}; --bg2: #efe9df; --ink: ${LIGHT.ink}; --muted: #6e655c; --gold: ${LIGHT.gold}; --dim: ${LIGHT.dim}; --line: #e3dbcf; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-text-size-adjust: 100%; }
body { background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 15px; line-height: 1.55; }
main { max-width: 880px; margin: 0 auto; padding: 40px 24px 96px; display: grid; gap: 40px; }
a { color: inherit; text-decoration: none; }
a:hover { color: var(--gold); }
h2 { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; font-weight: 400; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
section { display: grid; gap: 14px; }
nav { display: flex; gap: 22px; align-items: center; font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 13px; }
nav .sp { flex: 1; }
pre, code { font-family: "JetBrains Mono", ui-monospace, monospace; }
pre { font-size: 13px; line-height: 1.2; white-space: pre; overflow-x: auto; }
pre.blocks { color: var(--gold); }
code { font-size: 13px; }
.block { background: var(--bg2); padding: 20px 24px; border-radius: 2px; }
.muted { color: var(--muted); }
svg { display: block; }
.toggle { position: fixed; top: 14px; right: 14px; width: 30px; height: 30px; border: 1px solid var(--line); background: var(--bg); color: var(--muted); border-radius: 2px; cursor: pointer; display: grid; place-items: center; font-family: system-ui, sans-serif; font-size: 14px; line-height: 1; }
.toggle:hover { color: var(--gold); }
.toggle:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
@media (max-width: 480px) { main { padding: 28px 16px 64px; } nav { flex-wrap: wrap; gap: 14px; } }
`;

/** Applies the saved theme before first paint; the toggle flips and stores it. */
export const THEME_SCRIPT = `<script>(function(){try{var t=localStorage.getItem("shallot-theme");if(t==="light")document.documentElement.dataset.theme="light";}catch(e){}})();</script>`;

export const TOGGLE = `<button class="toggle" type="button" aria-label="Toggle light and dark" data-toggle>◐</button>
<script>document.querySelector("[data-toggle]").addEventListener("click",function(){var r=document.documentElement;var next=r.dataset.theme==="light"?"":"light";if(next)r.dataset.theme=next;else delete r.dataset.theme;try{localStorage.setItem("shallot-theme",next||"dark");}catch(e){}});</script>`;
