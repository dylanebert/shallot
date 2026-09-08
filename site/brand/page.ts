import { DARK, fromBlocks, LIGHT, lockup, MARK, toCells, toHtml, toSvg, word } from "./mark";
import { CSS_PALETTE, FONTS, nav, STYLE, THEME_SCRIPT, TOGGLE } from "./theme";

// The brand page at /shallot/brand/: the assets, shown plainly, and their downloads. Labels only;
// the sheet that argued for these choices is not the page. Inline SVG uses CSS-variable fills so
// the toggle flips every asset; downloads carry real hex per theme.

/** Download files the page links to, relative to `brand/`. */
export const DOWNLOADS = [
    ["mark.svg", "mark, svg"],
    ["mark.png", "mark, png 8×"],
    ["mark-16.png", "favicon 16"],
    ["mark-32.png", "favicon 32"],
    ["lockup-dark.svg", "lockup on dark, svg"],
    ["lockup-light.svg", "lockup on light, svg"],
    ["lockup-dark.png", "lockup on dark, png 4×"],
    ["lockup-light.png", "lockup on light, png 4×"],
    ["mark.txt", "half blocks, text"],
    ["mark.ts", "code that prints it"],
] as const;

const svg = (grid: ReturnType<typeof fromBlocks>, scale: number) => toSvg(grid, CSS_PALETTE, scale);

const swatch = (name: string, dark: string, light: string) =>
    `<div class="sw"><i style="--d:${dark};--l:${light}"></i><span>${name}</span><span class="muted">${dark} · ${light}</span></div>`;

export function brandPage(clientScript: string): string {
    const mark = fromBlocks(MARK.m);
    const lock = lockup();
    const lockText = toHtml(toCells(lock), CSS_PALETTE);
    const downloads = DOWNLOADS.map(
        ([file, label]) => `<a href="./${file}" download>${label}</a>`,
    ).join("");
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>shallot brand</title>
<link rel="icon" href="./mark-32.png">
${FONTS}
${THEME_SCRIPT}
<style>${STYLE}
.lockup { padding: 56px 24px; display: grid; place-items: center; cursor: pointer; }
.marks { display: flex; gap: 40px; align-items: flex-end; flex-wrap: wrap; }
.marks div { display: grid; gap: 8px; justify-items: center; font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted); }
.splash { padding: 40px 24px; display: grid; place-items: center; cursor: pointer; }
.splash pre { font-size: 15px; }
.sws { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 2px; }
.sw { background: var(--bg2); padding: 12px; display: grid; gap: 4px; font-family: "JetBrains Mono", monospace; font-size: 11px; }
.sw i { display: block; height: 40px; background: var(--d); border-radius: 2px; margin-bottom: 4px; }
:root[data-theme="light"] .sw i { background: var(--l); }
.type { display: grid; gap: 8px; }
.type .mono { font-family: "JetBrains Mono", monospace; font-size: 24px; font-weight: 700; }
.type .sans { font-size: 17px; max-width: 60ch; }
.dl { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 2px; font-family: "JetBrains Mono", monospace; font-size: 13px; }
.dl a { background: var(--bg2); padding: 12px 14px; }
</style>
</head>
<body>
${TOGGLE}
<main>
${nav("brand")}
<section>
<div class="lockup block"><div data-splash-svg data-scale="6">${svg(lock, 6)}</div></div>
</section>

<section>
<h2>mark</h2>
<div class="marks block">
<div>${svg(fromBlocks(MARK.s), 4)}<span>10 × 12</span></div>
<div>${svg(mark, 4)}<span>12 × 14</span></div>
<div>${svg(fromBlocks(MARK.l), 4)}<span>14 × 16</span></div>
<div>${svg(mark, 1)}<span>1×</span></div>
<div>${svg(mark, 2)}<span>2×</span></div>
</div>
</section>

<section>
<h2>wordmark</h2>
<div class="block">${svg(word(), 6)}</div>
</section>

<section>
<h2>terminal splash</h2>
<div class="splash block"><pre data-splash></pre></div>
</section>

<section>
<h2>terminal</h2>
<pre class="block">${lockText}</pre>
</section>

<section>
<h2>color</h2>
<div class="sws">
${swatch("ground", DARK.bg, LIGHT.bg)}
${swatch("ink", DARK.ink, LIGHT.ink)}
${swatch("gold", DARK.gold, LIGHT.gold)}
${swatch("gold dim", DARK.dim, LIGHT.dim)}
${swatch("muted", "#a08c78", "#6e655c")}
</div>
</section>

<section>
<h2>type</h2>
<div class="type block">
<div class="mono">JetBrains Mono</div>
<div class="sans">IBM Plex Sans for running text. JetBrains Mono for headings, code, and anything the terminal prints. The wordmark is a bitmap, not a font.</div>
</div>
</section>

<section>
<h2>download</h2>
<div class="dl">${downloads}</div>
</section>
</main>
<script type="module">${clientScript}</script>
</body>
</html>
`;
}
