import { lockup, toSvg } from "./brand/mark";
import { CSS_PALETTE, FONTS, STYLE, THEME_SCRIPT, TOGGLE } from "./brand/theme";
import type { DemoEntry } from "./roster";

// The site home at /shallot/: lockup, the three promises, quick start, the demo roster, and
// links out. Static HTML plus the theme toggle; the demos themselves are built separately.

export function siteIndex(
    demos: DemoEntry[],
    version: string,
    ref: string,
    mode: "prod" | "staging",
): string {
    // staging labels by ref, never by version tag — a staging build routinely runs ahead of the
    // last release, so `v${version}` may name a GitHub tag that doesn't exist yet.
    const codeUrl = (slug: string) =>
        mode === "staging"
            ? `https://github.com/dylanebert/shallot/tree/${ref}/examples/showcase/${slug}`
            : `https://github.com/dylanebert/shallot/tree/v${version}/examples/showcase/${slug}`;
    const rows = demos
        .map(
            (d) =>
                `<tr><td><a href="./${d.slug}/">${d.title}</a></td><td><a href="${codeUrl(d.slug)}">code</a></td></tr>`,
        )
        .join("\n");
    const label = mode === "staging" ? `staging · ${ref}` : `v${version} · ${ref}`;
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>shallot</title>
<link rel="icon" href="./brand/mark-32.png">
${FONTS}
${THEME_SCRIPT}
<style>${STYLE}
header { display: grid; gap: 18px; padding: 24px 0 8px; }
header p { font-size: 17px; max-width: 48ch; }
table { border-collapse: collapse; font-family: "JetBrains Mono", monospace; font-size: 13px; width: 100%; max-width: 480px; }
td { padding: 8px 0; border-bottom: 1px solid var(--line); }
td:last-child { text-align: right; color: var(--muted); }
tr:last-child td { border-bottom: none; }
.meta { font-family: "JetBrains Mono", monospace; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
${TOGGLE}
<main>
<nav><a href="./" style="color:var(--gold)">shallot</a><a href="#demos">demos</a><a href="./brand/">brand</a><span class="sp"></span><a href="https://github.com/dylanebert/shallot">github</a><a href="https://www.npmjs.com/package/@dylanebert/shallot">npm</a></nav>

<header>
${toSvg(lockup(), CSS_PALETTE, 5)}
<p>WebGPU game engine. Fast by default, instant iteration, runs anywhere.</p>
</header>

<section>
<h2>quick start</h2>
<pre class="block">bun create shallot my-game
cd my-game
bun install
bunx shallot dev</pre>
<p class="muted">A project is plain data plus code: a manifest, a scene file, and TypeScript plugins you edit in your IDE. The source is the reference; every public export carries its contract.</p>
</section>

<section id="demos">
<h2>demos</h2>
<table><tbody>
${rows}
</tbody></table>
<p class="meta">${label} · WebGPU required: Chrome, Edge, or Safari 26+ on desktop.</p>
</section>
</main>
</body>
</html>
`;
}
