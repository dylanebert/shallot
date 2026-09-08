import { lockup, toSvg } from "./brand/mark";
import { CSS_PALETTE, FONTS, nav, STYLE, THEME_SCRIPT, TOGGLE } from "./brand/theme";
import type { DemoEntry } from "./roster";

// The site home at /shallot/: the lockup splashing in, the one-line promise, quick start, the
// demos, and the build label in the foot. The nav is the one both pages share. The demos themselves are built
// separately; `clientScript` is the bundled `site/brand/client.ts`.

export function siteIndex(
    demos: DemoEntry[],
    version: string,
    ref: string,
    mode: "prod" | "staging",
    clientScript: string = "",
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
                `<li><a class="play" href="./${d.slug}/">${d.title}</a><a class="code" href="${codeUrl(d.slug)}">code</a></li>`,
        )
        .join("\n");
    const label = mode === "staging" ? `staging · ${ref}` : `v${version} · ${ref}`;
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>shallot</title>
<meta name="description" content="WebGPU game engine. Fast by default, instant iteration, runs anywhere.">
<link rel="icon" href="./brand/mark-32.png">
${FONTS}
${THEME_SCRIPT}
<style>${STYLE}
header { display: grid; gap: 18px; padding: 24px 0 8px; }
header p { font-size: 17px; text-wrap: balance; }
.demos { list-style: none; }
.demos li { display: flex; align-items: center; border-bottom: 1px solid var(--line); font-family: "JetBrains Mono", monospace; font-size: 13px; }
.demos li:last-child { border-bottom: none; }
.demos .play { flex: 1; padding: 10px 0; }
.demos .code { padding: 10px 0 10px 24px; color: var(--muted); }
.demos .code:hover { color: var(--gold); }
.needs { font-size: 13px; }
.steps { display: grid; gap: 8px; }
.steps .c { color: var(--muted); }
.steps a { color: inherit; text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 3px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
${TOGGLE}
<main>
${nav("home")}
<header>
<div data-splash-svg data-scale="5">${toSvg(lockup(), CSS_PALETTE, 5)}</div>
<p>WebGPU game engine. Fast by default, instant iteration, runs anywhere.</p>
</header>

<section>
<h2>quick start</h2>
<div class="steps">
<pre class="block"><span class="c"># install bun</span>
curl -fsSL https://<a href="https://bun.sh">bun.sh</a>/install | bash</pre>
<pre class="block"><span class="c"># new project</span>
bun create shallot my-game
cd my-game
bun install</pre>
<pre class="block"><span class="c"># run it</span>
bunx shallot dev</pre>
</div>
</section>

<section>
<h2>demos</h2>
<ul class="demos">
${rows}
</ul>
<p class="muted needs" data-webgpu-note hidden>This browser has no WebGPU. The demos need Chrome, Edge, or Safari 26+ on desktop.</p>
</section>

<footer>${label}</footer>
</main>
<script type="module">${clientScript}</script>
</body>
</html>
`;
}
