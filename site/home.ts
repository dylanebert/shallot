import { lockup, toSvg } from "./brand/mark";
import { AGENTS_LINK, CSS_PALETTE, FONTS, STYLE, THEME_SCRIPT, TOGGLE, top } from "./brand/theme";
import type { DemoEntry } from "./roster";

// The site home at /shallot/: the lockup splashing in, the one-line promise, quick start, the
// demos, and the foot: build label, then the brand link, low-key. The top row is only the GitHub
// and npm icons; brand is a detail that doesn't earn a place there. The demos themselves are built
// separately; `clientScript` is the bundled `site/brand/client.ts`, `rum` the Datadog init
// snippet (page views only, no frame sampler).

export function siteIndex(
    demos: DemoEntry[],
    version: string,
    ref: string,
    mode: "prod" | "staging",
    clientScript: string = "",
    rum: string = "",
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
                `<li><a class="play" href="./${d.slug}/">${d.slug}</a><a class="code" href="${codeUrl(d.slug)}" target="_blank" rel="noopener">code</a></li>`,
        )
        .join("\n");
    const out = (href: string, text: string) =>
        `<a href="${href}" target="_blank" rel="noopener">${text}</a>`;
    const commit = out(`https://github.com/dylanebert/shallot/commit/${ref}`, ref);
    const label =
        mode === "staging"
            ? `staging · ${commit}`
            : `${out(`https://github.com/dylanebert/shallot/releases/tag/v${version}`, `v${version}`)} · ${commit}`;
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>shallot</title>
<meta name="description" content="webgpu game engine. fast by default, instant iteration, runs where webgpu does.">
<link rel="icon" href="./brand/mark-32.png">
${AGENTS_LINK("home")}
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
.warn { color: var(--gold); font-size: 14px; }
.warn a { text-decoration: underline; text-decoration-color: color-mix(in srgb, var(--gold) 45%, transparent); text-underline-offset: 3px; }
.steps { display: grid; gap: 8px; }
.steps .c { color: var(--muted); }
.steps a { color: inherit; text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 3px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
${TOGGLE}
<main>
${top("home")}
<header>
<div data-splash-svg data-scale="5">${toSvg(lockup(), CSS_PALETTE, 5)}</div>
<p>webgpu game engine. fast by default, instant iteration, runs where webgpu does.</p>
</header>

<section>
<h2>quick start</h2>
<div class="steps">
<pre class="block"><span class="c"># install bun</span>
curl -fsSL https://<a href="https://bun.sh" target="_blank" rel="noopener">bun.sh</a>/install | bash</pre>
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
<p class="warn" data-webgpu-note hidden><a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">WebGPU not supported.</a> Demos will not run.</p>
<ul class="demos">
${rows}
</ul>
</section>

<footer><span>${label}</span><span class="sp"></span><a href="./brand/">brand</a></footer>
</main>
<script type="module">${clientScript}</script>
${rum}</body>
</html>
`;
}

/**
 * `/llms.txt`, the plain-text entry for agents: what shallot is and the raw files to read, pinned to
 * the built version (staging: the built ref). Nothing here is rendered; the docs live in the repo.
 */
export function llmsTxt(version: string, ref: string, mode: "prod" | "staging"): string {
    const at = mode === "staging" ? ref : `v${version}`;
    const raw = (path: string) =>
        `https://raw.githubusercontent.com/dylanebert/shallot/${at}/${path}`;
    return `# shallot

> webgpu game engine. fast by default, instant iteration, runs where webgpu does.

The source is the reference: every public export carries a JSDoc contract, and there is no docs site to drift from it. Two files carry the consumer surface. Read the first before writing a project; grep the second for the problem you have.

## Read

- [Consumer contract](${raw("packages/shallot/AGENTS.md")}): commands, the ECS and plugin conventions, the GPU, render, physics and verify rules. Ships in the npm package as AGENTS.md.
- [Examples index](${raw("examples/AGENTS.md")}): one line per recipe and showcase project, with the concept each teaches.
- [README](${raw("README.md")}): quick start, live demos, building from source.

## Start

\`\`\`
bun create shallot my-game
cd my-game
bun install
bunx shallot dev
\`\`\`

\`bunx shallot verify\` boots the project in a headless browser and exits 0 or nonzero, so a change can be checked without a person watching.

## Links

- Source: https://github.com/dylanebert/shallot
- Package: https://www.npmjs.com/package/@dylanebert/shallot
- Demos: https://dylanebert.com/shallot/
- Brand: https://dylanebert.com/shallot/brand/
`;
}
