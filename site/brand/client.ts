import { END_TICK, runSplash, splashFrame, toCells, toHtml, toSvg } from "./mark";

// Browser entry for the site pages. Splashes the lockup in: `[data-splash]` as half-block text,
// `[data-splash-svg]` as pixel squares. Click replays. Reads the theme tokens off the root so the
// toggle recolors a resting frame. Shows the WebGPU note only where WebGPU is missing.

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function palette() {
    const style = getComputedStyle(document.documentElement);
    const read = (name: string) => style.getPropertyValue(name).trim();
    return { gold: read("--gold"), dim: read("--dim"), ink: read("--ink"), bg: read("--bg") };
}
const vars = { gold: "var(--gold)", dim: "var(--dim)", ink: "var(--ink)", bg: "var(--bg)" };

const replays: (() => void)[] = [];
const rests: (() => void)[] = [];

for (const el of document.querySelectorAll<HTMLElement>("[data-splash]")) {
    const render = (grid: ReturnType<typeof splashFrame>) => toHtml(toCells(grid), palette());
    const replay = runSplash(el, render, reduced);
    el.addEventListener("click", replay);
    replays.push(replay);
    rests.push(() => {
        el.innerHTML = render(splashFrame(END_TICK + 1));
    });
}

for (const el of document.querySelectorAll<HTMLElement>("[data-splash-svg]")) {
    const scale = Number(el.dataset.scale ?? "4");
    const replay = runSplash(el, (grid) => toSvg(grid, vars, scale), reduced);
    el.addEventListener("click", replay);
    replays.push(replay);
}

document.querySelector("[data-toggle]")?.addEventListener("click", () => {
    for (const rest of rests) rest();
});

const note = document.querySelector<HTMLElement>("[data-webgpu-note]");
if (note && !("gpu" in navigator)) note.hidden = false;
