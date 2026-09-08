import { END_TICK, splashFrame, TICK_MS, toCells, toHtml } from "./mark";

// Browser entry for the brand page: runs the splash into every `[data-splash]` pre, replays on
// click, and follows the page theme by reading the tokens off the root element.

function palette() {
    const style = getComputedStyle(document.documentElement);
    const read = (name: string) => style.getPropertyValue(name).trim();
    return { gold: read("--gold"), dim: read("--dim"), ink: read("--ink"), bg: read("--bg") };
}

for (const pre of document.querySelectorAll<HTMLPreElement>("[data-splash]")) {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let start = 0;
    let last = -1;
    let raf = 0;
    const frame = () => {
        const tick = reduced ? END_TICK + 1 : Math.floor((performance.now() - start) / TICK_MS);
        if (tick !== last) {
            last = tick;
            pre.innerHTML = toHtml(toCells(splashFrame(Math.min(tick, END_TICK + 1))), palette());
        }
        if (tick <= END_TICK) raf = requestAnimationFrame(frame);
    };
    const run = () => {
        cancelAnimationFrame(raf);
        start = performance.now();
        last = -1;
        frame();
    };
    pre.addEventListener("click", run);
    document.querySelector("[data-toggle]")?.addEventListener("click", () => {
        pre.innerHTML = toHtml(toCells(splashFrame(END_TICK + 1)), palette());
    });
    run();
}
