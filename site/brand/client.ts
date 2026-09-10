import {
    DARK,
    END_TICK,
    type Grid,
    splash,
    splashFrame,
    TICK_MS,
    toSvg,
} from "../../packages/shallot/src/standard/loading/mark";

// Browser entry for the site pages. Splashes the lockup in as pixel squares on `[data-splash-svg]`
// and paints the terminal on `[data-terminal]`; click replays either. Shows the WebGPU note only
// where WebGPU is missing.

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

const vars = { gold: "var(--gold)", dim: "var(--dim)", ink: "var(--ink)", bg: "var(--bg)" };

for (const el of document.querySelectorAll<HTMLElement>("[data-splash-svg]")) {
    const scale = Number(el.dataset.scale ?? "4");
    const s = splash(el, (grid) => toSvg(grid, vars, scale), reduced);
    s.play();
    el.addEventListener("click", () => s.play());
}

// A terminal, as a terminal paints it: text in the real font, block cells as flush fills at the
// cell's own geometry. What `bun create shallot` prints, splash included.
for (const canvas of document.querySelectorAll<HTMLCanvasElement>("[data-terminal]")) {
    const font = 13;
    const lines = [
        "$ bun create shallot my-game",
        null,
        "· wrote my-game/shallot.json",
        "· wrote my-game/main.scene",
        "· ready in 41ms",
        "$ ",
    ];
    const splashRows = 7;
    const rows = lines.length - 1 + splashRows;
    const cols = 56;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    let cw = font * 0.6;
    const ch = Math.round(font * 1.2);
    const size = () => {
        canvas.width = cols * cw * dpr;
        canvas.height = rows * ch * dpr;
        canvas.style.width = `${cols * cw}px`;
        canvas.style.aspectRatio = `${cols * cw} / ${rows * ch}`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.font = `${font}px "JetBrains Mono", monospace`;
        ctx.textBaseline = "middle";
    };
    const text = (row: number, s: string, color: string) => {
        ctx.fillStyle = color;
        ctx.fillText(s, 0, row * ch + ch / 2);
    };
    const cells = (grid: Grid, row: number) => {
        grid.forEach((line, y) => {
            line.forEach((tone, x) => {
                if (!tone) return;
                ctx.fillStyle = DARK[tone];
                ctx.fillRect(x * cw, row * ch + (y * ch) / 2, cw + 0.5, ch / 2 + 0.5);
            });
        });
    };
    const after = [END_TICK + 4, END_TICK + 7, END_TICK + 10, END_TICK + 13];
    const draw = (tick: number) => {
        ctx.clearRect(0, 0, cols * cw, rows * ch);
        text(0, lines[0] ?? "", DARK.ink);
        cells(splashFrame(Math.min(tick, END_TICK + 1)), 1);
        let row = 1 + splashRows;
        lines.slice(2).forEach((line, i) => {
            const at = after[i] ?? 0;
            if (tick >= at && line) text(row, line, i === 3 ? DARK.ink : "#a08c78");
            if (tick >= at && i === 3) {
                ctx.fillStyle = DARK.gold;
                ctx.fillRect(2 * cw, row * ch, cw, ch);
            }
            row++;
        });
    };
    let start = 0;
    let last = -1;
    let raf = 0;
    const loop = () => {
        const tick = reduced ? (after[3] ?? 0) : Math.floor((performance.now() - start) / TICK_MS);
        if (tick !== last) {
            last = tick;
            draw(tick);
        }
        if (tick <= (after[3] ?? 0)) raf = requestAnimationFrame(loop);
    };
    const run = () => {
        cancelAnimationFrame(raf);
        start = performance.now();
        last = -1;
        loop();
    };
    canvas.addEventListener("click", run);
    document.fonts.load(`${font}px "JetBrains Mono"`).then(() => {
        size();
        cw = ctx.measureText("█").width;
        size();
        run();
    });
}

// `?webgpu=0` previews the note on a machine that has WebGPU; `?webgpu=1` hides it on one that
// doesn't. Without the switch the page reads the real adapter.
const note = document.querySelector<HTMLElement>("[data-webgpu-note]");
const forced = new URLSearchParams(location.search).get("webgpu");
const hasGpu = forced === null ? "gpu" in navigator : forced !== "0";
if (note && !hasGpu) note.hidden = false;
