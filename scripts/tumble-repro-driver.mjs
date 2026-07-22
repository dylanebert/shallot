// Trusted-input floor-vanish repro driver (spec tumble-inline stage 6b / F1). Runs under NODE, not bun:
// Bun's Playwright client hangs after the ws upgrade on WSL (scripts/wsl-bridge.ts fact 2), so the same
// node-bundled path `shallot verify --connect` uses is the only one that can drive the host's real-GPU
// browser. This is a SIBLING of that path — it reuses the wsl-bridge browser server (via `--connect`) but
// drives the page DIRECTLY through `page.mouse` (browser-trusted CDP input), which the `window.__harness`
// wrapper exposes no seam for. The gym's `window.__tumbleProbe` / `__tumbleAim` (examples/gym/src/
// tumble-watch.ts) are the per-frame instrument: a drawArgs readback + a NaN/Inf scan over body poses AND
// Part instance transforms. The driver drags a bridge plank at escalating flick violence (one-frame cursor
// jumps) and, the moment any pair's drawn count drops below the distinct-mesh count or any pose/transform
// goes non-finite, screenshots and stops — the leading hypothesis (a real-flick grab-spring explosion →
// non-finite transform → shared pack-scan poisoning) made observable.
//
// The orchestrator (scripts/tumble-repro.ts, bun) owns the bridge lifecycle; this owns the vite server + the
// browser drive. Off WSL (`--connect` absent) it launches a local chromium instead. NEVER left open — the
// server + browser tear down on every exit path.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { PNG } from "pngjs";
import { createServer as viteServer } from "vite";

const require = createRequire(import.meta.url);
const REPO_ROOT = resolve(import.meta.dirname, "..");
const GYM_ROOT = resolve(REPO_ROOT, "examples/gym");
const COOP = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

function arg(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const CONNECT = arg("connect", process.env.TUMBLE_REPRO_CONNECT || "");
const OUTDIR = resolve(arg("out", process.env.TUMBLE_REPRO_OUT || "/tmp/tumble-repro"));
const INJECT = arg("inject", ""); // "nan" | "inf" — the red-first self-test for the auto-dump watcher
const GATE = process.argv.includes("--gate"); // standing-gate stress: flick-out/back, all levels, no early return
const RECIPE = arg("recipe", ""); // "drag-below" — the F1′ sustained-downward-drag pixel-level repro
const SPEED = arg("speed", "slow"); // drag speed for the recipe: slow | med
const HOLD_MS = Number(arg("hold", "3000")); // how long to hold the handle below ground
const DEPTH = arg("depth", "below"); // how far the cursor is driven: below (past the bottom edge) | bottom
const PITCH = arg("pitch", "default"); // default framing | steep (orbit more downward first)
const DPR = Number(arg("dpr", "1.5"));
const VW = 1280;
const VH = 800;

// F1′ pixel-breach constants — MIRROR of examples/gym/src/tumble-watch.ts `pixelBreach` (a .mjs can't import the
// TS module). Kept in lockstep with that unit-tested pure rule: a reference patch dimmer than MIN_REF_LUM is
// discarded; a whole static surface (≥2 lit patches) dropping below DARK_FRACTION of reference, or all blowing
// past WASH_MULTIPLE, is a breach.
const MIN_REF_LUM = 25;
const DARK_FRACTION = 0.35;
const WASH_MULTIPLE = 2.5;
const PATCH_HALF = 6; // half-size of a sampled patch, CSS px (before DPR scaling into the screenshot)

// mean linear-ish luminance (Rec.709) over a PATCH_HALF box centered on a client pixel, read out of a decoded
// screenshot (the pixels the user actually sees — the whole point of F1′). `cx`/`cy` are CSS px; the screenshot
// is DPR-scaled, so the box maps to `cx*dpr ± PATCH_HALF*dpr`.
function meanLum(img, cx, cy, dpr) {
    const x0 = Math.max(0, Math.round((cx - PATCH_HALF) * dpr));
    const x1 = Math.min(img.width - 1, Math.round((cx + PATCH_HALF) * dpr));
    const y0 = Math.max(0, Math.round((cy - PATCH_HALF) * dpr));
    const y1 = Math.min(img.height - 1, Math.round((cy + PATCH_HALF) * dpr));
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
            const i = (y * img.width + x) * 4;
            r += img.data[i];
            g += img.data[i + 1];
            b += img.data[i + 2];
            n++;
        }
    }
    if (n === 0) return { r: 0, g: 0, b: 0, lum: 0 };
    r /= n;
    g /= n;
    b /= n;
    return { r, g, b, lum: 0.2126 * r + 0.7152 * g + 0.0722 * b };
}

// one full screenshot, decoded once, sampled at every patch — the pixel-layer read.
async function samplePatches(page, patches) {
    const buf = await page.screenshot({ type: "png" });
    const img = PNG.sync.read(buf);
    return patches.map((p) => ({ surface: p.surface, ...meanLum(img, p.cx, p.cy, DPR) }));
}

// the pure pixel-breach rule (mirror of tumble-watch.ts pixelBreach — see the constants note above).
function pixelBreach(ref, samp) {
    const surfaces = [...new Set(ref.map((r) => r.surface))];
    for (const s of surfaces) {
        const idx = ref.map((_, i) => i).filter((i) => ref[i].surface === s && ref[i].lum >= MIN_REF_LUM);
        if (idx.length < 2) continue;
        if (idx.every((i) => samp[i].lum < DARK_FRACTION * ref[i].lum)) {
            return { surface: s, kind: "dark", patches: idx.length };
        }
        if (idx.every((i) => samp[i].lum > WASH_MULTIPLE * ref[i].lum)) {
            return { surface: s, kind: "wash", patches: idx.length };
        }
    }
    return null;
}

mkdirSync(OUTDIR, { recursive: true });

const log = (...m) => console.error("[repro]", ...m); // stderr — stdout carries only the final JSON
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// escalating drag violence. Each level: grab a plank, then N one-frame cursor jumps of the given magnitude
// (fraction of the viewport diagonal), with rapid direction reversals — a real flick's cursor teleport the
// dispatched-event probe (14 gentle interpolated steps) structurally cannot produce.
const LEVELS = [
    { name: "L1-firm-lift", mag: 0.25, jumps: 8, reversals: 0, settleMs: 40 },
    { name: "L2-fast-swing", mag: 0.45, jumps: 12, reversals: 2, settleMs: 25 },
    { name: "L3-slam-reversals", mag: 0.7, jumps: 16, reversals: 6, settleMs: 16 },
    { name: "L4-full-viewport-whip", mag: 1.0, jumps: 20, reversals: 10, settleMs: 12 },
    { name: "L5-offscreen-overshoot", mag: 1.6, jumps: 24, reversals: 14, settleMs: 8 },
];

async function importPlaywright() {
    try {
        return await import("playwright");
    } catch {
        return await import(require.resolve("playwright"));
    }
}

function pickPort() {
    return new Promise((res, rej) => {
        const net = require("node:net");
        const s = net.createServer();
        s.on("error", rej);
        s.listen(0, "127.0.0.1", () => {
            const p = s.address().port;
            s.close(() => res(p));
        });
    });
}

// the invariant break, mirrored from tumble-watch.ts detectBreach (a .mjs can't import the TS module).
function breachOf(snap) {
    if (!snap) return null;
    if (snap.nonFinite && snap.nonFinite.length > 0) return { kind: "non-finite", ...snap };
    if (snap.drawing >= 0 && snap.drawing < snap.meshes) return { kind: "draw-drop", ...snap };
    return null;
}

async function probe(page) {
    return page.evaluate(() => (window.__tumbleProbe ? window.__tumbleProbe() : null));
}

async function shoot(page, name) {
    const path = resolve(OUTDIR, name);
    try {
        await page.screenshot({ path });
        return path;
    } catch (e) {
        log("screenshot failed:", e?.message ?? e);
        return null;
    }
}

async function main() {
    const port = await pickPort();
    const server = await viteServer({
        root: GYM_ROOT,
        configFile: resolve(GYM_ROOT, "vite.config.ts"),
        server: { port, strictPort: true, open: false, headers: COOP },
    });
    await server.listen();
    const url = `http://localhost:${port}/?scenario=joints-bridge&watch=1`;

    const { chromium } = await importPlaywright();
    const browser = CONNECT
        ? await chromium.connect(CONNECT, { timeout: 30_000 })
        : await chromium.launch({
              headless: true,
              channel: "chromium",
              args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures"],
          });

    const result = {
        scenario: "joints-bridge",
        connect: !!CONNECT,
        dpr: DPR,
        attempts: [],
        reproduced: false,
        breach: null,
        firstBreakFrame: null,
        artifacts: [],
        hardware: "unknown",
    };

    try {
        const context = await browser.newContext({
            viewport: { width: VW, height: VH },
            deviceScaleFactor: DPR,
        });
        const page = await context.newPage();
        const errors = [];
        page.on("pageerror", (e) => errors.push(`${e.name}: ${e.message}`));
        page.on("console", (m) => {
            if (m.type() === "error") errors.push(m.text());
        });

        await page.goto(url, { timeout: 30_000 });
        await page.waitForFunction(() => window.__harness?.ready === true, null, { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.__tumbleProbe === "function", null, {
            timeout: 20_000,
        });
        result.hardware = await page
            .evaluate(async () => {
                const a = await navigator.gpu?.requestAdapter();
                return a ? [a.info.vendor, a.info.architecture, a.info.device].filter(Boolean).join(" / ") : "unknown";
            })
            .catch(() => "unknown");
        log("hardware:", result.hardware);

        // let the deck settle a moment, then baseline the instrument (must read clean before we drag).
        await sleep(1500);
        const baseline = await probe(page);
        result.baseline = baseline;
        result.artifacts.push(await shoot(page, "00-baseline.png"));
        log("baseline:", JSON.stringify(baseline));

        // inject self-test: nan/inf prove the auto-dump watcher fires (red-first); `far` synthetically
        // reproduces the TRACED mechanism — a finite displacement out of frustum → correct multi-pair cull.
        if (INJECT) {
            log(`inject=${INJECT}: injecting to demonstrate the mechanism...`);
            const renderPre = await page.evaluate(() =>
                window.__tumbleRender ? window.__tumbleRender() : null,
            );
            const ok = await page.evaluate((k) => window.__tumbleInject?.(k) ?? false, INJECT);
            // let physics + the solid-layer reconcile settle so the world change reaches the drawn state
            await sleep(1200);
            const renderPost = await page.evaluate(() =>
                window.__tumbleRender ? window.__tumbleRender() : null,
            );
            result.renderPre = renderPre;
            result.renderPost = renderPost;
            log(`inject render pre statics=${renderPre?.statics?.length} bodyCount=${renderPre?.bodyCount} worldGeoms=${renderPre?.worldGeoms}`);
            log(`inject render post statics=${renderPost?.statics?.length} bodyCount=${renderPost?.bodyCount} worldGeoms=${renderPost?.worldGeoms}`);
            const dump = await page.evaluate(() => window.__tumbleFloorDump ?? null);
            const deep = await page.evaluate(() => (window.__tumbleDeep ? window.__tumbleDeep() : null));
            const injShot = await shoot(page, `inject-${INJECT}.png`);
            // a post-inject snapshot (drawing/meshes) so the standing gate can assert the multi-pair loss:
            // `far` bypasses the grab entirely, so the cap can't touch it — a drop here proves the detector
            // still detects (drawing < meshes = some pairs culled, drawing > 0 = others survive).
            const snapshot = await probe(page);
            result.inject = { requested: INJECT, injected: ok, dump, deep, screenshot: injShot, snapshot };
            result.deep = deep;
            log("inject dump:", JSON.stringify(dump));
            if (deep) {
                log(`inject deep: frustumFinite=${deep.frustumFinite} mismatch=${deep.mismatch.length} pairs=${JSON.stringify(deep.pairs)} flung=${deep.flung.length}`);
            }
            process.stdout.write(`${JSON.stringify(result)}\n`);
            return;
        }

        // F1′ — the sustained-downward-drag pixel-level repro: the user's exact gesture, asserted at the pixel
        // layer (the drawArgs checks the escalation driver runs are structurally blind to a shadow/tonemap
        // blackout that keeps instanceCounts intact).
        if (RECIPE === "drag-below") {
            await runRecipe(page, result);
            result.pageErrors = errors.slice(0, 8);
            process.stdout.write(`${JSON.stringify(result)}\n`);
            return;
        }

        const detected = GATE ? await runGate(page, result) : await runAttempts(page, result);
        if (!detected) log(GATE ? "gate clean — no fling across all levels" : "no invariant break across all escalation levels");

        // capture any auto-dump the watcher latched even if the active probe missed the exact frame.
        const dump = await page.evaluate(() => window.__tumbleFloorDump ?? null);
        if (dump) {
            result.watcherDump = dump;
            result.firstBreakFrame = dump.frame;
            if (!result.reproduced) {
                result.reproduced = true;
                result.breach = dump.breach;
                result.artifacts.push(await shoot(page, "watcher-break.png"));
            }
        }
        result.pageErrors = errors.slice(0, 8);
    } finally {
        await browser.close().catch(() => {});
        await server.close().catch(() => {});
    }

    process.stdout.write(`${JSON.stringify(result)}\n`);
}

async function runAttempts(page, result) {
    const diag = Math.hypot(VW, VH);
    for (let li = 0; li < LEVELS.length; li++) {
        const lvl = LEVELS[li];
        // re-warm + re-aim a fresh plank each level (different planks → different draw pairs).
        await sleep(600);
        const aim = await page.evaluate(() => (window.__tumbleAim ? window.__tumbleAim() : null));
        const attempt = { level: lvl.name, magnitude: lvl.mag, aim, bursts: [], broke: false };
        result.attempts.push(attempt);
        if (!aim) {
            log(`${lvl.name}: no grabbable plank on screen, skipping`);
            continue;
        }
        log(`${lvl.name}: grab at ${aim.clientX.toFixed(0)},${aim.clientY.toFixed(0)} mag=${lvl.mag}`);

        // grab: hover, press left over the plank, let the down-edge grab engage.
        await page.mouse.move(aim.clientX, aim.clientY, { steps: 3 });
        await sleep(30);
        await page.mouse.down({ button: "left" });
        await sleep(80);

        const amp = (lvl.mag * diag) / 2;
        let broke = false;
        for (let j = 0; j < lvl.jumps && !broke; j++) {
            // one-frame cursor teleport: a single move event (steps:1), no interpolation — the flick a real
            // hand produces and a dispatched-event sweep can't. Alternate the target hard for reversals.
            const rev = j < lvl.reversals ? (j % 2 === 0 ? 1 : -1) : 1;
            const tx = aim.clientX + rev * amp * (0.4 + 0.6 * Math.random());
            const ty = aim.clientY - amp * (0.4 + 0.6 * Math.random());
            await page.mouse.move(tx, ty, { steps: 1 });
            await sleep(lvl.settleMs);

            const snap = await probe(page);
            const breach = breachOf(snap);
            attempt.bursts.push({ j, target: [Math.round(tx), Math.round(ty)], snap });
            if (breach) {
                broke = true;
                attempt.broke = true;
                result.reproduced = true;
                result.breach = breach;
                log(`*** BREACH at ${lvl.name} burst ${j}: ${breach.kind} drawing=${breach.drawing}/${breach.meshes} nonFinite=${breach.nonFinite?.length ?? 0}`);
                // F2: the layer-bisection readback captured at the exact break frame.
                const deep = await page.evaluate(() => (window.__tumbleDeep ? window.__tumbleDeep() : null));
                result.deep = deep;
                if (deep) {
                    log(`    deep: frustumFinite=${deep.frustumFinite} extreme=${deep.frustumExtreme?.toExponential?.(2)} flung=${deep.flung.length} mismatch=${deep.mismatch.length}`);
                    log(`    pairs: ${JSON.stringify(deep.pairs)}`);
                    log(`    flung: ${JSON.stringify(deep.flung.slice(0, 8))}`);
                }
                result.artifacts.push(await shoot(page, `break-${lvl.name}-burst${j}.png`));
            }
        }
        await page.mouse.up({ button: "left" });
        await sleep(120);
        if (broke) return true;
    }
    return false;
}

// The standing-gate stress (`--gate`): a real-device smoke that a violent trusted-input drag never flings a
// pair out of frustum under the grab-energy cap. For each escalating magnitude, grab a plank and repeatedly
// flick the cursor one frame FAR then snap it BACK to the grab point — a reversing whip with ~zero net drift,
// so a pair-drop can only be a fling (a neighbour launched out of frustum), never the grabbed body legitimately
// dragged off-screen. Probes after every teleport (plus the always-on watch=1 auto-dump backstop); runs ALL
// levels, returning whether any breach fired. NOTE: this is a smoke, not the red→green discriminator — the cap
// bounds a SUSTAINED far-anchor, which a hand can't produce here and this reversing whip deliberately avoids
// (so it stays green pre- and post-cap); the deterministic cap regression is the headless whip-cap unit test
// (tumble-pilot.test.ts). The gate's discriminating half is `--inject far` (below).
async function runGate(page, result) {
    const diag = Math.hypot(VW, VH);
    const mags = [0.7, 1.0, 1.4];
    let any = false;
    for (const mag of mags) {
        await sleep(500);
        const aim = await page.evaluate(() => (window.__tumbleAim ? window.__tumbleAim() : null));
        const attempt = { level: `gate-flick-${mag}`, magnitude: mag, aim, bursts: [], broke: false };
        result.attempts.push(attempt);
        if (!aim) {
            log(`gate ${mag}: no grabbable plank on screen, skipping`);
            continue;
        }
        log(`gate ${mag}: grab at ${aim.clientX.toFixed(0)},${aim.clientY.toFixed(0)}`);
        await page.mouse.move(aim.clientX, aim.clientY, { steps: 3 });
        await sleep(30);
        await page.mouse.down({ button: "left" });
        await sleep(80);

        const amp = (mag * diag) / 2;
        const flicks = 12;
        let broke = false;
        for (let j = 0; j < flicks && !broke; j++) {
            const sign = j % 2 === 0 ? 1 : -1;
            // flick one frame FAR (a single move, no interpolation) then snap BACK to the exact grab point,
            // probing after each — the far flick is the velocity spike, the return keeps net drift ~zero.
            for (const [tx, ty] of [
                [aim.clientX + sign * amp, aim.clientY - amp * 0.5],
                [aim.clientX, aim.clientY],
            ]) {
                await page.mouse.move(tx, ty, { steps: 1 });
                await sleep(12);
                const snap = await probe(page);
                const breach = breachOf(snap);
                attempt.bursts.push({ j, target: [Math.round(tx), Math.round(ty)], snap });
                if (breach) {
                    broke = true;
                    any = true;
                    attempt.broke = true;
                    result.reproduced = true;
                    result.breach = breach;
                    log(`*** GATE FLING at mag ${mag} flick ${j}: ${breach.kind} drawing=${breach.drawing}/${breach.meshes} nonFinite=${breach.nonFinite?.length ?? 0}`);
                    const deep = await page.evaluate(() => (window.__tumbleDeep ? window.__tumbleDeep() : null));
                    result.deep = deep;
                    result.artifacts.push(await shoot(page, `gate-fling-${mag}-flick${j}.png`));
                    break;
                }
            }
        }
        await page.mouse.up({ button: "left" });
        await sleep(150);
    }
    return any;
}

// ── F1′ recipe: sustained downward central-plank drag, asserted at the pixel layer ──────────────────────────
// The user's exact gesture (post-cap, native): grab a CENTRAL bridge plank, drag it DOWNWARD until the grab
// handle is below the ground plane, HOLD there, then continue — and the static ground + static end posts go
// black while dynamics stay correct. That symptom is invisible to a drawArgs readback (a shadow/regather/
// tonemap blackout keeps instanceCounts whole), so this reads PIXELS: reference patches over the known-static
// ground + posts, sampled each drag/hold step, breached when a whole static surface drops near-black at once.
// The KEY BIT recorded on a breach: pixels broken WITH drawArgs green (render-side corruption) vs drawArgs
// dropped (the cull layer). One attempt per driver run; the orchestrator (tumble-repro.ts) sweeps the params.

async function orbitPitch(page, dyPixels) {
    // left-drag over empty sky (top-center — no body there, so OrbitPick doesn't claim the press) to change the
    // camera pitch. Direction/magnitude are best-effort; the caller records the resulting pitch either way.
    const sx = VW * 0.5;
    const sy = 60;
    await page.mouse.move(sx, sy, { steps: 2 });
    await page.mouse.down({ button: "left" });
    const steps = 24;
    for (let i = 1; i <= steps; i++) {
        await page.mouse.move(sx, sy + (dyPixels * i) / steps, { steps: 2 });
        await sleep(14);
    }
    await page.mouse.up({ button: "left" });
}

// place reference pixel patches over the static surfaces: the two end posts (from __tumbleStatics, y≈6) and a
// spread of ground-top points to the sides/front of the bridge (away from where the dragged bodies travel).
// Keep only patches that project on-screen AND are reliably lit at baseline (a dark patch can't distinguish a
// black-out from an unlit surface).
async function buildPatches(page) {
    const statics = await page.evaluate(() => window.__tumbleStatics());
    const patches = [];

    // posts: static bodies sitting at the deck height (y in [4,8]); ground is the one near y=0.
    const posts = statics.filter((p) => p[1] >= 4 && p[1] <= 8);
    if (posts.length) {
        const s = await page.evaluate((pts) => window.__tumbleProject(pts), posts);
        posts.forEach((p, i) => {
            if (s[i].inView) patches.push({ surface: "post", cx: s[i].x, cy: s[i].y, world: p });
        });
    }

    // ground: points on the ground top face (y=1), spread wide so the dragged plank/boxes don't occlude them.
    const cands = [
        [10, 1, 10],
        [-10, 1, 8],
        [11, 1, -6],
        [-9, 1, -10],
        [14, 1, 3],
        [-14, 1, -2],
        [7, 1, 13],
        [-7, 1, -13],
        [15, 1, 11],
        [-15, 1, 10],
        [12, 1, 14],
        [-12, 1, -14],
    ];
    const gs = await page.evaluate((pts) => window.__tumbleProject(pts), cands);
    cands.forEach((p, i) => {
        if (gs[i].inView) patches.push({ surface: "ground", cx: gs[i].x, cy: gs[i].y, world: p });
    });

    // baseline-brightness filter: drop patches that aren't reliably lit (< MIN_REF_LUM).
    const lum = await samplePatches(page, patches);
    return patches.filter((_, i) => lum[i].lum >= MIN_REF_LUM);
}

async function runRecipe(page, result) {
    const rec = {
        speed: SPEED,
        holdMs: HOLD_MS,
        depth: DEPTH,
        pitch: PITCH,
        samples: [],
        fired: false,
    };
    result.recipe = rec;

    // 1. settle the dropped boxes onto the deck before we aim (the gold runs 600 steps; the live scene needs a
    // moment too — a box mid-fall over the centre would grab instead of a plank).
    await sleep(2800);

    const camBefore = await page.evaluate(() => window.__tumbleCam());
    if (PITCH === "steep") {
        await orbitPitch(page, 0.35 * VH);
        await sleep(500);
    }
    const cam0 = await page.evaluate(() => window.__tumbleCam());
    rec.cam0 = cam0;
    log(`camera pitch ${camBefore.pitch.toFixed(3)} → ${cam0.pitch.toFixed(3)} (${PITCH})`);

    // 2. reference patches over the static surfaces
    const patches = await buildPatches(page);
    rec.patchDefs = patches.map((p) => ({
        surface: p.surface,
        cx: Math.round(p.cx),
        cy: Math.round(p.cy),
        world: p.world,
    }));
    const bySurface = {};
    for (const p of patches) bySurface[p.surface] = (bySurface[p.surface] ?? 0) + 1;
    log(`patches: ${JSON.stringify(bySurface)}`);
    if (Object.values(bySurface).every((n) => n < 2)) {
        result.error = "no static surface with ≥2 reliably-lit reference patches";
        return;
    }
    const refLum = await samplePatches(page, patches);
    rec.refLum = refLum.map((l) => Math.round(l.lum));
    result.artifacts.push(await shoot(page, `recipe-00-baseline.png`));

    // F2′ render-layer readback at baseline — the inputs the color FS shades from, captured clean before the
    // drag so the breach capture (step 5–7) can be diffed against it to name the layer that lost truth.
    rec.renderBaseline = await page.evaluate(() =>
        window.__tumbleRender ? window.__tumbleRender() : null,
    );
    log(`render baseline: ${JSON.stringify(rec.renderBaseline)}`);

    // 3. aim the CENTRAL plank (x≈0, between the two central dropped boxes → an exposed plank).
    let aim = await page.evaluate(() => window.__tumbleAimAt(0, 6, 0));
    if (!aim || !aim.hitDynamic) {
        for (const dx of [-0.2, 0.2, -0.5, 0.5, -0.75, 0.75]) {
            aim = await page.evaluate((x) => window.__tumbleAimAt(x, 6, 0), dx);
            if (aim?.hitDynamic) break;
        }
    }
    if (!aim || !aim.hitDynamic) {
        result.error = "no central plank on screen to grab";
        return;
    }
    rec.aim = { clientX: Math.round(aim.clientX), clientY: Math.round(aim.clientY), bodyPos: aim.bodyPos };
    log(`grab central plank at ${aim.clientX | 0},${aim.clientY | 0} bodyY=${aim.bodyPos?.[1]?.toFixed(2)}`);

    // 4. grab (down-edge) — reset the per-frame velocity peak so it captures only the drag's spike
    await page.evaluate(() => window.__tumbleVelPeak?.(true));
    await page.mouse.move(aim.clientX, aim.clientY, { steps: 3 });
    await sleep(50);
    await page.mouse.down({ button: "left" });
    await sleep(150);

    let fired = null;
    const anchorYs = [];
    const sampleOnce = async (tag) => {
        const lum = await samplePatches(page, patches);
        const probe = await page.evaluate(() => window.__tumbleProbe());
        const grab = await page.evaluate(() => window.__tumbleGrab());
        const breach = pixelBreach(refLum, lum);
        const anchorY = grab?.anchor?.[1];
        if (typeof anchorY === "number") anchorYs.push(anchorY);
        rec.samples.push({
            tag,
            lum: lum.map((l) => Math.round(l.lum)),
            drawing: probe?.drawing,
            meshes: probe?.meshes,
            nonFinite: probe?.nonFinite?.length ?? 0,
            anchorY: anchorY != null ? Number(anchorY.toFixed(2)) : null,
            plankY: grab?.body?.[1] != null ? Number(grab.body[1].toFixed(2)) : null,
            breach: breach ?? null,
        });
        if (breach) {
            // F2′: capture the render-layer readback AT the exact breach frame (not a few frames later) so the
            // world-layer truth (body count, unbounded geometry count, static positions) is the breach state.
            const render = await page.evaluate(() =>
                window.__tumbleRender ? window.__tumbleRender() : null,
            );
            return { breach, probe, grab, lum, render };
        }
        return null;
    };

    // 5. smooth interpolated downward drag — many small steps, no one-frame jumps — carrying the cursor down to
    // (and, for DEPTH=below, past) the bottom edge so the pick ray dips below the ground plane.
    const incr = SPEED === "slow" ? 9 : 22;
    const dwell = SPEED === "slow" ? 55 : 28;
    const targetY = DEPTH === "below" ? VH + 140 : VH - 6;
    const cx = aim.clientX;
    let cy = aim.clientY;
    let k = 0;
    while (cy < targetY && !fired) {
        cy += incr;
        await page.mouse.move(cx, cy, { steps: 2 });
        await sleep(dwell);
        if (k++ % 2 === 0) fired = await sampleOnce("drag");
    }

    // 6. HOLD the handle below the plane, sampling — the frames-accumulate window the report describes.
    const holdEnd = Date.now() + HOLD_MS;
    while (Date.now() < holdEnd && !fired) {
        await page.mouse.move(cx, cy, { steps: 1 });
        await sleep(140);
        fired = await sampleOnce("hold");
    }

    // 7. continue: sweep sideways-below and further down (the report's "then continue dragging").
    if (!fired) {
        const sweeps = [
            [cx - 0.35 * VW, cy],
            [cx + 0.35 * VW, cy],
            [cx, cy + 90],
            [cx - 0.3 * VW, cy + 60],
        ];
        for (const [tx, ty] of sweeps) {
            if (fired) break;
            const seg = 8;
            for (let i = 1; i <= seg && !fired; i++) {
                await page.mouse.move(cx + ((tx - cx) * i) / seg, cy + ((ty - cy) * i) / seg, {
                    steps: 2,
                });
                await sleep(dwell);
                if (i % 2 === 0) fired = await sampleOnce("sweep");
            }
            const he = Date.now() + 900;
            while (Date.now() < he && !fired) {
                await page.mouse.move(tx, ty, { steps: 1 });
                await sleep(140);
                fired = await sampleOnce("sweep-hold");
            }
        }
    }

    rec.minAnchorY = anchorYs.length ? Math.min(...anchorYs) : null;
    rec.anchorWentBelowGround = rec.minAnchorY != null && rec.minAnchorY < 1; // ground top is y=1

    if (fired) {
        result.reproduced = true;
        rec.fired = true;
        const green =
            fired.probe &&
            fired.probe.drawing === fired.probe.meshes &&
            (fired.probe.nonFinite?.length ?? 0) === 0;
        const deep = await page.evaluate(() => window.__tumbleDeep());
        const cam = await page.evaluate(() => window.__tumbleCam());
        // F2′ render-layer readback AT the breach frame (captured in sampleOnce) — diff against renderBaseline.
        rec.renderBreach = fired.render ?? null;
        // the peak dynamic velocity the drag produced (transient — the pose scan misses it; this is the value
        // that fattened the broadphase AABB the frame the statics dropped out).
        rec.velPeak = await page.evaluate(() => window.__tumbleVelPeak?.() ?? null);
        log(`render breach: ${JSON.stringify(rec.renderBreach)}`);
        log(`velPeak during drag: ${JSON.stringify(rec.velPeak)}`);
        const atShot = await shoot(page, `recipe-BREACH-${rec.speed}-${rec.pitch}.png`);
        result.artifacts.push(atShot);
        rec.breach = {
            surface: fired.breach.surface,
            kind: fired.breach.kind,
            patches: fired.breach.patches,
            // THE KEY BIT: pixels broken with the drawArgs layer still GREEN = render-side corruption; drawArgs
            // dropped = the cull/count layer. This single bit is the most important output of the stage.
            drawArgsGreen: !!green,
            drawing: fired.probe?.drawing,
            meshes: fired.probe?.meshes,
            nonFinite: fired.probe?.nonFinite?.length ?? 0,
            refLum: refLum.map((l) => Math.round(l.lum)),
            atLum: fired.lum.map((l) => Math.round(l.lum)),
            anchor: fired.grab?.anchor,
            anchorBelowGround: fired.grab ? fired.grab.anchor[1] < 1 : null,
            plank: fired.grab?.body,
            cam,
            deep: deep
                ? {
                      frustumFinite: deep.frustumFinite,
                      frustumExtreme: deep.frustumExtreme,
                      mismatch: deep.mismatch.length,
                      pairs: deep.pairs,
                      flung: deep.flung.length,
                  }
                : null,
            atShot,
        };
        log(
            `*** PIXEL BREACH: ${fired.breach.surface} ${fired.breach.kind} — drawArgsGreen=${!!green} anchorY=${fired.grab?.anchor?.[1]?.toFixed(2)} drawing=${fired.probe?.drawing}/${fired.probe?.meshes}`,
        );

        // recovery: is it transient? release and watch the patches recover.
        await page.mouse.up({ button: "left" });
        await sleep(300);
        const recov = [];
        for (let i = 0; i < 6; i++) {
            const lum = await samplePatches(page, patches);
            recov.push({
                lum: lum.map((l) => Math.round(l.lum)),
                breached: pixelBreach(refLum, lum) != null,
            });
            await sleep(200);
        }
        rec.recovery = recov;
        result.artifacts.push(await shoot(page, `recipe-post-release.png`));
    } else {
        await page.mouse.up({ button: "left" });
        await sleep(200);
        result.artifacts.push(await shoot(page, `recipe-nofire-${rec.speed}-${rec.pitch}-${rec.depth}.png`));
        log(
            `no pixel breach — minAnchorY=${rec.minAnchorY} (below ground=${rec.anchorWentBelowGround}), ${rec.samples.length} samples`,
        );
    }
}

main().catch((e) => {
    console.error("[repro] fatal:", e?.stack ?? e);
    process.stdout.write(`${JSON.stringify({ reproduced: false, error: String(e?.message ?? e) })}\n`);
    process.exit(1);
});
