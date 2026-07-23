import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PNG } from "pngjs";
import { skipReason, verify } from "./verify";

// `bun run flows` — the standalone-app engine flows, each a self-contained ejected vite app under
// examples/flows/, driven through `shallot verify`. survive-reload self-drives a real page reload and its
// harness asserts a runtime value + a warm-derived entity survived (verify's unified wait polls across the
// self-navigation by construction). ui-containment mounts a deliberately invalid `config.ui` HUD; paint
// containment isn't observable in-page, so verify captures the screenshot and this asserts the magenta /
// host-chrome pixels node-side. blank is the pixel-gate red-proof: a draw-nothing app verify must fail
// with rendered:false (an expected-fail). Display-gated; on WSL/headless it skips (native hardware only).

const MAGENTA: [number, number, number] = [255, 0, 255];
const near = (a: number, b: number, t = 40): boolean => Math.abs(a - b) <= t;
const isMagenta = (r: number, g: number, b: number): boolean =>
    near(r, MAGENTA[0]) && near(g, MAGENTA[1]) && near(b, MAGENTA[2]);

// survive-reload: verify drives the app's self-reload dance and its restored-boot harness (n restored +
// exactly one warm-derived sprout). The gate is verify's pass AND the harness verdict's ok AND both named
// checks present — a harness that readies without a run() reports ok:true as a bare boot smoke, and this
// flow must never degrade to that (the assertions are the point).
const SURVIVE_CHECKS = ["runtime value survived the reload", "warm-derived sprout not doubled"];

async function surviveReload(): Promise<boolean> {
    console.log("\n--- survive-reload ---");
    const result = await verify("examples/flows/survive-reload", ["--timeout", "60000"]);
    let ok = result?.pass === true && result.verdict?.ok === true;
    for (const name of SURVIVE_CHECKS) {
        if (!result?.verdict?.checks?.some((c) => c.name === name && c.ok)) {
            console.log(`  ✗ missing or failed check: ${name}`);
            ok = false;
        }
    }
    console.log(ok ? "PASS: survive-reload" : "FAIL: survive-reload");
    return ok;
}

// ui-containment: verify boots the host-chrome page + captures a post-run screenshot; the pixel assertion
// lives here (node-side) because paint containment can't be observed from inside the page. The canvas is
// inset 64px from the window, so any pixel in that border is host chrome and must stay clear of the app's
// magenta HUD; the window center is inside the canvas and must be magenta (the UI mounted + rendered).
async function uiContainment(): Promise<boolean> {
    console.log("\n--- ui-containment ---");
    const shot = join(tmpdir(), `shallot-flow-ui-${Date.now()}.png`);
    const result = await verify("examples/flows/ui-containment", ["--screenshot", shot]);
    if (result?.pass !== true) {
        console.log("FAIL: ui-containment — verify did not pass");
        return false;
    }
    if (result.rendered === "opt-out") {
        console.log(
            "  ○ rendered: opt-out (solid-fill HUD — paint containment checked node-side below)",
        );
    }

    const { width: w, height: h, data } = PNG.sync.read(readFileSync(shot));
    const at = (x: number, y: number): [number, number, number] => {
        const i = (y * w + x) * 4;
        return [data[i], data[i + 1], data[i + 2]];
    };
    let ok = true;
    // points well inside the 64px chrome border (and the top-center strip directly above the canvas) —
    // all host chrome the contained UI must not reach
    const chrome: [number, number][] = [
        [8, 8],
        [w - 9, 8],
        [8, h - 9],
        [w - 9, h - 9],
        [Math.floor(w / 2), 8],
    ];
    for (const [x, y] of chrome) {
        const [r, g, b] = at(x, y);
        if (isMagenta(r, g, b)) {
            console.log(
                `  ✗ host chrome at (${x},${y}) painted by the app UI (rgb ${r},${g},${b})`,
            );
            ok = false;
        }
    }
    // positive: the UI actually mounted + rendered, inside the canvas frame (window center)
    const [cr, cg, cb] = at(Math.floor(w / 2), Math.floor(h / 2));
    if (!isMagenta(cr, cg, cb)) {
        console.log(
            `  ✗ app UI did not render inside the canvas region (center rgb ${cr},${cg},${cb})`,
        );
        ok = false;
    }
    console.log(ok ? "PASS: ui-containment" : "FAIL: ui-containment");
    return ok;
}

// blank red-proof: the standing oracle for the pixel-honest `rendered` verdict. examples/flows/blank draws
// nothing, its harness reports ok:true, and it does NOT declare noRender — so verify MUST fail it with
// rendered:false. This is an expected-fail: the flow passes when verify correctly goes red, so it never
// turns the matrix red while re-proving the gate catches a real blank on every run. The gate is
// specifically the pixel check: also require the fixture's own harness verdict to have SUCCEEDED
// (`verdict.ok`). verify's harness-ready-timeout path returns the same pass:false/rendered:false shape, so
// a bitrotted fixture (dev server dies, harness never installs) would vacuously "pass" the red-proof —
// asserting verdict.ok pins the failure to the pixel gate reading the blank, not to a broken boot.
async function blankRedProof(): Promise<boolean> {
    console.log("\n--- blank (red-proof) ---");
    const result = await verify("examples/flows/blank", ["--timeout", "60000"]);
    const wentRed =
        result?.pass === false && result?.rendered === false && result?.verdict?.ok === true;
    if (wentRed) {
        console.log("PASS: blank — verify went red on the blank canvas (rendered:false)");
    } else {
        console.log(
            `FAIL: blank — expected verify to FAIL on the pixel gate (rendered:false with a passing harness verdict), got pass=${result?.pass} rendered=${result?.rendered} verdict.ok=${result?.verdict?.ok}`,
        );
    }
    return wentRed;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run flows [--flow <name>]

Runs the standalone-app engine flows through \`shallot verify\`. Display-gated (native hardware only).

Options:
  --flow <name>   Run a single flow: survive-reload | ui-containment | blank`);
        process.exit(0);
    }
    const flowIdx = args.indexOf("--flow");
    const only = flowIdx !== -1 ? args[flowIdx + 1] : undefined;

    const skip = skipReason();
    if (skip) {
        console.log(`bun run flows needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    console.log("Running flows...");
    let allPass = true;
    if (!only || only === "survive-reload" || only === "survive") {
        allPass = (await surviveReload()) && allPass;
    }
    if (!only || only === "ui-containment" || only === "ui") {
        allPass = (await uiContainment()) && allPass;
    }
    if (!only || only === "blank") {
        allPass = (await blankRedProof()) && allPass;
    }

    if (!allPass) {
        console.error("\nFAIL: flows failed");
        process.exit(1);
    }
    console.log("\nPASS: flows green");
    process.exit(0);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
