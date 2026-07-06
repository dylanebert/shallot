import type { BenchmarkMeasurement } from "@dylanebert/shallot/extras";
import type { MemoryStats } from "../core";
import type { Check } from "./verdict";

// Mixed fixed/variable timing: the frame interval (variable, rAF/vsync paced) is reported with its
// decomposition (cpu + GPU fence-wait + idle gap), so a number near the rAF floor reads as idle, not
// slow. GPU is split by clock — sim passes per fixed step, render passes per frame — so a heavy step
// is never under-reported by amortizing it across the idle frames between steps.
export function printMeasurement(
    label: string,
    measurement: BenchmarkMeasurement,
    mem: MemoryStats | null,
): void {
    const bar = "=".repeat(40);
    const r = measurement;
    console.log(`\n${bar}`);
    console.log(`  ${label} Results`);
    console.log(bar);
    console.log(`  Frames measured: ${r.frames}`);

    if (r.frame) {
        const f = r.frame;
        const idlePct = f.avg > 0 ? Math.round((f.gapMs / f.avg) * 100) : 0;
        console.log(
            `  Frame:   avg ${f.avg.toFixed(2)}  median ${f.median.toFixed(2)}  p95 ${f.p95.toFixed(2)}  p99 ${f.p99.toFixed(2)}  max ${f.max.toFixed(2)} ms`,
        );
        console.log(
            `    = cpu ${f.cpuMs.toFixed(2)} + fence ${f.fenceMs.toFixed(2)} + idle ${f.gapMs.toFixed(2)} ms   (${idlePct}% idle — rAF/vsync paced)`,
        );
        console.log(
            `    fence p95 ${f.fenceP95.toFixed(2)} ms · ${f.stepsPerFrame.toFixed(2)} steps/frame · clamped ${f.clampedFrames} · pending ${f.maxPending}`,
        );
        // even-pacing + the un-clamped spike the dt clamp hides — the robustness-suite signals
        console.log(
            `    stddev ${f.stddev.toFixed(2)} ms · spike(raw) p99 ${f.rawP99.toFixed(2)} / max ${f.rawMax.toFixed(2)} ms`,
        );
    }

    if (r.gpu) {
        const g = r.gpu;
        const steps = r.frame ? r.frame.stepsPerFrame.toFixed(2) : "?";
        console.log(
            `  GPU busy: ${g.busyPerFrameMs.toFixed(3)} ms/frame = render ${g.renderPerFrameMs.toFixed(3)}/frame + sim ${g.simPerStepMs.toFixed(3)}/step × ${steps}`,
        );
        const entries = Object.entries(g.passes);
        const sim = entries
            .filter(([, p]) => p.clock === "sim")
            .sort((a, b) => b[1].occMs - a[1].occMs);
        const render = entries
            .filter(([, p]) => p.clock === "render")
            .sort((a, b) => b[1].perFrameMs - a[1].perFrameMs);
        if (sim.length > 0) {
            console.log(`    sim (per step):`);
            for (const [name, p] of sim)
                console.log(
                    `      ${name.padEnd(20)} ${p.occMs.toFixed(3)} ms  (p99 ${p.occP99.toFixed(3)})`,
                );
        }
        if (render.length > 0) {
            console.log(`    render (per frame):`);
            for (const [name, p] of render)
                console.log(
                    `      ${name.padEnd(20)} ${p.perFrameMs.toFixed(3)} ms  (p99 ${p.occP99.toFixed(3)})`,
                );
        }
    } else {
        console.log(`  GPU timing unavailable (no timestamp-query support)`);
    }

    if (mem) {
        const mb = (b: number) => (b / 1024 / 1024).toFixed(1);
        console.log(`  Memory:  ${mb(mem.start)} → ${mb(mem.end)} MB`);
        console.log(
            `  Growth:  ${(mem.growthPerFrame / 1024).toFixed(2)} KB/frame${mem.leak ? " ⚠ LEAK" : ""}`,
        );
    }
    console.log(`${bar}\n`);
}

// Returns true if every check passed. Prints a one-line verdict per check.
export function printChecks(checks: Check[]): boolean {
    console.log(`  Checks:`);
    let allPass = true;
    for (const c of checks) {
        allPass = allPass && c.pass;
        const mark = c.pass ? "✓" : "✗";
        const detail = c.detail ? `  — ${c.detail}` : "";
        console.log(`    ${mark} ${c.name}${detail}`);
    }
    return allPass;
}
