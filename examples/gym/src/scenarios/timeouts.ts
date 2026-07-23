// Per-scenario run() budget, in ms — the scenario-side declaration `bun bench` reads to drive a heavy
// scenario under more than the harness default 60s `--timeout`. It's plain data with no imports, so the
// bench script reads it node-side WITHOUT booting a page (the same committed-data shape bench-tumble.ts
// reads its twin list from tests/tumble/samples/index.json). A scenario with no entry keeps the 60s
// default (verify.ts) — the hang detector stays tight for everything that doesn't legitimately need more.
// An explicit `bun bench --timeout N` overrides any entry here (operator override); scripts/bench.ts
// `benchTimeout` is the resolution.
//
// stress: the bottleneck-saturation atom ramps four resource axes (compute, bandwidth, submission,
// cpu-memory) to the felt-lag wall and then runs fixed-frame profiler measure windows AT that wall — each
// window is a fixed ~230 frames, but a saturated frame is 30–55 ms (vs ~4 ms idle), so the run's wall-clock
// is dominated by frame count × the induced ms/frame and legitimately exceeds the 60s default. Budget is
// derived, not tuned: measured wall-clock 61–70 s on nvidia lovelace (`bun bench --scenario stress`, two
// runs), which the 60s default reds on for sitting just past the boundary. The measure windows are fixed
// frame counts run at a bounded per-frame time (the wall is defined as ms/frame, ~28 ms), so the runtime is
// roughly hardware-independent — a slower device reaches the wall at a lower induced level and runs fewer
// ramp windows, not longer ones. 180_000 is ~2.6× the measured wall-clock: comfortably clear of run-to-run
// variance, matches the sweep's proven-green 180s reference (specs/shallot-081-release.md stage 3), and
// stays well under a genuinely-hung run.
export const SCENARIO_TIMEOUTS: Record<string, number> = {
    stress: 180_000,
};
