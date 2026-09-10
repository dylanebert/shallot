// Per-scenario gate metadata — plain data with no imports, so a driver reads it node-side WITHOUT
// booting a page (the same committed-data shape bench-physics.ts reads its twin list from
// tests/physics/samples/index.json). `bun bench --for src/standard/sear/pipelines.ts` has to resolve a
// path to scenario names before any browser exists, which a `covers:` field inside a scenario's own
// registration cannot do — so this stays a side table, not a field on the registered scenario objects, and it
// deliberately touches none of the scenario files it describes.
//
// Fields, all optional:
// - `timeoutMs` — the `run()` budget `bun bench` drives a scenario under, above the harness's tight 60s
//   default hang detector (`verify.ts`). A scenario with no entry keeps the 60s default. An explicit
//   `bun bench --timeout N` overrides any entry here (operator override); `scripts/bench.ts`
//   `benchTimeout` is the resolution.
// - `isolate` — true for a scenario carrying perf-threshold checks. An `isolate` scenario runs in its
//   own process, never folded into the shared-boot batch — grounded on the
//   `stress` sweep-contention finding: a
//   perf-threshold gate measured under back-to-back sweep contention is not trustworthy.
// - `covers` — glob(s) into `src` naming the GPU-side modules this scenario exercises.
//   The coverage check (`coverage.ts`) asserts every glob resolves, every table key is a registered
//   scenario, and (once 3b populates the rest) every scenario has an entry and every GPU-side module is
//   either covered or explicitly exempted.
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
// variance, matches the sweep's proven-green 180s reference, and
// stays well under a genuinely-hung run. It also `isolate`s: it is the scenario the sweep-contention
// finding was measured on (failed mid-sweep on its bandwidth/submission saturation rows, passed standalone
// at two different commits), so its perf-threshold checks run in their own process, never batched.
export interface ScenarioGate {
    timeoutMs?: number;
    isolate?: boolean;
    covers?: string[];
    /** the pinned assets (names in the repo's `assets.json`) this scenario needs for the run's params.
     *  `bun bench` reds a scenario whose declared asset is absent, naming `bun run assets <name>`.
     *  Return `[]` when the params need none. */
    assets?: (params: Record<string, string | number | boolean>) => string[];
}

export const SCENARIO_GATES: Record<string, ScenarioGate> = {
    stress: {
        timeoutMs: 180_000,
        isolate: true,
        covers: ["src/extras/profile/**/*.ts"],
    },
    outline: {
        covers: ["src/extras/outline/**/*.ts"],
    },
    sprite: {
        covers: ["src/extras/sprite/**/*.ts"],
    },
    text: {
        covers: ["src/extras/text/**/*.ts"],
    },
    cells: {
        covers: ["src/extras/cells/**/*.ts"],
    },
    gltf: {
        covers: ["src/extras/gltf/**/*.ts"],
        // keyed to gltf.ts's SOURCES
        assets: (p) => [(p.source as string) === "fox" ? "fox" : "sponza"],
    },
    // `accel`'s framebuffer probe (`assertLineDraw`) reads the restored live scene through the lines
    // surface's real rendered output (the ray overlay) — a verified real GPU exerciser of `extras/lines`,
    // not an incidental import.
    accel: {
        covers: ["src/standard/bvh/**/*.ts", "src/extras/lines/**/*.ts"],
    },
    // `render` is one registered scenario carrying many `mode`-selected rows (barrel header,
    // examples/gym/src/scenarios/index.ts) — skin-live and background/sky are rows of it, not their own
    // scenarios, so their coverage folds into this one entry rather than a same-named table key that
    // wouldn't match a registered scenario. `isolate`: `assertFog`'s "march-cost" check is a per-step
    // GPU-timing tripwire (fog:march / sear:color ratio bounded under `PerStepMax`) — a perf-threshold
    // check, untrustworthy under sweep contention per the `stress` finding. `standard/slab` rides here
    // per `gpu.md`'s own citation ("the render gym scenario exercises the flush via its slab:flush span +
    // a transport round-trip assert" — verified: render.ts's colorBuf Mirror + verifyScatter transport
    // check). `engine/utils/encode.ts` rides here too: its GPU storage codecs (color/quat/normal/position
    // pack) are consumed throughout `standard/render`+`sear` (verified via `utils/core` importers), so a
    // regression there surfaces in render's own pixel/transport probes even though the file sits outside
    // `standard/render/**`.
    render: {
        isolate: true,
        covers: [
            "src/standard/render/**/*.ts",
            "src/standard/sear/**/*.ts",
            "src/standard/part/**/*.ts",
            "src/standard/slab/**/*.ts",
            "src/extras/sky/**/*.ts",
            "src/extras/skin/**/*.ts",
            "src/engine/utils/encode.ts",
        ],
        // keyed to render.ts's GLTF_VARIANTS / FOX / SPILL_ASSETS / MULTI / WORKER_*; only the gltf
        // modes load files, the rest author their scenes in code.
        assets: (p) => {
            const mode = (p.mode as string) ?? "cull";
            if (mode === "gltf-model") {
                const variant = (p.variant as string) ?? "gltf";
                return [variant === "ktx-draco" ? "sponza" : `sponza-${variant}`];
            }
            if (mode === "gltf-animated") return ["fox"];
            if (mode === "gltf-spill") return ["stained-glass-lamp", "chronograph-watch"];
            if (mode === "gltf-multi")
                return ["damaged-helmet", "water-bottle", "fox", "cesium-man"];
            if (mode === "gltf-worker")
                return ["box-draco", "box-textured", "anisotropy-barn-lamp"];
            return [];
        },
    },

    // the release-prerequisite final-compositor hardening fixture: a
    // custom-registered surface + a Part entity on the built-in `unlit` surface, both drawn through the
    // real part/render/sear pipeline — folds into the same modules `render`'s entry already covers.
    "mesh-fixture": {
        covers: [
            "src/standard/part/**/*.ts",
            "src/standard/render/**/*.ts",
            "src/standard/sear/**/*.ts",
        ],
    },

    character: {
        // felt-lag GPU-load probe: `probeChecks`'s "position input→camera carries no GPU readback" check
        // asserts a frame-count delta under a calibrated GPU load — a perf-threshold check (the same
        // felt-lag shape as `stress`), untrustworthy under sweep contention.
        isolate: true,
    },
    backend: {},
    constraints: {},
    motor: {},
    pile: {},
    sat: {},

    // ── everything below is a physics (CPU wasm) scenario: no `covers`, deliberately. physics is
    // bit-exact-gated by `bun test` + the committed fixtures/gold corpus (`physics.md`), not by this
    // check's GPU-src population, which excludes `standard/physics` for exactly that reason. Registered
    // names are gold slugs / sample names, not filenames — resolved from the real roster
    // (`scenarioNames()`), not guessed, since a scenario name mismatch here would silently pass
    // `checkCompleteness` on the wrong key.

    // upstream sample twins (`sampleScenario`, bit-exact vs a committed gold — no perf-threshold
    // assert to isolate):
    "stacking-arch": {},
    "stacking-box-pyramid": {},
    "stacking-dominoes": {},
    "joints-bridge": {},
    "joints-cantilever": {},
    "joints-driving": {},
    "joints-elevator": {},
    "joints-filter": {},
    "joints-paddle": {},
    "joints-parallel": {},
    "joints-pendulum": {},
    "joints-rope": {},
    "joints-suspension": {},
    "bodies-body-type": {},
    "bodies-motion-locks": {},
    "bodies-spinning-book": {},
    "collision-overlap-box": {},
    "collision-ray-curtain": {},
    "collision-shape-cast": {},
    "continuous-bullet-vs-stack": {},
    "continuous-thin-wall": {},
    "determinism-falling-ragdolls": {},
    "events-hit": {},
    "events-joint-break": {},
    "events-sensor-sweep": {},
    "geometry-convex-hull": {},
    "geometry-convex-primitives": {},
    "geometry-hull-reduction": {},
    "shapes-inclined-plane": {},
    "shapes-restitution": {},
    "shapes-shape-soup": {},
    "mesh-terrain": {},
    "mesh-torus": {},
    "ragdoll-ragdoll": {},
    "compound-simple": {},
    "compound-spheres": {},
    "compound-tile-floor": {},
    "character-mover": {},

    // hand-authored physics/diagnostic scenarios, no GPU-src coverage claim:
    queries: {}, // Physics.world spatial-query surface (castRayClosest/castShape/overlapAABB)
    rotation: {}, // Dzhanibekov flip via StepSystem — physics/core, not GPU-src
    raining: {}, // physics create/destroy marshal path under constant churn
    chain: {}, // synthetic compute-chain microbench; uses RenderPlugin only as a frame-boundary hook
    // gpu-diagnostic directly drives `validateGpu` (gpu.ts), `drainLog` (log.ts), and `probeBuffer`/
    // `probeTexture` (probe.ts) — verified by import, not guessed.
    "gpu-diagnostic": {
        covers: [
            "src/engine/runtime/gpu.ts",
            "src/engine/runtime/log.ts",
            "src/engine/runtime/probe.ts",
        ],
    },
    // orbit-touch (`shallot-mobile-controls` spec, S4): an Orbit camera targeting a box, driven by the
    // driver-level touch gate (`../../test/touch.playwright.ts`), not by `assert` — the scenario's own
    // header states the verdict lives entirely in the external driver. RenderPlugin/PartPlugin/SearPlugin/
    // GlazePlugin render the box so there's a live scene to aim at, the same incidental-render role
    // `chain`'s comment above claims for its own RenderPlugin use, not a verified GPU-src exerciser — no
    // `covers` claim. `extras/orbit` is CPU camera math (`NON_GPU_EXTRAS`, coverage.ts), so this scenario's
    // real subject sits entirely outside the GPU-src population this table covers.
    "orbit-touch": {},
};

/** every module path this scenario's checks are explicitly exempted from covering, and why. An honest
 *  partial list is deliberate ("an honest initial exemption list is the point"), not a gap. A reason
 *  names the property
 *  that is actually load-bearing — no GPU surface — never a structural shape a reader would have to
 *  re-verify: "barrel re-export" was twice the stated reason for a file that was nothing of the kind. */
export const GATE_EXEMPTIONS: Record<string, string> = {
    // NOT `standard/sear/index.ts`: it's a genuine barrel too, but `render`'s `covers` glob already
    // matches every file under `standard/sear/**`, so exempting it would be shadowed — the coverage check
    // (`coverage.ts`) asserts covered ∩ exempt = ∅ for exactly this reason.
    //
    // engine/runtime/index.ts: verified — every line is a bare `export { ... } from "./..."`, no logic
    // of its own.
    "src/engine/runtime/index.ts": "barrel re-export, no logic of its own",
    // engine/runtime/platform.ts: `Runtime`/`now`/`requestFrame`/`readFile`/`readBinary` are cross-cutting
    // environment/timing primitives every scenario exercises identically through `run()`'s frame loop, so
    // no single scenario's assert targets a regression here specifically — its own correctness is
    // unit-tested (`runtime.test.ts`, verified: schedules-frame-callbacks + reads-files + timing cases),
    // not something a real-device scenario would newly catch.
    "src/engine/runtime/platform.ts":
        "cross-cutting frame/timing primitive, unit-gated by runtime.test.ts, not a real-device concern",

    // extras/orbit/** and extras/animation/** are CPU-only (`NON_GPU_EXTRAS`, coverage.ts), so they are not
    // exempted here — the classification lives beside the population, not as exemptions in this table.

    // extras/cells/**: covered by the `cells` scenario's own `covers` glob above, not exempted — an
    // exemption row here would be shadowed (`coverage.ts` asserts covered ∩ exempt = ∅), and shadowing is
    // exactly the earlier defect this table's own doc comment names.
};
