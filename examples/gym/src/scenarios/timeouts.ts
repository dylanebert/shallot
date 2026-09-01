// Per-scenario gate metadata — plain data with no imports, so a driver reads it node-side WITHOUT
// booting a page (the same committed-data shape bench-tumble.ts reads its twin list from
// tests/tumble/samples/index.json). `bun bench --for src/standard/sear/pipelines.ts` has to resolve a
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
// - `covers` — glob(s) into `packages/shallot/src` naming the GPU-side modules this scenario exercises.
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
    /** the public asset paths (relative to `examples/gym/public/`) this scenario needs, resolved
     *  against the run's params so per-mode selection is exact. `bun bench` checks these on the
     *  filesystem before booting a page; a missing path skips the scenario with a clear message
     *  instead of failing through the glTF loader. Return `[]` when the params need no mount. */
    assets?: (params: Record<string, string | number | boolean>) => string[];
}

export const SCENARIO_GATES: Record<string, ScenarioGate> = {
    stress: {
        timeoutMs: 180_000,
        isolate: true,
        covers: ["packages/shallot/src/extras/profile/**/*.ts"],
    },
    outline: {
        covers: ["packages/shallot/src/extras/outline/**/*.ts"],
    },
    sprite: {
        covers: ["packages/shallot/src/extras/sprite/**/*.ts"],
    },
    text: {
        covers: ["packages/shallot/src/extras/text/**/*.ts"],
    },
    gltf: {
        covers: ["packages/shallot/src/extras/gltf/**/*.ts"],
        // keyed to gltf.ts's SOURCES — the same paths loadGltf fetches
        assets: (p) => {
            const source = (p.source as string) ?? "sponza";
            return source === "fox"
                ? ["gltf-samples/Fox/glTF/Fox.gltf"]
                : ["sponza/Sponza-KTX-Draco.glb"];
        },
    },
    // `accel`'s framebuffer probe (`assertLineDraw`) reads the restored live scene through the lines
    // surface's real rendered output (the ray overlay) — a verified real GPU exerciser of `extras/lines`,
    // not an incidental import.
    accel: {
        covers: [
            "packages/shallot/src/standard/bvh/**/*.ts",
            "packages/shallot/src/extras/lines/**/*.ts",
        ],
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
            "packages/shallot/src/standard/render/**/*.ts",
            "packages/shallot/src/standard/sear/**/*.ts",
            "packages/shallot/src/standard/part/**/*.ts",
            "packages/shallot/src/standard/slab/**/*.ts",
            "packages/shallot/src/extras/sky/**/*.ts",
            "packages/shallot/src/extras/skin/**/*.ts",
            "packages/shallot/src/engine/utils/encode.ts",
        ],
        // keyed to render.ts's GLTF_VARIANTS / FOX / SPILL_ASSETS / MULTI / WORKER_* — the same paths
        // loadGltf and the scene preloader fetch. Only the gltf modes need mounts; cull/shaded/fog
        // etc. author their scenes in code.
        assets: (p) => {
            const mode = (p.mode as string) ?? "cull";
            const gltfModes = [
                "gltf-model",
                "gltf-animated",
                "gltf-spill",
                "gltf-multi",
                "gltf-worker",
            ];
            if (!gltfModes.includes(mode)) return [];
            if (mode === "gltf-model") {
                const variant = (p.variant as string) ?? "gltf";
                const variants: Record<string, string> = {
                    gltf: "sponza/Sponza.gltf",
                    draco: "sponza/Sponza-Draco.glb",
                    ktx: "sponza/Sponza-KTX.glb",
                    "ktx-draco": "sponza/Sponza-KTX-Draco.glb",
                };
                return [variants[variant] ?? variants.gltf];
            }
            if (mode === "gltf-animated") return ["gltf-samples/Fox/glTF/Fox.gltf"];
            if (mode === "gltf-spill")
                return [
                    "gltf-samples/StainedGlassLamp/glTF-KTX-BasisU/StainedGlassLamp.gltf",
                    "gltf-samples/ChronographWatch/glTF-KTX-BasisU/ChronographWatch.gltf",
                ];
            if (mode === "gltf-multi")
                return [
                    "gltf-samples/DamagedHelmet/glTF/DamagedHelmet.gltf",
                    "gltf-samples/WaterBottle/glTF/WaterBottle.gltf",
                    "gltf-samples/Fox/glTF/Fox.gltf",
                    "gltf-samples/CesiumMan/glTF/CesiumMan.gltf",
                ];
            // gltf-worker: the scene authors Box/Draco + BoxTextured by name (preloader), and the
            // assert decodes KTX + Corrupt directly — all under gltf-samples/
            return [
                "gltf-samples/Box/glTF-Draco/Box.gltf",
                "gltf-samples/BoxTextured/glTF-Binary/BoxTextured.glb",
                "gltf-samples/AnisotropyBarnLamp/glTF-KTX-BasisU/AnisotropyBarnLamp.gltf",
                "gltf-samples/Box/glTF-Draco/Box.bin",
            ];
        },
    },

    // the release-prerequisite final-compositor hardening fixture: a
    // custom-registered surface + a Part entity on the built-in `unlit` surface, both drawn through the
    // real part/render/sear pipeline — folds into the same modules `render`'s entry already covers.
    "mesh-fixture": {
        covers: [
            "packages/shallot/src/standard/part/**/*.ts",
            "packages/shallot/src/standard/render/**/*.ts",
            "packages/shallot/src/standard/sear/**/*.ts",
        ],
    },

    // ── AVBD (GPU physics) — each imports AvbdPlugin and gates a slice of the solver end to end ──
    backend: {
        // the substrate swap gate: `--param backend=tumble|avbd` runs the same scene under either
        // backend, so it is a real (if secondary) exerciser of the avbd path.
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },
    character: {
        // felt-lag GPU-load probe: `probeChecks`'s "position input→camera carries no GPU readback" check
        // asserts a frame-count delta under a calibrated GPU load — a perf-threshold check (the same
        // felt-lag shape as `stress`), untrustworthy under sweep contention.
        isolate: true,
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },
    constraints: {
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },
    motor: {
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },
    pile: {
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },
    sat: {
        // the GPU-SAT codegen gate (hull/rounded narrowphase matrix vs the C gold vectors) — the codegen
        // it validates lives in `standard/avbd`.
        covers: ["packages/shallot/src/standard/avbd/**/*.ts"],
    },

    // ── everything below is a tumble (CPU wasm) scenario: no `covers`, deliberately. Tumble physics is
    // bit-exact-gated by `bun test` + the committed fixtures/gold corpus (`tumble.md`), not by this
    // check's GPU-src population, which excludes `standard/tumble` for exactly that reason. Registered
    // names are gold slugs / sample names, not filenames — resolved from the real roster
    // (`scenarioNames()`), not guessed, since a scenario name mismatch here would silently pass
    // `checkCompleteness` on the wrong key.

    // tumble.js sample twins (`sampleScenario`, bit-exact vs a committed gold — no perf-threshold
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

    // hand-authored tumble/diagnostic scenarios, no GPU-src coverage claim:
    queries: {}, // Tumble.world spatial-query surface (castRayClosest/castShape/overlapAABB)
    rotation: {}, // Dzhanibekov flip via StepSystem — physics/core, not GPU-src
    raining: {}, // tumble create/destroy marshal path under constant churn
    chain: {}, // synthetic compute-chain microbench; uses RenderPlugin only as a frame-boundary hook
    // gpu-diagnostic directly drives `validateGpu` (gpu.ts), `drainLog` (log.ts), and `probeBuffer`/
    // `probeTexture` (probe.ts) — verified by import, not guessed.
    "gpu-diagnostic": {
        covers: [
            "packages/shallot/src/engine/runtime/gpu.ts",
            "packages/shallot/src/engine/runtime/log.ts",
            "packages/shallot/src/engine/runtime/probe.ts",
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
    "packages/shallot/src/engine/runtime/index.ts": "barrel re-export, no logic of its own",
    // engine/runtime/platform.ts: `Runtime`/`now`/`requestFrame`/`readFile`/`readBinary` are cross-cutting
    // environment/timing primitives every scenario exercises identically through `run()`'s frame loop, so
    // no single scenario's assert targets a regression here specifically — its own correctness is
    // unit-tested (`runtime.test.ts`, verified: schedules-frame-callbacks + reads-files + timing cases),
    // not something a real-device scenario would newly catch.
    "packages/shallot/src/engine/runtime/platform.ts":
        "cross-cutting frame/timing primitive, unit-gated by runtime.test.ts, not a real-device concern",

    // extras/orbit/** and extras/tween/** are CPU-only (`NON_GPU_EXTRAS`, coverage.ts), so they are not
    // exempted here — the classification lives beside the population, not as exemptions in this table.

    // extras/cells/**: shallot-tui S1 ships the cell-grid contract + a headless fill producer (the
    // compute pass, bind group layout, and pack/unpack codec) with no gym scenario driving it — S1's own
    // scope excludes wiring an example/scenario (that lands with S3's web sink or S4's terminal command).
    // The CPU-logic rungs (schema layout, resolved WGSL, the pack/unpack differential) are `bun test`
    // gated in `extras/cells/*.test.ts`; real-device dispatch coverage is owed once a scene drives it.
    "packages/shallot/src/extras/cells/**/*.ts":
        "no scene wires the fill pass through a scenario yet — deferred to S3/S4, not a gap in this module",
};
