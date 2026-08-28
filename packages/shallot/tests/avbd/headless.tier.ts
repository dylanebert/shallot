// the headless lavapipe physics gate — the AVBD stack builds, steps, and reads back on the
// software adapter `bun test` preloads, at an entity capacity its contact store fits under.
//
// What this proves (and what it deliberately does not): the physics build pipeline compiles and
// the solver executes on a real WebGPU device headlessly, and `probeBuffer` reads a body's
// solver-owned pose back with finite values — the de-risking probe S1 of the
// shallot-headless-verifiability unit. It is NOT a correctness oracle: lavapipe is explicitly
// non-conformant/testing-only, and per-step math parity against the f64 CPU oracle stays in
// `*.oracle.ts` (CPU) + the gym `pile` scenario (real GPU). The assertions here are the
// closed-form rung only — finite poses, a falling body strictly below its authored start after
// a few fixed ticks (closed-form gravity direction, avbd.md "Gate where it's deterministic"
// rung 1 — no chaos accumulates in five ticks), and a grounded character at its closed-form
// rest height — never a number read back and tuned to.
//
// Why capacity 8192: the contact store sizes off the entity capacity (`eidCap × 3584 B`:
// PAIRS_PER_BODY 8 × CONTACTS_PER_PAIR 4 × CONTACT_VEC4 7 × 16 B per eid), and lavapipe's real
// adapter ceiling is the spec-default `maxStorageBufferBindingSize` of 128 MiB (measured, Mesa
// 25.3.4). 8192 × 3584 B ≈ 28 MiB fits with 4.5× headroom; the engine default 65536 (~235 MiB)
// does not — that, not a device capability, is what kept physics out of the headless roster
// (the exclusion comment this tier deleted from `conformance-roster.ts`). `build({ capacity })`
// is fixed at app construction (`engine/app`).
//
// Placement: `.tier.ts`, decided by the measured per-arm wall clock against the 5000 ms
// per-file cap (`tests/test-cap.ts`): cold (first run in a fresh checkout, no adapter shader
// cache) physics 1652 ms, character 95 ms, player 150 ms, 2.30 s for the file — 54% headroom
// under the cap, below the ~60% headroom bar that earns a `.test.ts` suffix. Warm (adapter shader cache
// present) the file runs ~0.7 s, but the placement decision must hold for the cold worst
// case, so it stays a by-path tier (the cap's own promotion move, 2). Re-run numbers are
// expected to vary with the host (lavapipe is CPU-executed). Run by path from the shallot
// root: `bun test ./packages/shallot/tests/avbd/headless.tier.ts`.
//
// Trigger cone (a by-path tier file's header is its registry, test-cap.ts:96): the transitive
// import cone of this file's arms — `src/standard/avbd/**` (plugin + PhysicsStep),
// `src/standard/physics/**` (substrate), `src/standard/character/**` + `src/standard/player/**`
// (sweep arms), `src/standard/{slab,mirror,input,render,transforms}/**` (declared dependencies
// the arms build with), and `src/engine/runtime/probe.ts` (the readback seam). Re-derive from
// the imports below if it drifts.
//
// The readback seam is the existing `probeBuffer` (`engine/runtime/probe.ts`) — it owns the
// encoder/submit/staging/mapAsync chain and validates usage + alignment, so this test adds zero
// plumbing. `Avbd.step.bodies` is the solver's SoA cols-buffer (`bodies[col * eidCap + eid]`,
// `COPY_SRC` by construction), and the B_* column indices are imported from `step.ts` so the
// probe offsets cannot drift from the solver's own layout.

import { describe, expect, test } from "bun:test";
import { Body, build, Compute, probeBuffer } from "../../src";
import { Avbd, AvbdPlugin } from "../../src/standard/avbd";
import { B_POS, B_QUAT, B_VELL } from "../../src/standard/avbd/step";
import { Character, CharacterPlugin } from "../../src/standard/character";
import { InputPlugin } from "../../src/standard/input";
import { MirrorPlugin } from "../../src/standard/mirror";
import { PlayerPlugin } from "../../src/standard/player";
import { RenderPlugin } from "../../src/standard/render";
import { SlabPlugin } from "../../src/standard/slab";
import { TransformsPlugin } from "../../src/standard/transforms";

/** the headless entity capacity — the contact store fits under lavapipe's 128 MiB binding ceiling. */
const CAPACITY = 8192;
/** fixed ticks to step before probing — five ticks of closed-form fall, no contact yet. */
const TICKS = 5;

/** the solver-owned pose of one body, read through the existing `probeBuffer` seam. */
async function probePose(eid: number): Promise<{ pos: number[]; quat: number[]; vel: number[] }> {
    const step = Avbd.step;
    const device = Compute.device;
    if (!step || !device) throw new Error("physics step or device missing after build");
    // the 32-byte read below assumes the quat column directly follows the pos column — guard the
    // contiguity instead of assuming it silently.
    if (B_QUAT !== B_POS + 1) throw new Error("pos/quat columns are no longer contiguous");
    // SoA cols-buffer: body eid's column c lives at `c * eidCap + eid`, one vec4 (16 B) per column.
    // pos (B_POS) + quat (B_QUAT) are contiguous; the linear velocity (B_VELL) is its own probe.
    const at = (col: number) => (col * step.eidCap + eid) * 16;
    const [pose, vel] = await Promise.all([
        probeBuffer(device, step.bodies, {
            offset: at(B_POS),
            size: 32,
            label: "headless-body-probe",
        }),
        probeBuffer(device, step.bodies, {
            offset: at(B_VELL),
            size: 16,
            label: "headless-vel-probe",
        }),
    ]);
    const f = new Float32Array(pose.bytes);
    const v = new Float32Array(vel.bytes);
    return {
        pos: [f[0], f[1], f[2]],
        quat: [f[4], f[5], f[6], f[7]],
        vel: [v[0], v[1], v[2]],
    };
}

/** every pose lane finite — the S1 bar: the build executed and the readback returned real values. */
function expectFinite(pose: { pos: number[]; quat: number[]; vel: number[] }): void {
    for (const lane of [...pose.pos, ...pose.quat, ...pose.vel]) {
        expect(Number.isFinite(lane), `expected a finite pose lane, got ${lane}`).toBe(true);
    }
}

describe("headless lavapipe physics (build + step + probeBuffer readback)", () => {
    test("AvbdPlugin builds, steps, and probes finite body poses at capacity 8192", async () => {
        const app = await build({
            plugins: [SlabPlugin, MirrorPlugin, AvbdPlugin],
            defaults: false,
            capacity: CAPACITY,
            scene: `<scene>
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
                <a body="pos: 0 6 0; half-extents: 0.6 0.6 0.6; mass: 1" />
            </scene>`,
        });
        expect(app.skipped).toEqual([]);
        expect(Avbd.step).not.toBeNull();
        expect(Avbd.step!.eidCap).toBe(CAPACITY);
        // the falling box is the scene's only mass > 0 body; the ground is static.
        const box = [...app.state.query([Body])].find((eid) => Body.mass.get(eid) > 0);
        expect(box).toBeDefined();
        // one state.step() = one fixed tick (dt defaults to Time.FIXED_DT); the first tick's draw-group
        // pack seeds the box's slot, later ticks integrate it — and 4.9 m above contact is far outside
        // the 0.04 speculative band, so every tick is closed-form free fall.
        for (let i = 0; i < TICKS; i++) app.state.step();
        const pose = await probePose(box!);
        expectFinite(pose);
        // closed-form gravity direction: after solved free-fall ticks the box sits strictly below its
        // authored start (Δy ≈ −0.5·g·(ticks·h)² ≈ −0.03 at 5 ticks, h = 1/60) — a band derived from
        // free-fall kinematics, never from the observed value. The floor is the same derivation's
        // other side: five ticks of free fall cover at most Δy = ½·g·(TICKS·h)² ≈ 0.035 m (the box
        // starts 4.9 m above contact, far outside the 0.04 speculative band, so no contact can have
        // accelerated it), so the pose must still exceed 6 − 0.035 ≈ 5.97 — assert > 5.9, the derived
        // floor with band margin, so a zero readback (wrong eid or a broken readback seam) reds.
        expect(pose.pos[1]).toBeLessThan(6);
        expect(pose.pos[1]).toBeGreaterThan(5.9);
        app.dispose();
    });

    test("CharacterPlugin sweeps headlessly at the same capacity", async () => {
        const app = await build({
            plugins: [SlabPlugin, MirrorPlugin, AvbdPlugin, CharacterPlugin],
            defaults: false,
            capacity: CAPACITY,
            scene: `<scene>
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
                <a
                    id="char"
                    body="pos: 0 3 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0"
                    character
                />
            </scene>`,
        });
        expect(app.skipped).toEqual([]);
        // the kinematic character: the sweep (fixed group, before the solve) reads candidate poses
        // through the body Mirror and writes its pose via setKinematic — the whole readback-bounded
        // coupling, headless.
        const chars = [...app.state.query([Character])];
        expect(chars.length).toBe(1);
        for (let i = 0; i < TICKS; i++) app.state.step();
        const pose = await probePose(chars[0]);
        expectFinite(pose);
        // the sweep applies the world gravity (−10, the plugin's configured GRAVITY) to an un-driven
        // character, so it too falls from its authored 3 m — derived from the sweep contract, not
        // tuned. The floor is the same derivation's other side: five ticks of un-driven fall cover at
        // most Δy = ½·g·(TICKS·h)² ≈ 0.035 m (the capsule's bottom at 2.1 sits 1.6 m above the ground
        // top at 0.5, so no depenetration or contact can have moved it), so the pose must still exceed
        // 3 − 0.035 ≈ 2.97 — assert > 2.9, the derived floor with band margin, so a zero readback reds.
        expect(pose.pos[1]).toBeLessThan(3);
        expect(pose.pos[1]).toBeGreaterThan(2.9);
        app.dispose();
    });

    test("PlayerPlugin composes headlessly at the same capacity", async () => {
        const app = await build({
            plugins: [
                SlabPlugin,
                TransformsPlugin,
                RenderPlugin,
                InputPlugin,
                MirrorPlugin,
                AvbdPlugin,
                CharacterPlugin,
                PlayerPlugin,
            ],
            defaults: false,
            capacity: CAPACITY,
            scene: `<scene>
                <a id="eye" camera transform="pos: 0 1.5 5" />
                <a
                    id="player"
                    body="pos: 0 1 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0"
                    character
                    player="camera: @eye"
                />
                <a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
            </scene>`,
        });
        expect(app.skipped).toEqual([]);
        const chars = [...app.state.query([Character])];
        expect(chars.length).toBe(1);
        for (let i = 0; i < TICKS; i++) app.state.step();
        const pose = await probePose(chars[0]);
        expectFinite(pose);
        // the player is an un-driven character grounded at its closed-form rest height: authored at
        // y = 1 the capsule's bottom (0.9 below center) embeds in the ground (top at 0.5), and the
        // sweep depenetrates to rest at center y = 0.5 + 0.6 + 0.3 = 1.4 — derived from the authored
        // geometry, never from the observed value; the 0.05 band absorbs the depenetration residual.
        expect(pose.pos[1]).toBeGreaterThan(1.35);
        expect(pose.pos[1]).toBeLessThan(1.45);
        app.dispose();
    });
});
