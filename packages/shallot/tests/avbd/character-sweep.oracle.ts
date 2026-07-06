import { describe, expect, test } from "bun:test";
// the shipped runtime CPU sweep (src) — the third tier beside this f64 oracle (the spec) and the GPU
// character pass; validated against the oracle here exactly as the GPU pass is gated against it in the gym.
import {
    type CharState,
    MAX_CHAR_CANDIDATES,
    type SweepBody,
    type SweepDiag,
    sweepCharacter,
} from "../../src/standard/character/sweep";
import { type Character, character, moveCharacter } from "./character";
import { boxHull } from "./hull";
import { length, type Quat, scale, sub, type Vec3 } from "./math";
import { type Body, body, capsule, hull as hullBody, massOf, ShapeKind } from "./rigid";

// CPU character sweep validation (roadmap "Physics — CPU character controller") — the runtime collide-
// and-slide (`standard/character/sweep.ts`) is a faithful f32-tier port of the f64 oracle `moveCharacter`
// (character.ts). Two gates, the physics.md discipline:
//   • per-step parity (single-step INJECTION): drive the oracle through each behavioral trajectory, and at
//     every frame snapshot its pre-step state into the sweep, step BOTH once, compare pose / velocity /
//     grounded. Memoryless ⇒ tight (no chaos accumulation) — the single-step-exact gate, the GPU == oracle
//     shape the gym uses. Covers penetrating-and-walking / at-rest / airborne / per-character-gravity /
//     slope / step / wall / carry / push / hull-ground.
//   • behavioral semantics (FREE-RUN): run the sweep ALONE and assert the same integrated invariants the
//     oracle's own tests assert (slope hold-vs-slide, step-up, wall-stop, single-jump, coyote, buffer) — the
//     timer evolution + slide/step/jump semantics emerge from the sweep, not just the per-step math.
//   • the gather: cull == brute (the sphere cull is a contact-set-preserving superset), + the diagnostics.

const G = -10;
const DT = 1 / 60;
const HALF_H = 0.5; // capsule core half-height
const RADIUS = 0.3; // capsule radius → resting offset = HALF_H + RADIUS = 0.8 above a surface
const REST_OFFSET = HALF_H + RADIUS;
const restY = 0.5 + REST_OFFSET; // 1.3 on the floor
const MAX_SLOPE = 45; // 30° holds, 60° slides
const JUMP = 5; // apex = JUMP²/(2|G|) = 1.25 m above the launch
// the gym's f32(GPU)-vs-f64(oracle) single-step tolerances. The CPU sweep is f64-arithmetic over f32-stored
// inputs, so its error sits well under these (the gate's job is the upper bound, not the actual magnitude).
const POS_TOL = 1e-3;
const VEL_TOL = 6e-2; // = POS_TOL / DT (realized velocity is the swept displacement over dt)

// the runtime stores Body + the character pose on f32 slabs; quantize the slab-sourced fields so the
// validation exercises the precision tier the roadmap names ("f32, reading runtime Body"). The controller
// state (velocity, jump timers, grounded) lives in the CPU CharState, not a slab, so it's carried at f64.
const _f = new Float32Array(1);
const f1 = (x: number): number => {
    _f[0] = x;
    return _f[0];
};
const q3 = (v: readonly number[]): Vec3 => [f1(v[0]), f1(v[1]), f1(v[2])];
const q4 = (v: readonly number[]): Quat => [f1(v[0]), f1(v[1]), f1(v[2]), f1(v[3])];
const dist = (a: Vec3, b: Vec3): number => length(sub(a, b));

// oracle Body → the runtime-shaped candidate the sweep reads (poses the runtime gets off the slab / Mirror).
// `half` is the core half-extents (size/2); a hull carries its registry geometry (the oracle's lacks `name`).
function sweepBody(b: Body): SweepBody {
    return {
        shape: b.shape,
        pos: q3(b.posLin),
        quat: q4(b.posAng),
        half: q3(scale(b.size, 0.5)),
        radius: f1(b.roundRadius),
        hull: b.shape === ShapeKind.Hull && b.hull ? { name: "test", ...b.hull } : undefined,
        vel: q3(b.velLin),
    };
}

// oracle Character → the CPU controller state (the slab-sourced pose quantized, the controller state at f64)
function charState(o: Character): CharState {
    return {
        pos: q3(o.body.posLin),
        quat: q4(o.body.posAng),
        half: f1(o.body.size[1] / 2),
        radius: f1(o.body.roundRadius),
        maxSlopeCos: o.maxSlopeCos,
        jumpSpeed: o.jumpSpeed,
        vel: [...o.vel] as Vec3,
        realizedVel: [0, 0, 0],
        grounded: o.grounded,
        groundNormal: [...o.groundNormal] as Vec3,
        coyote: o.coyote,
        buffer: o.buffer,
    };
}

// a large static ground/slope box (top face REST_OFFSET below the capsule centre at rest)
const ground = (pos: Vec3, quat: Quat = [0, 0, 0, 1]): Body =>
    body([40, 1, 40], 0, 0.8, pos, [0, 0, 0], quat);
const qz = (rad: number): Quat => [0, 0, Math.sin(rad / 2), Math.cos(rad / 2)];
const idle = (): Vec3 => [0, 0, 0];
const noJump = (): boolean => false;

// ── per-step parity: single-step injection along the oracle's trajectory ───────────────────────────────

interface LockScene {
    name: string;
    ch: Character;
    statics: Body[];
    push: Body[];
    gravity: number;
    input: (f: number) => Vec3;
    jump: (f: number) => boolean;
    pre?: (f: number) => void; // advance a moving platform before the step
    frames: number;
}

function lockstep(s: LockScene): { pos: number; vel: number; push: number; gm: number } {
    let mp = 0;
    let mv = 0;
    let mpush = 0;
    let gm = 0;
    for (let f = 0; f < s.frames; f++) {
        s.pre?.(f);
        const input = s.input(f);
        const jp = s.jump(f);
        // snapshot the oracle's PRE-step state into the sweep (build sweep bodies before moveCharacter
        // mutates the oracle), step both once from that identical state, compare the outputs.
        const sc = charState(s.ch);
        const ss = s.statics.map(sweepBody);
        const sp = s.push.map(sweepBody);
        sweepCharacter(sc, input, ss, s.gravity, DT, jp, sp);
        moveCharacter(s.ch, input, s.statics, s.gravity, DT, jp, s.push);
        mp = Math.max(mp, dist(sc.pos, s.ch.body.posLin as Vec3));
        mv = Math.max(mv, dist(sc.realizedVel, s.ch.body.velLin as Vec3));
        if (sc.grounded !== s.ch.grounded) gm++;
        for (let i = 0; i < sp.length; i++)
            mpush = Math.max(mpush, dist(sp[i].vel, s.push[i].velLin as Vec3));
    }
    return { pos: mp, vel: mv, push: mpush, gm };
}

function scenes(): LockScene[] {
    const out: LockScene[] = [];
    // drop-to-rest — airborne → penetrating → at-rest, with the grounding transition
    out.push({
        name: "drop",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 3, 0])),
        statics: [ground([0, 0, 0])],
        push: [],
        gravity: G,
        input: idle,
        jump: noJump,
        frames: 200,
    });
    // slope 30° (walkable, holds) + 60° (too steep, slides)
    for (const deg of [30, 60]) {
        const a = (deg * Math.PI) / 180;
        const n: Vec3 = [-Math.sin(a), Math.cos(a), 0];
        const top: Vec3 = [n[0] * (0.5 + REST_OFFSET), 5 + n[1] * (0.5 + REST_OFFSET) + 1.5, 0];
        out.push({
            name: `slope${deg}`,
            ch: character(capsule(HALF_H, RADIUS, 0, 0.8, top), MAX_SLOPE),
            statics: [ground([0, 5, 0], qz(a))],
            push: [],
            gravity: G,
            input: idle,
            jump: noJump,
            frames: 150,
        });
    }
    // step-up — the rounded bottom climbs a sub-radius step onto the plateau
    out.push({
        name: "step-up",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])),
        statics: [ground([0, 0, 0]), body([40, 0.7, 4], 0, 0.8, [23, 0.35, 0])],
        push: [],
        gravity: G,
        input: () => [2, 0, 0],
        jump: noJump,
        frames: 200,
    });
    // wall-stop — a tall wall blocks the sweep
    out.push({
        name: "wall-stop",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])),
        statics: [ground([0, 0, 0]), body([2, 8, 2], 0, 0.8, [3, 4, 0])],
        push: [],
        gravity: G,
        input: () => [2, 0, 0],
        jump: noJump,
        frames: 200,
    });
    // moving-platform carry (horizontal) — the platform pose advances each frame, its velocity carries the char
    {
        const V = 1.5;
        const plat = body([10, 1, 10], 0, 0.8, [0, 0, 0], [V, 0, 0]);
        out.push({
            name: "platform-h",
            ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0.5 + REST_OFFSET, 0])),
            statics: [plat],
            push: [],
            gravity: G,
            input: idle,
            jump: noJump,
            pre: () => {
                plat.posLin[0] += V * DT;
            },
            frames: 90,
        });
    }
    // descending platform — the carry's vertical realized velocity (the snap alone leaves it 0)
    {
        const V = 1;
        const plat = body([10, 1, 10], 0, 0.8, [0, 0, 0], [0, -V, 0]);
        out.push({
            name: "platform-down",
            ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0.5 + REST_OFFSET, 0])),
            statics: [plat],
            push: [],
            gravity: G,
            input: idle,
            jump: noJump,
            pre: () => {
                plat.posLin[1] -= V * DT;
            },
            frames: 30,
        });
    }
    // walk into a held dynamic box — side block + the full-speed velocity-transfer push
    out.push({
        name: "walk-into-box",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.5, [0, 1.3, 0])),
        statics: [ground([0, 0, 0])],
        push: [body([1, 1, 1], massOf([1, 1, 1], 1), 0.3, [1.5, 1.0, 0])],
        gravity: G,
        input: () => [3, 0, 0],
        jump: noJump,
        frames: 120,
    });
    // stand on a held dynamic box — a walkable contact on a dynamic supports the char (push down-cancelled)
    out.push({
        name: "stand-on-box",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 4, 0])),
        statics: [ground([0, 0, 0])],
        push: [body([1, 1, 1], massOf([1, 1, 1], 1), 0.8, [0, 1.0, 0])],
        gravity: G,
        input: idle,
        jump: noJump,
        frames: 200,
    });
    // spam jump — per-step the launch / arc / landing (single jump per landing, the controller gates it)
    out.push({
        name: "spam-jump",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]), 50, JUMP),
        statics: [ground([0, 0, 0])],
        push: [],
        gravity: G,
        input: idle,
        jump: () => true,
        frames: 300,
    });
    // per-character gravity — the char falls at -50, not the world -10
    out.push({
        name: "per-char-gravity",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 2.5, 0])),
        statics: [ground([0, 0, 0])],
        push: [],
        gravity: -50,
        input: idle,
        jump: noJump,
        frames: 30,
    });
    // hull ground — a box-as-hull static, exercising the convex closestPointHull path the box clamp skips
    out.push({
        name: "hull-ground",
        ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 1.2, 0])),
        statics: [hullBody(boxHull([40, 1, 40]), 0, 0.8, [0, 0, 0])],
        push: [],
        gravity: G,
        input: () => [2, 0, 0],
        jump: noJump,
        frames: 60,
    });
    return out;
}

describe("CPU character sweep — per-step parity vs the f64 oracle (single-step injection)", () => {
    test("pose / velocity / grounded match across the behavioral scenes", () => {
        let mp = 0;
        let mv = 0;
        let mpush = 0;
        let gm = 0;
        const log: string[] = [];
        for (const s of scenes()) {
            const r = lockstep(s);
            mp = Math.max(mp, r.pos);
            mv = Math.max(mv, r.vel);
            mpush = Math.max(mpush, r.push);
            gm += r.gm;
            log.push(
                `${s.name} pos ${r.pos.toExponential(1)}/vel ${r.vel.toExponential(1)}${r.push > 0 ? `/push ${r.push.toExponential(1)}` : ""}${r.gm ? ` GM×${r.gm}` : ""}`,
            );
        }
        console.log(`[char-sweep] per-step parity — ${log.join(" · ")}`);
        expect(gm).toBe(0); // grounded matches the oracle every frame
        expect(mp).toBeLessThan(POS_TOL);
        expect(mv).toBeLessThan(VEL_TOL);
        expect(mpush).toBeLessThan(VEL_TOL);
    });
});

// ── behavioral semantics: the sweep run ALONE reproduces the oracle's slide / step / jump invariants ───

describe("CPU character sweep — behavioral semantics (free-run)", () => {
    test("drop-to-rest: settles on the floor, grounded, no residual jitter", () => {
        const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 3, 0])));
        const ss = [sweepBody(ground([0, 0, 0]))];
        for (let f = 0; f < 200; f++) sweepCharacter(sc, [0, 0, 0], ss, G, DT);
        console.log(
            `[char-sweep] drop-to-rest y ${sc.pos[1].toFixed(4)} (rest ${restY}), grounded ${sc.grounded}`,
        );
        expect(Math.abs(sc.pos[1] - restY)).toBeLessThan(0.03);
        expect(sc.grounded).toBe(true);
        expect(length(sc.realizedVel)).toBeLessThan(1e-3);
    });

    test("slope: 30° holds (walkable), 60° slides (too steep)", () => {
        const slide = (deg: number): number => {
            const a = (deg * Math.PI) / 180;
            const n: Vec3 = [-Math.sin(a), Math.cos(a), 0];
            const top: Vec3 = [n[0] * (0.5 + REST_OFFSET), 5 + n[1] * (0.5 + REST_OFFSET) + 1.5, 0];
            const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, top), MAX_SLOPE));
            const ss = [sweepBody(ground([0, 5, 0], qz(a)))];
            for (let f = 0; f < 60; f++) sweepCharacter(sc, [0, 0, 0], ss, G, DT); // land + settle
            const landed: Vec3 = [...sc.pos] as Vec3;
            for (let f = 0; f < 90; f++) sweepCharacter(sc, [0, 0, 0], ss, G, DT); // measure window
            return length(sub(sc.pos, landed));
        };
        const shallow = slide(30);
        const steep = slide(60);
        console.log(
            `[char-sweep] slope — 30° ${shallow.toFixed(3)} m (holds), 60° ${steep.toFixed(3)} m (slides)`,
        );
        expect(shallow).toBeLessThan(0.1);
        expect(steep).toBeGreaterThan(1.0);
        expect(steep).toBeGreaterThan(shallow * 10);
    });

    test("step-up climbs a sub-radius step; a tall wall stops the char (bounded, no jitter)", () => {
        // step-up onto the plateau (near face x = 3)
        const climber = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])));
        const stepStatics = [
            sweepBody(ground([0, 0, 0])),
            sweepBody(body([40, 0.7, 4], 0, 0.8, [23, 0.35, 0])),
        ];
        for (let f = 0; f < 200; f++) sweepCharacter(climber, [2, 0, 0], stepStatics, G, DT);
        const stepTopY = 0.7 + REST_OFFSET; // 1.5 on the plateau
        console.log(
            `[char-sweep] step-up — pos ${climber.pos.map((v) => v.toFixed(2)).join(",")} (top ~${stepTopY})`,
        );
        expect(climber.pos[0]).toBeGreaterThan(4); // climbed + kept walking onto the plateau
        expect(Math.abs(climber.pos[1] - stepTopY)).toBeLessThan(0.06);

        // tall wall (near face x = 2) — pushed back, bounded, no jitter
        const blocked = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])));
        const wallStatics = [
            sweepBody(ground([0, 0, 0])),
            sweepBody(body([2, 8, 2], 0, 0.8, [3, 4, 0])),
        ];
        let minX = Infinity;
        let maxX = -Infinity;
        for (let f = 0; f < 200; f++) {
            sweepCharacter(blocked, [2, 0, 0], wallStatics, G, DT);
            if (f >= 150) {
                minX = Math.min(minX, blocked.pos[0]);
                maxX = Math.max(maxX, blocked.pos[0]);
            }
        }
        console.log(
            `[char-sweep] wall-stop — final x ${blocked.pos[0].toFixed(3)} (surface ~${2 - RADIUS}), drift ${(maxX - minX).toExponential(2)}`,
        );
        expect(blocked.pos[0]).toBeLessThan(2); // never tunnels past the wall face
        expect(blocked.pos[0]).toBeGreaterThan(2 - RADIUS - 0.1); // stops at the surface
        expect(maxX - minX).toBeLessThan(1e-3); // no jitter
        expect(Math.abs(blocked.pos[1] - restY)).toBeLessThan(0.05); // still on the floor (didn't climb)
    });

    test("jump: spam → single jump per landing, bounded apex (no double-jump)", () => {
        const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]), 50, JUMP));
        const ss = [sweepBody(ground([0, 0, 0]))];
        let maxY = Number.NEGATIVE_INFINITY;
        let jumps = 0;
        for (let f = 0; f < 300; f++) {
            sweepCharacter(sc, [0, 0, 0], ss, G, DT, true); // jump held every frame
            maxY = Math.max(maxY, sc.pos[1]);
            if (sc.vel[1] === JUMP) jumps++; // a launch frame sets vel.y exactly to jumpSpeed
        }
        const apex = restY + (JUMP * JUMP) / (2 * Math.abs(G)); // 2.55
        console.log(
            `[char-sweep] spam-jump — ${jumps} jumps/300f, maxY ${maxY.toFixed(3)} (apex ${apex.toFixed(2)})`,
        );
        expect(jumps).toBeGreaterThan(1); // jumped repeatedly (on each landing)
        expect(jumps).toBeLessThan(10); // NOT every frame — single jump per cycle
        expect(maxY).toBeLessThan(apex + 0.4); // bounded near one jump's apex
        expect(maxY).toBeGreaterThan(restY + 0.8); // clearly left the ground
    });

    test("coyote: a jump pressed just after walking off a ledge still fires", () => {
        const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0.5, restY, 0]), 50, JUMP));
        const ss = [sweepBody(body([2, 1, 4], 0, 0.8, [0, 0, 0]))]; // platform x ∈ [-1, 1]
        let leftAt = -1;
        let coyoteJumped = false;
        for (let f = 0; f < 120; f++) {
            const wasGrounded = sc.grounded;
            const press = leftAt >= 0 && f === leftAt + 1; // first airborne frame after the ledge
            sweepCharacter(sc, [3, 0, 0], ss, G, DT, press);
            if (wasGrounded && !sc.grounded && leftAt < 0) leftAt = f;
            if (press && sc.vel[1] === JUMP) coyoteJumped = true;
        }
        console.log(`[char-sweep] coyote — left at frame ${leftAt}, coyote jump ${coyoteJumped}`);
        expect(leftAt).toBeGreaterThan(0); // walked off the ledge (grounded lapsed)
        expect(coyoteJumped).toBe(true); // the jump fired within the coyote window despite being airborne
    });

    test("buffer: a jump pressed just before landing fires on touchdown", () => {
        const sc = charState(
            character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY + 0.4, 0]), 50, JUMP),
        );
        const ss = [sweepBody(ground([0, 0, 0]))];
        let buffered = false;
        let bufferJumped = false;
        for (let f = 0; f < 120; f++) {
            const press = !buffered && !sc.grounded && sc.vel[1] < 0 && sc.pos[1] < restY + 0.15;
            if (press) buffered = true;
            sweepCharacter(sc, [0, 0, 0], ss, G, DT, press);
            if (buffered && sc.vel[1] === JUMP) bufferJumped = true;
        }
        console.log(
            `[char-sweep] buffer — pressed-before-landing ${buffered}, fired ${bufferJumped}`,
        );
        expect(buffered).toBe(true); // the press happened while airborne + descending
        expect(bufferJumped).toBe(true); // fired on/just after landing (the buffer carried it)
    });
});

// ── gather: the sphere cull is a contact-set-preserving superset ───────────────────────────────────────

describe("CPU character sweep — gather (cull == brute, diagnostics)", () => {
    // far filler statics in a ring — gathered by neither phase, probed only by the brute run
    const fillers = (n: number, r: number): Body[] => {
        const out: Body[] = [];
        for (let i = 0; i < n; i++) {
            const a = (i / n) * Math.PI * 2;
            out.push(body([1, 1, 1], 0, 0.8, [Math.cos(a) * r, 1 + (i % 5), Math.sin(a) * r]));
        }
        return out;
    };

    // run a scene culled and brute, frame-locked, returning the max pose+velocity divergence (expect 0 exact)
    const divergence = (
        mk: () => { sc: CharState; ss: SweepBody[]; sp: SweepBody[]; input: Vec3 },
        frames: number,
        jumpAt = -1,
    ): number => {
        const a = mk();
        const b = mk();
        let max = 0;
        for (let f = 0; f < frames; f++) {
            sweepCharacter(a.sc, a.input, a.ss, G, DT, f === jumpAt, a.sp);
            sweepCharacter(b.sc, b.input, b.ss, G, DT, f === jumpAt, b.sp, { cull: false });
            max = Math.max(max, dist(a.sc.pos, b.sc.pos), dist(a.sc.realizedVel, b.sc.realizedVel));
            for (let i = 0; i < a.sp.length; i++)
                max = Math.max(max, dist(a.sp[i].vel, b.sp[i].vel));
        }
        return max;
    };

    test("cull == brute, bit-exact, across the behavioral scene shapes", () => {
        const stepUp = divergence(
            () => ({
                sc: charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]))),
                ss: [
                    sweepBody(ground([0, 0, 0])),
                    sweepBody(body([40, 0.7, 4], 0, 0.8, [23, 0.35, 0])),
                    ...fillers(24, 30).map(sweepBody),
                ],
                sp: [],
                input: [2, 0, 0] as Vec3,
            }),
            200,
        );
        const pushScene = divergence(
            () => ({
                sc: charState(character(capsule(HALF_H, RADIUS, 0, 0.5, [0, restY, 0]), 50, JUMP)),
                ss: [sweepBody(ground([0, 0, 0])), ...fillers(24, 25).map(sweepBody)],
                sp: [
                    sweepBody(
                        body([0.8, 0.8, 0.8], massOf([0.8, 0.8, 0.8], 0.5), 0.3, [1.2, 0.9, 0]),
                    ),
                ],
                input: [3, 0, 0] as Vec3,
            }),
            120,
            30,
        );
        console.log(
            `[char-sweep] cull divergence — step-up ${stepUp.toExponential(1)}, push ${pushScene.toExponential(1)}`,
        );
        expect(stepUp).toBe(0); // bit-identical — the cull never changes the contact set
        expect(pushScene).toBe(0);
    });

    test("far bodies culled / overflow flagged / guard tripped", () => {
        const diag = (): SweepDiag => ({ candidates: 0, overflow: false, guard: false });

        // far ring fully culled — only the floor survives
        {
            const d = diag();
            const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])));
            sweepCharacter(
                sc,
                [1, 0, 0],
                [sweepBody(ground([0, 0, 0])), ...fillers(500, 30).map(sweepBody)],
                G,
                DT,
                false,
                [],
                { diag: d },
            );
            console.log(`[char-sweep] cull — candidates ${d.candidates} of 501`);
            expect(d.candidates).toBe(1);
            expect(d.overflow).toBe(false);
        }

        // > 64 near statics — overflow flagged loudly (keeps the first 64 in scan order)
        {
            const d = diag();
            const near: Body[] = [];
            for (let i = 0; i < 100; i++) {
                const a = i * 2.4;
                const r = (i % 10) * 0.12;
                near.push(
                    body([0.2, 0.2, 0.2], 0, 0.8, [
                        Math.cos(a) * r,
                        0.6 + (i % 6) * 0.12,
                        Math.sin(a) * r,
                    ]),
                );
            }
            const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])));
            sweepCharacter(sc, [0.5, 0, 0], near.map(sweepBody), G, DT, false, [], { diag: d });
            console.log(`[char-sweep] cull — overflow ${d.overflow}, candidates ${d.candidates}`);
            expect(d.overflow).toBe(true);
            expect(d.candidates).toBe(MAX_CHAR_CANDIDATES);
        }

        // spawn deep inside geometry — the displacement guard trips, the pose stays finite
        {
            const d = diag();
            const sc = charState(character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0, 0])));
            sweepCharacter(
                sc,
                [0, 0, 0],
                [sweepBody(body([4, 4, 4], 0, 0.8, [0, 0, 0]))],
                G,
                DT,
                false,
                [],
                { diag: d },
            );
            console.log(
                `[char-sweep] guard — tripped ${d.guard}, pos ${sc.pos.map((v) => v.toFixed(2)).join(",")}`,
            );
            expect(d.guard).toBe(true);
            expect(sc.pos.every(Number.isFinite)).toBe(true);
        }
    });
});
