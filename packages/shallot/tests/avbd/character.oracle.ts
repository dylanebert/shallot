import { describe, expect, test } from "bun:test";
import { character, type MoveDiag, moveCharacter } from "./character";
import { length, type Quat, sub, type Vec3 } from "./math";
import { type Body, body, capsule, massOf } from "./rigid";
import { makeSolver, step } from "./solver";

// The kinematic character controller (roadmap §6.4) — closed-form gates on the collide-and-slide sweep
// (character.ts) plus the integration with the AVBD solver + the kinematic-pushing fix (manifold.ts
// updateDual). No AVBD/webphysics reference covers a character, so these are first-principles invariants:
// a capsule under gravity rests on flat ground, holds on a walkable slope but slides on a too-steep one,
// climbs a sub-radius step on its rounded bottom but is stopped by a tall wall (bounded, no jitter), and —
// driven into a static wall through the full solver — never escalates the all-static contacts (the §6.4
// fix) and leaves a nearby dynamic body undisturbed. f64; the runtime CPU sweep reproduces it.

const G = -10;
const DT = 1 / 60;
const HALF_H = 0.5; // capsule core half-height
const RADIUS = 0.3; // capsule radius → resting offset = HALF_H + RADIUS = 0.8 above a surface
const REST_OFFSET = HALF_H + RADIUS;

const qz = (rad: number): Quat => [0, 0, Math.sin(rad / 2), Math.cos(rad / 2)];

// a large static ground/slope box (full-width size; top face REST_OFFSET below the capsule centre at rest)
const ground = (pos: Vec3, quat: Quat = [0, 0, 0, 1]) =>
    body([40, 1, 40], 0, 0.8, pos, [0, 0, 0], quat);

describe("AVBD character — collide-and-slide sweep", () => {
    test("drop-to-rest: a capsule falls and settles with its surface on flat ground, grounded", () => {
        const floor = ground([0, 0, 0]); // top at y = 0.5
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 3, 0]));
        for (let f = 0; f < 200; f++) moveCharacter(ch, [0, 0, 0], [floor], G, DT);

        const restY = 0.5 + REST_OFFSET; // 1.3 — capsule surface touching the floor top
        console.log(
            `[character] drop-to-rest y ${ch.body.posLin[1].toFixed(4)} (expect ${restY}), grounded ${ch.grounded}, speed ${length(ch.body.velLin).toExponential(2)}`,
        );
        expect(ch.body.posLin.every(Number.isFinite)).toBe(true);
        expect(Math.abs(ch.body.posLin[1] - restY)).toBeLessThan(0.03); // settles at the surface, no sink
        expect(ch.grounded).toBe(true);
        expect(length(ch.body.velLin)).toBeLessThan(1e-3); // at rest — no residual jitter
    });

    test("slope-limit: holds on a walkable slope (30° < 45° cutoff), slides on a too-steep one (60°)", () => {
        // maxSlope 45° → cos 0.707. A 30° slope (normal.y = cos30 = 0.866 > cutoff) is walkable: the char
        // grounds, gravity stops accumulating, it holds. A 60° slope (normal.y = cos60 = 0.5 < cutoff) is
        // not walkable: the char never grounds, gravity keeps building, it slides down-slope. The slide is
        // measured AFTER landing (warm up, record, then measure) — the discriminator is stuck vs sliding.
        const slide = (deg: number): number => {
            const a = (deg * Math.PI) / 180;
            const n: Vec3 = [-Math.sin(a), Math.cos(a), 0]; // the slope's top-face normal
            const slope = ground([0, 5, 0], qz(a));
            // drop onto the top surface above the slope centre
            const top: Vec3 = [n[0] * (0.5 + REST_OFFSET), 5 + n[1] * (0.5 + REST_OFFSET) + 1.5, 0];
            const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, top), 45);
            for (let f = 0; f < 60; f++) moveCharacter(ch, [0, 0, 0], [slope], G, DT); // land + settle
            const landed: Vec3 = [...ch.body.posLin] as Vec3;
            for (let f = 0; f < 90; f++) moveCharacter(ch, [0, 0, 0], [slope], G, DT); // measure window
            return length(sub(ch.body.posLin, landed));
        };
        const shallow = slide(30);
        const steep = slide(60);
        console.log(
            `[character] slope slide — 30° ${shallow.toFixed(3)} m (holds), 60° ${steep.toFixed(3)} m (slides)`,
        );
        expect(shallow).toBeLessThan(0.1); // walkable: holds
        expect(steep).toBeGreaterThan(1.0); // too steep: slides far
        expect(steep).toBeGreaterThan(shallow * 10);
    });

    test("step-up: the rounded bottom climbs a sub-radius step but a tall wall stops it (bounded, no jitter)", () => {
        const floor = ground([0, 0, 0]); // top at y = 0.5
        const restY = 0.5 + REST_OFFSET; // 1.3 on the floor

        // a 0.2 m step (< the 0.3 radius) up onto a long plateau (near face at x = 3) — the capsule's
        // bottom hemisphere catches the top edge, whose closest-point normal tilts up, lifting it over (the
        // rounded-bottom free step-up). The plateau extends far in +x so the char stays on top after climbing.
        const step = body([40, 0.7, 4], 0, 0.8, [23, 0.35, 0]); // x ∈ [3, 43], top at y = 0.7
        const stepTopY = 0.7 + REST_OFFSET; // 1.5 standing on the plateau
        const climber = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]));
        for (let f = 0; f < 200; f++) moveCharacter(climber, [2, 0, 0], [floor, step], G, DT);
        console.log(
            `[character] step-up — pos ${climber.body.posLin.map((v) => v.toFixed(2)).join(",")} (climbed onto the plateau, y ~${stepTopY})`,
        );
        expect(climber.body.posLin.every(Number.isFinite)).toBe(true);
        expect(climber.body.posLin[0]).toBeGreaterThan(4); // climbed the step and kept walking onto the plateau
        expect(Math.abs(climber.body.posLin[1] - stepTopY)).toBeLessThan(0.06); // standing on top, not the floor

        // a tall wall (top y = 4, near face x = 2) — the cylinder hits the vertical face, normal −x, so the
        // char is pushed straight back: it can't climb and can't tunnel. Pushed for 200 frames it converges
        // to the surface (last 50 frames don't drift — bounded, no jitter).
        const wall = body([2, 8, 2], 0, 0.8, [3, 4, 0]); // near face x = 2
        const blocked = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]));
        let minX = Infinity;
        let maxX = -Infinity;
        for (let f = 0; f < 200; f++) {
            moveCharacter(blocked, [2, 0, 0], [floor, wall], G, DT);
            if (f >= 150) {
                minX = Math.min(minX, blocked.body.posLin[0]);
                maxX = Math.max(maxX, blocked.body.posLin[0]);
            }
        }
        console.log(
            `[character] wall-stop — final x ${blocked.body.posLin[0].toFixed(3)} (face 2, surface ~${2 - RADIUS}), last-50 drift ${(maxX - minX).toExponential(2)}, y ${blocked.body.posLin[1].toFixed(2)}`,
        );
        expect(blocked.body.posLin.every(Number.isFinite)).toBe(true);
        expect(blocked.body.posLin[0]).toBeLessThan(2); // never tunnels past the wall face
        expect(blocked.body.posLin[0]).toBeGreaterThan(2 - RADIUS - 0.1); // stops at the surface, not pushed away
        expect(maxX - minX).toBeLessThan(1e-3); // no jitter — the contact is bounded, settled
        expect(Math.abs(blocked.body.posLin[1] - restY)).toBeLessThan(0.05); // still on the floor (didn't climb)
    });
});

describe("AVBD character — the kinematic-pushing fix through the full solver", () => {
    test("a kinematic capsule driven into a static wall stays bounded (no escalation, no tunnel, no jitter)", () => {
        // The §6.4 headline, end to end: the character is a mass ≤ 0 capsule in the solver's body list, its
        // controller driving it straight into a static wall. The controller's sweep stops it at the wall
        // surface; the AVBD then generates the char–wall AND char–floor contacts — both all-static (mass ≤ 0
        // on both sides), the exact contacts the legacy stack escalated. The dual fix keeps their penalty at
        // the seed, so the run is bounded, the char never tunnels, and it settles without jitter. A dynamic
        // box resting on the floor nearby must stay undisturbed (the escalation can't leak energy into the sim).
        const floor = ground([0, 0, 0]);
        const wall = body([1, 8, 8], 0, 0.8, [4, 1, 0]); // static, near face at x = 3.5
        const box = body([1, 1, 1], massOf([1, 1, 1], 1), 0.5, [-3, 1.0, 0]); // dynamic, at rest, away from the char
        const cap = capsule(HALF_H, RADIUS, 0, 0.5, [0, 1.3, 0]); // kinematic, on the floor
        const ch = character(cap);
        // creation order sets the index; the char (highest) is bodyA in its pairs
        const s = makeSolver([floor, wall, box, cap], { layer: "warmstart", gravity: G });

        let maxStaticPen = 0;
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        for (let f = 0; f < 200; f++) {
            moveCharacter(ch, [2, 0, 0], [floor, wall], G, DT); // drive +x into the wall, sweeping both statics
            step(s);
            for (const m of s.manifolds.values())
                if (m.a.mass <= 0 && m.b.mass <= 0)
                    for (const c of m.contacts) maxStaticPen = Math.max(maxStaticPen, c.penalty[0]);
            if (f >= 150) {
                minX = Math.min(minX, cap.posLin[0]);
                maxX = Math.max(maxX, cap.posLin[0]);
            }
        }
        console.log(
            `[character] full-solver wall — char x ${cap.posLin[0].toFixed(3)} (face 3.5, surface ~${3.5 - RADIUS}), ` +
                `last-50 drift ${(maxX - minX).toExponential(2)}, all-static max penalty ${maxStaticPen.toExponential(2)}, ` +
                `box ${box.posLin.map((v) => v.toFixed(2)).join(",")}`,
        );
        expect(cap.posLin.every(Number.isFinite)).toBe(true);
        expect(cap.posLin[0]).toBeLessThan(3.5); // never tunnels through the wall
        expect(cap.posLin[0]).toBeGreaterThan(3.5 - RADIUS - 0.1); // stopped at the surface, not bounced off
        expect(maxX - minX).toBeLessThan(1e-3); // bounded + settled, no jitter
        expect(maxStaticPen).toBeLessThan(2); // the char–wall + char–floor contacts never ramped (the §6.4 fix)
        // the dynamic box, away from the char, is undisturbed — the escalation didn't leak into the sim
        expect(box.posLin.every(Number.isFinite)).toBe(true);
        expect(Math.hypot(box.posLin[0] - -3, box.posLin[1] - 1.0)).toBeLessThan(0.1);
        expect(length(box.velLin)).toBeLessThan(0.05);
    });
});

describe("AVBD character — moving-platform carry", () => {
    // the char riding a translating kinematic platform tracks it (legacy charMove += groundVelocity): the
    // controller reads the supporting body's velocity and adds it to its own motion. A true static has
    // velLin 0, so a flat floor never carries — only a body the scene is actively moving.
    test("rides a horizontally translating platform (tracks its x)", () => {
        const V = 1.5; // platform speed m/s
        // a wide flat kinematic platform (mass 0) moving +x; top at y = 0.5. The scene advances its pose
        // each step (as a PlatformSystem does), and its velLin carries the rider.
        const platform = body([10, 1, 10], 0, 0.8, [0, 0, 0], [V, 0, 0]);
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0.5 + REST_OFFSET, 0]));
        for (let f = 0; f < 90; f++) {
            platform.posLin[0] += V * DT; // the kinematic platform moves first (fixed, before the sweep)
            moveCharacter(ch, [0, 0, 0], [platform], G, DT);
        }
        console.log(
            `[character] platform-carry — char x ${ch.body.posLin[0].toFixed(3)}, platform x ${platform.posLin[0].toFixed(3)}`,
        );
        expect(ch.grounded).toBe(true);
        // tracks the platform: without carry the char stays at x≈0 while the platform slides to ~2.25
        expect(Math.abs(ch.body.posLin[0] - platform.posLin[0])).toBeLessThan(0.1);
    });

    test("a descending platform produces the carry's realized velocity (snap alone leaves it 0)", () => {
        // vertical POSITION tracking is masked by the ground snap (within its band the snap glues the char
        // down each step even without carry), so the carry's isolable vertical effect is the REALIZED velocity
        // it writes — (swept − start)/dt, computed before the snap. Without carry that's 0 (the char looks
        // stationary while the snap relocates it); with carry it's −V, the value the broadphase pad + the carry
        // of riders ON the char read. (Horizontal tracking, above, is where the snap can't mask the carry.)
        const V = 1.0;
        const platform = body([10, 1, 10], 0, 0.8, [0, 0, 0], [0, -V, 0]);
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0.5 + REST_OFFSET, 0]));
        for (let f = 0; f < 30; f++) {
            platform.posLin[1] -= V * DT;
            moveCharacter(ch, [0, 0, 0], [platform], G, DT);
        }
        console.log(
            `[character] platform-carry down — char vy ${ch.body.velLin[1].toFixed(3)} (platform −${V}), grounded ${ch.grounded}`,
        );
        expect(ch.grounded).toBe(true);
        expect(ch.body.velLin[1]).toBeCloseTo(-V, 1); // the carry's realized descent — 0 without the carry
    });
});

describe("AVBD character — full-speed push (velocity transfer)", () => {
    test("shoves a light box ahead at walking speed without overtaking it", () => {
        // a char at the floor walks +x at 3 m/s into a light box ahead of it. The box is NOT swept against
        // (it's dynamic, passed in `push`, not `statics`), so without velocity transfer the char tunnels
        // through it (the soft solver contact only carries the box at ~0.4 m/s, far under 3). The transfer
        // drives the box at the char's speed, so it stays ahead and the char never passes through it.
        const floor = ground([0, 0, 0]);
        const box = body([0.8, 0.8, 0.8], massOf([0.8, 0.8, 0.8], 0.5), 0.3, [1.2, 0.9, 0]);
        const cap = capsule(HALF_H, RADIUS, 0, 0.5, [0, 1.3, 0]);
        const ch = character(cap);
        const s = makeSolver([floor, box, cap], { layer: "warmstart", gravity: G });
        for (let f = 0; f < 120; f++) {
            moveCharacter(ch, [3, 0, 0], [floor], G, DT, false, [box]);
            step(s);
        }
        console.log(
            `[character] full-speed push — char x ${cap.posLin[0].toFixed(2)}, box x ${box.posLin[0].toFixed(2)}`,
        );
        expect(box.posLin.every(Number.isFinite)).toBe(true);
        expect(box.posLin[0]).toBeGreaterThan(3); // the box was shoved well forward (not left at ~1.2)
        expect(cap.posLin[0]).toBeLessThan(box.posLin[0]); // the char stayed BEHIND it — no overtake/tunnel
    });

    test("walking into a dynamic box's side is blocked at the face — never pops on top", () => {
        // the report: walking into a chest-high dynamic crate teleported the char on top of it. The old sweep
        // skipped every non-walkable dynamic contact, so the capsule penetrated the side freely; once deep
        // enough the closest-point MTV flipped to the box's TOP face (walkable), and the sweep ejected the
        // char upward by the full depth in one tick. Jolt's CharacterVirtual never excludes dynamics from
        // collision — the char is blocked at the face while the velocity transfer shoves the box. The box is
        // held in place here (push bodies are never moved by moveCharacter) — a worst-case immovable crate.
        const floor = ground([0, 0, 0]);
        const box = body([1, 1, 1], massOf([1, 1, 1], 1), 0.3, [1.5, 1.0, 0]); // faces x ∈ [1, 2], top y = 1.5
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.5, [0, 1.3, 0]));
        let maxY = 0;
        let maxX = 0;
        for (let f = 0; f < 120; f++) {
            moveCharacter(ch, [3, 0, 0], [floor], G, DT, false, [box]);
            maxY = Math.max(maxY, ch.body.posLin[1]);
            maxX = Math.max(maxX, ch.body.posLin[0]);
        }
        console.log(
            `[character] walk-into-dynamic — max x ${maxX.toFixed(3)} (face ${1 - RADIUS}), max y ${maxY.toFixed(3)} (rest 1.3)`,
        );
        expect(maxX).toBeLessThan(1 - RADIUS + 0.02); // stopped at the face — never tunneled into/through it
        expect(maxY).toBeLessThan(1.3 + 0.06); // stayed at floor rest — never ejected onto the box top (2.3)
        expect(box.velLin[0]).toBeGreaterThan(2.9); // the full-speed push still drives the box at the char's pace
    });

    test("jumping into a tall dynamic box's side slides off — never lands on top", () => {
        // the report's other shape: jumping into a dynamic body went through it and landed on top. With side
        // contacts excluded from the sweep, the airborne capsule sank into the face until the MTV flipped
        // upward. Fixed, the side blocks like a static wall: the char rises along the face, falls back, and
        // lands on the floor behind the box.
        const floor = ground([0, 0, 0]);
        const box = body([1, 4, 4], massOf([1, 4, 4], 1), 0.3, [1.5, 2.5, 0]); // faces x ∈ [1, 2], top y = 4.5
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.5, [0, 1.3, 0]), 50, 5); // jump apex 1.25 above rest
        let maxY = 0;
        let maxX = 0;
        for (let f = 0; f < 180; f++) {
            moveCharacter(ch, [3, 0, 0], [floor], G, DT, f === 0, [box]);
            maxY = Math.max(maxY, ch.body.posLin[1]);
            maxX = Math.max(maxX, ch.body.posLin[0]);
        }
        console.log(
            `[character] jump-into-dynamic — max x ${maxX.toFixed(3)} (face ${1 - RADIUS}), max y ${maxY.toFixed(3)} (apex 2.55), final y ${ch.body.posLin[1].toFixed(3)}`,
        );
        expect(maxX).toBeLessThan(1 - RADIUS + 0.02); // blocked at the face through the whole arc
        expect(maxY).toBeLessThan(2.55 + 0.06); // never above the jump apex — no upward ejection
        expect(Math.abs(ch.body.posLin[1] - 1.3)).toBeLessThan(0.03); // back at floor rest, beside the box
        expect(ch.grounded).toBe(true);
    });

    test("stands ON a dynamic box (a walkable contact supports it) instead of sinking through", () => {
        // the regression: a char dropped onto a DYNAMIC box (passed in `push`, like the full-speed shove) fell
        // straight through to the floor — the sweep depenetrated only against statics, so nothing held it up on
        // a box. A box the char stands ON is a WALKABLE contact (normal up), so the sweep must push the char up
        // off it and ground it, exactly as on a static floor — while a box walked INTO from the side (non-
        // walkable) is still left to the velocity-transfer push above (shoved, not climbed). The box sits in
        // `push`; moveCharacter never moves a push body's position, so it's a fixed support here and the floor
        // is the fell-through indicator: no fix → the char ends on the floor (y 1.3), fixed → on the box (y 2.3).
        const floor = ground([0, 0, 0]); // top y = 0.5 — where the char lands if it falls through the box
        const box = body([1, 1, 1], massOf([1, 1, 1], 1), 0.8, [0, 1.0, 0]); // dynamic, top at y = 1.5
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 4, 0])); // dropped from above the box
        for (let f = 0; f < 200; f++) moveCharacter(ch, [0, 0, 0], [floor], G, DT, false, [box]);

        const restY = 1.5 + REST_OFFSET; // 2.3 — capsule surface on the box top (floor-rest would be 1.3)
        console.log(
            `[character] stand-on-dynamic — char y ${ch.body.posLin[1].toFixed(3)} (rest ${restY}, floor ${0.5 + REST_OFFSET}), grounded ${ch.grounded}`,
        );
        expect(ch.body.posLin.every(Number.isFinite)).toBe(true);
        expect(Math.abs(ch.body.posLin[1] - restY)).toBeLessThan(0.06); // on the box top, NOT sunk to the floor
        expect(ch.grounded).toBe(true); // standing on the box counts as ground (can jump off it)
        expect(length(ch.body.velLin)).toBeLessThan(1e-3); // settled — no residual jitter
        expect(length(box.velLin)).toBe(0); // down-cancel: landing never hammers the box down at fall speed
    });
});

describe("AVBD character — candidate cull (the gather)", () => {
    // The sphere cull is a contact-set-preserving superset: every phase gates on gap < GROUND_SNAP or
    // depth > 0, and a culled body's gap stays above GROUND_SNAP at every pose the tick visits, so the
    // culled run is BIT-IDENTICAL to the brute one (every filler contribution sits behind a conditional
    // that fails). The GPU runs the same predicate; these gates pin the oracle half of GPU == oracle.

    const diag = (): MoveDiag => ({ candidates: 0, overflow: false, guard: false });

    // far filler statics in a ring — gathered by neither side's contact set, probed only by brute
    const fillers = (n: number, r: number): Body[] => {
        const out: Body[] = [];
        for (let i = 0; i < n; i++) {
            const a = (i / n) * Math.PI * 2;
            out.push(body([1, 1, 1], 0, 0.8, [Math.cos(a) * r, 1 + (i % 5), Math.sin(a) * r]));
        }
        return out;
    };

    // run the same scene culled and brute, frame-locked; return the max |pos| divergence (expect 0 exact)
    const divergence = (
        mk: () => { ch: ReturnType<typeof character>; statics: Body[]; push: Body[]; input: Vec3 },
        frames: number,
        jumpAt = -1,
    ): number => {
        const a = mk();
        const b = mk();
        let max = 0;
        for (let f = 0; f < frames; f++) {
            moveCharacter(a.ch, a.input, a.statics, G, DT, f === jumpAt, a.push);
            moveCharacter(b.ch, b.input, b.statics, G, DT, f === jumpAt, b.push, { cull: false });
            max = Math.max(
                max,
                length(sub(a.ch.body.posLin, b.ch.body.posLin)),
                length(sub(a.ch.body.velLin, b.ch.body.velLin)),
            );
            for (let i = 0; i < a.push.length; i++)
                max = Math.max(max, length(sub(a.push[i].velLin, b.push[i].velLin)));
        }
        return max;
    };

    test("cull equivalence: culled == brute, exact f64, across the behavioral scene shapes", () => {
        const restY = 0.5 + REST_OFFSET;
        // step-up + walk among far fillers (slide + snap + grounded paths). The cull is
        // contact-set-preserving, so culled == brute holds at any filler count — a handful of culled
        // bodies exercises the gather's reject path; the dense >64 stress is the `overflow` test.
        const stepUp = divergence(
            () => ({
                ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])),
                statics: [
                    ground([0, 0, 0]),
                    body([40, 0.7, 4], 0, 0.8, [23, 0.35, 0]),
                    ...fillers(24, 30),
                ],
                push: [],
                input: [2, 0, 0] as Vec3,
            }),
            200,
        );
        // full-speed push + jump among fillers (push + transfer + jump paths)
        const pushScene = divergence(
            () => ({
                ch: character(capsule(HALF_H, RADIUS, 0, 0.5, [0, restY, 0]), 50, 5),
                statics: [ground([0, 0, 0]), ...fillers(24, 25)],
                push: [body([0.8, 0.8, 0.8], massOf([0.8, 0.8, 0.8], 0.5), 0.3, [1.2, 0.9, 0])],
                input: [3, 0, 0] as Vec3,
            }),
            120,
            30,
        );
        // moving-platform carry among fillers (carry + re-gather paths) — the platform pose advances
        // per-frame, which `divergence` can't express, so this scene drives both sims directly
        const carry = (() => {
            const mk = () => ({
                platform: body([10, 1, 10], 0, 0.8, [0, 0, 0], [1.5, 0, 0]),
                ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])),
                far: fillers(24, 28),
            });
            const a = mk();
            const b = mk();
            let max = 0;
            for (let f = 0; f < 90; f++) {
                a.platform.posLin[0] += 1.5 * DT;
                b.platform.posLin[0] += 1.5 * DT;
                moveCharacter(a.ch, [0, 0, 0], [a.platform, ...a.far], G, DT);
                moveCharacter(b.ch, [0, 0, 0], [b.platform, ...b.far], G, DT, false, [], {
                    cull: false,
                });
                max = Math.max(max, length(sub(a.ch.body.posLin, b.ch.body.posLin)));
            }
            return max;
        })();
        console.log(
            `[character/cull] divergence — step-up ${stepUp.toExponential(1)}, push ${pushScene.toExponential(1)}, carry ${carry.toExponential(1)}`,
        );
        expect(stepUp).toBe(0); // bit-identical — the cull never changes the contact set
        expect(pushScene).toBe(0);
        expect(carry).toBe(0);
    });

    test("cull equivalence: seeded randomized sweep (near + far bodies, random input)", () => {
        let seed = 0; // re-seeded by mk, so both runs build the identical scene
        const rnd = () => {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            return seed / 0xffffffff;
        };
        const restY = 0.5 + REST_OFFSET;
        const mk = () => {
            seed = 0x12345678;
            const statics: Body[] = [ground([0, 0, 0])];
            for (let i = 0; i < 60; i++) {
                const r = 2 + rnd() * 40; // near boxes the char can hit, far ones it can't
                const a = rnd() * Math.PI * 2;
                const s = 0.3 + rnd() * 1.5;
                statics.push(
                    body([s, s, s], 0, 0.8, [Math.cos(a) * r, 0.5 + s / 2, Math.sin(a) * r]),
                );
            }
            const input: Vec3 = [1 + rnd() * 2, 0, rnd() - 0.5];
            return {
                ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0])),
                statics,
                push: [],
                input,
            };
        };
        const max = divergence(mk, 120);
        console.log(`[character/cull] random sweep divergence ${max.toExponential(1)}`);
        expect(max).toBe(0);
    });

    test("far bodies are culled (the gather actually culls)", () => {
        const d = diag();
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 1.3, 0]));
        moveCharacter(ch, [1, 0, 0], [ground([0, 0, 0]), ...fillers(500, 30)], G, DT, false, [], {
            diag: d,
        });
        console.log(`[character/cull] candidates ${d.candidates} of 501`);
        expect(d.candidates).toBe(1); // only the floor — every ring body rejected by the sphere
        expect(d.overflow).toBe(false);
    });

    test("re-gather: a fast platform's carry widens the band to include a body the provisional gather missed", () => {
        // provisional band (groundVel unknown) keeps a small box only within ~1.96 m; the platform carry
        // (60 m/s → 1 m/tick) raises the full band past 2.4 m. Without the re-gather the box would stay
        // culled while the carry could sweep the char into it — the gate asserts it lands in the set.
        const d = diag();
        const platform = body([100, 1, 100], 0, 0.8, [0, 0, 0], [60, 0, 0]);
        const box = body([0.3, 0.3, 0.3], 0, 0.8, [2.4, 1.3, 0]);
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 1.3, 0]));
        moveCharacter(ch, [0, 0, 0], [platform, box], G, DT, false, [], { diag: d });
        console.log(`[character/cull] re-gather candidates ${d.candidates} (platform + box)`);
        expect(d.candidates).toBe(2); // the re-gather pulled the box in; provisional alone keeps only the platform
    });

    test("overflow: >64 candidates keeps the first 64 in scan order and flags loudly", () => {
        let seed = 7;
        const rnd = () => {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            return seed / 0xffffffff;
        };
        const mk = () => {
            seed = 7;
            const statics: Body[] = [];
            for (let i = 0; i < 100; i++) {
                const a = rnd() * Math.PI * 2;
                const r = rnd() * 1.5;
                statics.push(
                    body([0.2, 0.2, 0.2], 0, 0.8, [Math.cos(a) * r, 0.6 + rnd(), Math.sin(a) * r]),
                );
            }
            return { ch: character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 1.3, 0])), statics };
        };
        const d = diag();
        let overflowed = false;
        let div = 0;
        const a = mk();
        const b = mk();
        // "culled(100) == brute(first 64)" pins the cap policy, and only holds while every body is in
        // range (the cap active each tick) — the depenetration ejects the char from the cluster after a
        // few ticks, so compare until the overflow lapses, not a fixed window.
        for (let f = 0; f < 30; f++) {
            moveCharacter(a.ch, [0.5, 0, 0], a.statics, G, DT, false, [], { diag: d });
            if (!d.overflow) break;
            overflowed = true;
            // the brute reference for the cap policy: the first 64 bodies, un-culled
            moveCharacter(b.ch, [0.5, 0, 0], b.statics.slice(0, 64), G, DT, false, [], {
                cull: false,
            });
            div = Math.max(div, length(sub(a.ch.body.posLin, b.ch.body.posLin)));
        }
        console.log(
            `[character/cull] overflow — flagged ${overflowed}, divergence vs first-64 brute ${div.toExponential(1)}`,
        );
        expect(overflowed).toBe(true);
        expect(div).toBe(0); // the cap policy IS "first 64 in scan order" — pinned so the GPU can mirror it
    });

    test("guard: spawning deep inside geometry trips the displacement guard, stays finite", () => {
        const d = diag();
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, 0, 0]));
        moveCharacter(ch, [0, 0, 0], [body([4, 4, 4], 0, 0.8, [0, 0, 0])], G, DT, false, [], {
            diag: d,
        });
        console.log(
            `[character/cull] guard — tripped ${d.guard}, pos ${ch.body.posLin.map((v) => v.toFixed(2)).join(",")}`,
        );
        expect(d.guard).toBe(true); // the band budget was exceeded — loud, never silent
        expect(ch.body.posLin.every(Number.isFinite)).toBe(true);
    });
});

describe("AVBD character — jump (single jump, buffering, coyote)", () => {
    const Jump = 5; // launch speed → apex = JUMP²/(2|G|) = 1.25 m above the launch
    const restY = 0.5 + REST_OFFSET; // 1.3 on the floor

    test("spam jump: one jump per landing, never double-jumps (bounded apex)", () => {
        // holding/spamming the jump button must yield a SINGLE jump per ground contact — the coyote credit is
        // consumed on launch and won't refill until the next landing. A double-jump (re-firing mid-air on the
        // held press) would gain height every airborne frame and fly away unbounded; a single jump arcs to a
        // fixed apex, lands, and jumps again. So a bounded apex + a small jump count (≈ one per bounce cycle,
        // not one per frame) is the discriminator.
        const floor = ground([0, 0, 0]);
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY, 0]), 50, Jump);
        let maxY = Number.NEGATIVE_INFINITY;
        let jumps = 0;
        for (let f = 0; f < 300; f++) {
            moveCharacter(ch, [0, 0, 0], [floor], G, DT, true); // jump held every frame
            maxY = Math.max(maxY, ch.body.posLin[1]);
            if (ch.vel[1] === Jump) jumps++; // a launch frame sets vel.y exactly to jumpSpeed
        }
        const apex = restY + (Jump * Jump) / (2 * Math.abs(G)); // 2.55
        console.log(
            `[character/jump] spam — ${jumps} jumps / 300f, maxY ${maxY.toFixed(3)} (apex ~${apex.toFixed(2)})`,
        );
        expect(jumps).toBeGreaterThan(1); // it jumped repeatedly (on each landing)
        expect(jumps).toBeLessThan(10); // NOT every frame — single jump per cycle (a held button can't re-fire)
        expect(maxY).toBeLessThan(apex + 0.4); // bounded near one jump's apex (a double-jump would exceed it)
        expect(maxY).toBeGreaterThan(restY + 0.8); // it clearly left the ground
    });

    test("coyote: a jump pressed just after walking off a ledge still fires", () => {
        // a small platform (top y 0.5, +x edge at x = 1); walk +x off it. A jump pressed on the first airborne
        // frame after the ledge fires (coyote credit hasn't decayed), even though the character isn't grounded.
        const plat = body([2, 1, 4], 0, 0.8, [0, 0, 0]); // x ∈ [-1, 1]
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0.5, restY, 0]), 50, Jump);
        let leftAt = -1;
        let coyoteJumped = false;
        for (let f = 0; f < 120; f++) {
            const wasGrounded = ch.grounded;
            const press = leftAt >= 0 && f === leftAt + 1; // the first airborne frame after the ledge
            moveCharacter(ch, [3, 0, 0], [plat], G, DT, press);
            if (wasGrounded && !ch.grounded && leftAt < 0) leftAt = f;
            if (press && ch.vel[1] === Jump) coyoteJumped = true;
        }
        console.log(
            `[character/jump] coyote — left at frame ${leftAt}, coyote jump ${coyoteJumped}`,
        );
        expect(leftAt).toBeGreaterThan(0); // it walked off the ledge (grounded lapsed)
        expect(coyoteJumped).toBe(true); // the jump fired within the coyote window despite being airborne
    });

    test("buffer: a jump pressed just before landing fires on touchdown", () => {
        // drop the capsule a short way; press jump ONCE while descending close to the ground (a few frames
        // before landing), then release. The buffered press must survive to touchdown and fire the jump there —
        // without buffering it would be swallowed (pressed mid-air, not grounded) and the character would just settle.
        const floor = ground([0, 0, 0]);
        const ch = character(capsule(HALF_H, RADIUS, 0, 0.8, [0, restY + 0.4, 0]), 50, Jump);
        let buffered = false;
        let bufferJumped = false;
        for (let f = 0; f < 120; f++) {
            const press =
                !buffered && !ch.grounded && ch.vel[1] < 0 && ch.body.posLin[1] < restY + 0.15;
            if (press) buffered = true;
            moveCharacter(ch, [0, 0, 0], [floor], G, DT, press);
            if (buffered && ch.vel[1] === Jump) bufferJumped = true;
        }
        console.log(
            `[character/jump] buffer — pressed-before-landing ${buffered}, fired ${bufferJumped}`,
        );
        expect(buffered).toBe(true); // the press happened while still airborne + descending
        expect(bufferJumped).toBe(true); // it fired on/just after landing (the buffer carried it)
    });
});
