import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { load, parse, State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Slab } from "../slab";
import {
    Body,
    bodyTraits,
    ConstraintSystem,
    installBackend,
    Joint,
    type JointDef,
    jointTraits,
    type PhysicsBackend,
    Spring,
    type SpringDef,
    springTraits,
    uninstallBackend,
} from "./index";

// Spring / Joint scene authoring (Phase 6.6): a constraint is a standalone `<a spring|joint="…">` entity
// that references its two bodies by `@name`, the consumer-shaped relation (like `Animation.target`). This is the
// "components just work in scene files" contract — pure scene-parse + load, no device: `@name` → eid, the
// vec3 anchors → lanes, the trait defaults, and the fixed-joint `∞` parse hook. The solve itself (warmstart
// across an authored set) is the gym `constraints` scenario (real GPU); this is the authoring half.
describe("constraint authoring (scene)", () => {
    let state: State;

    beforeEach(() => {
        clear();
        state = new State();
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
    });

    test("spring resolves both body refs, anchors, and scalars", () => {
        const nodes = parse(
            `<scene><a id="anchor" /><a id="block" /><a spring="a: @anchor; b: @block; r-b: 1 2 3; stiffness: 50; rest: 4" /></scene>`,
        );
        const map = load(nodes, state);
        const anchor = map.get(nodes[0])!;
        const block = map.get(nodes[1])!;
        const spring = map.get(nodes[2])!;
        expect(Spring.a.get(spring)).toBe(anchor);
        expect(Spring.b.get(spring)).toBe(block);
        expect(Spring.rB.x.get(spring)).toBe(1);
        expect(Spring.rB.y.get(spring)).toBe(2);
        expect(Spring.rB.z.get(spring)).toBe(3);
        expect(Spring.stiffness.get(spring)).toBe(50);
        expect(Spring.rest.get(spring)).toBe(4);
    });

    test("joint defaults to spherical (stiffnessAng 0) with an unauthored angular lock", () => {
        const nodes = parse(
            `<scene><a id="pivot" /><a id="bob" /><a joint="a: @pivot; b: @bob; r-b: 0 2.5 0" /></scene>`,
        );
        const map = load(nodes, state);
        const joint = map.get(nodes[2])!;
        expect(Joint.a.get(joint)).toBe(map.get(nodes[0])!);
        expect(Joint.rB.y.get(joint)).toBe(2.5);
        expect(Joint.stiffnessAng.get(joint)).toBe(0);
    });

    test("joint `stiffness-ang: fixed` parses to ∞ (the fixed-joint angular lock)", () => {
        const nodes = parse(
            `<scene><a id="a" /><a id="b" /><a joint="a: @a; b: @b; stiffness-ang: fixed" /></scene>`,
        );
        const map = load(nodes, state);
        expect(Joint.stiffnessAng.get(map.get(nodes[2])!)).toBe(Number.POSITIVE_INFINITY);
    });

    test("joint `stiffness-ang` still accepts a number (the hook only catches the keyword)", () => {
        const nodes = parse(
            `<scene><a id="a" /><a id="b" /><a joint="a: @a; b: @b; stiffness-ang: 1000" /></scene>`,
        );
        const map = load(nodes, state);
        expect(Joint.stiffnessAng.get(map.get(nodes[2])!)).toBe(1000);
    });
});

// A same-update realias of a body an authored Joint references (destroy + create recycling its eid) leaves
// the Joint's numeric a/b refs unchanged, so the re-upload signature must fold each endpoint's create-stamp
// or the backend joint silently pins the NEW occupant at the old anchors (ecs.md "An eid is a borrow").
describe("constraint re-upload on an endpoint realias", () => {
    let state: State;
    let joints: JointDef[][];

    function recordingBackend(): PhysicsBackend {
        return {
            step() {},
            readBody: () => null,
            setKinematic() {},
            setVelocity() {},
            setSprings() {},
            setJoints(j) {
                joints.push([...j]);
            },
            get gravity() {
                return -10;
            },
            get dt() {
                return 1 / 60;
            },
            compose() {},
        };
    }

    beforeEach(() => {
        clear();
        state = new State();
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        joints = [];
        uninstallBackend();
        installBackend(recordingBackend()); // arms the constraint re-upload for the fresh backend
    });

    afterEach(() => {
        uninstallBackend();
    });

    test("a recycled endpoint eid re-uploads the joint set", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);
        Body.pos.set(bob, 0, -2, 0, 0);

        const joint = state.create();
        state.add(joint, Joint);
        Joint.a.set(joint, anchor);
        Joint.b.set(joint, bob);

        // first sync uploads the authored joint once
        ConstraintSystem.update?.(state);
        expect(joints.length).toBe(1);
        expect(joints[0][0]?.b).toBe(bob);

        // a no-op re-run does NOT re-upload (signature unchanged)
        ConstraintSystem.update?.(state);
        expect(joints.length).toBe(1);

        // same update: destroy the bob endpoint and recycle its eid with a fresh Body — the Joint's numeric
        // `b` is identical, so only the create-stamp fold makes the realias visible.
        state.destroy(bob);
        const bob2 = state.create();
        expect(bob2).toBe(bob);
        state.add(bob2, Body);
        Body.pos.set(bob2, 5, -2, 0, 0);

        ConstraintSystem.update?.(state);
        expect(joints.length).toBe(2); // re-uploaded so the backend joint rebinds to the new occupant
    });
});

// The stiffness guard at the backend-neutral authoring layer (jointDefs/springDefs in index.ts): a
// negative or NaN stiffnessAng/stiffness is dropped with a warnOnce before reaching either backend, so
// both backends inherit one behavior. The recording backend is a proxy for any backend (tumble or AVBD)
// since the guard sits in the shared ConstraintSystem path. Valid authored values (0, finite-positive, ∞)
// pass through unchanged — the grant arm pins that the guard does not over-refuse.
describe("stiffness guard (authoring layer)", () => {
    let state: State;
    let joints: JointDef[][];
    let springs: SpringDef[][];

    function recordingBackend(): PhysicsBackend {
        return {
            step() {},
            readBody: () => null,
            setKinematic() {},
            setVelocity() {},
            setSprings(s) {
                springs.push([...s]);
            },
            setJoints(j) {
                joints.push([...j]);
            },
            get gravity() {
                return -10;
            },
            get dt() {
                return 1 / 60;
            },
            compose() {},
        };
    }

    beforeEach(() => {
        clear();
        state = new State();
        register("body", Body, bodyTraits);
        register("spring", Spring, springTraits);
        register("joint", Joint, jointTraits);
        Slab.collect();
        joints = [];
        springs = [];
        uninstallBackend();
        installBackend(recordingBackend());
    });

    afterEach(() => {
        uninstallBackend();
    });

    // witnessed red: exit code 1 — without the guard branch in jointDefs, the -1 def passes through and
    // setJoints receives a 1-element array; the assertion `expect(joints[0]).toHaveLength(0)` fails.
    test("a negative stiffnessAng joint is dropped — no joint reaches the backend", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);
        Body.pos.set(bob, 0, -2, 0, 0);

        const joint = state.create();
        state.add(joint, Joint);
        Joint.a.set(joint, anchor);
        Joint.b.set(joint, bob);
        Joint.stiffnessAng.set(joint, -1);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            ConstraintSystem.update?.(state);
            expect(joints.length).toBe(1);
            expect(joints[0]).toHaveLength(0); // the invalid def was dropped — no joint for the backend
            expect(warn).toHaveBeenCalled();
            const msg = warn.mock.calls.find((c) => String(c[0]).includes("joint"))?.[0];
            expect(msg).toBeDefined();
            expect(String(msg)).toContain("skipped");
        } finally {
            warn.mockRestore();
        }
    });

    // witnessed red: exit code 1 — without the guard, NaN passes the comparison-only checks (NaN < 0 is
    // false) and setJoints receives a 1-element array with NaN stiffnessAng; the length assertion fails.
    test("a NaN stiffnessAng joint is dropped — no joint reaches the backend", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);
        Body.pos.set(bob, 0, -2, 0, 0);

        const joint = state.create();
        state.add(joint, Joint);
        Joint.a.set(joint, anchor);
        Joint.b.set(joint, bob);
        Joint.stiffnessAng.set(joint, Number.NaN);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            ConstraintSystem.update?.(state);
            expect(joints.length).toBe(1);
            expect(joints[0]).toHaveLength(0); // NaN was dropped — comparison-only guards can't catch it
            expect(warn).toHaveBeenCalled();
            const msg = warn.mock.calls.find((c) => String(c[0]).includes("joint"))?.[0];
            expect(msg).toBeDefined();
            expect(String(msg)).toContain("skipped");
        } finally {
            warn.mockRestore();
        }
    });

    // witnessed red: exit code 1 — without the guard, the -1 stiffness passes through and setSprings
    // receives a 1-element array; the length assertion fails.
    test("a negative stiffness spring is dropped — no spring reaches the backend", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);

        const spring = state.create();
        state.add(spring, Spring);
        Spring.a.set(spring, anchor);
        Spring.b.set(spring, bob);
        Spring.stiffness.set(spring, -1);
        Spring.rest.set(spring, 4);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            ConstraintSystem.update?.(state);
            expect(springs.length).toBe(1);
            expect(springs[0]).toHaveLength(0);
            expect(warn).toHaveBeenCalled();
            const msg = warn.mock.calls.find((c) => String(c[0]).includes("spring"))?.[0];
            expect(msg).toBeDefined();
            expect(String(msg)).toContain("skipped");
        } finally {
            warn.mockRestore();
        }
    });

    // witnessed red: exit code 1 — without the guard, NaN passes the comparison-only checks and
    // setSprings receives a 1-element array with NaN stiffness; the length assertion fails.
    test("a NaN stiffness spring is dropped — no spring reaches the backend", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);

        const spring = state.create();
        state.add(spring, Spring);
        Spring.a.set(spring, anchor);
        Spring.b.set(spring, bob);
        Spring.stiffness.set(spring, Number.NaN);
        Spring.rest.set(spring, 4);

        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            ConstraintSystem.update?.(state);
            expect(springs.length).toBe(1);
            expect(springs[0]).toHaveLength(0);
            expect(warn).toHaveBeenCalled();
            const msg = warn.mock.calls.find((c) => String(c[0]).includes("spring"))?.[0];
            expect(msg).toBeDefined();
            expect(String(msg)).toContain("skipped");
        } finally {
            warn.mockRestore();
        }
    });

    // grant arm: a finite-positive stiffnessAng passes the guard and builds a joint — the over-refusal
    // check no refusal arm can show. Always green (the guard admits finite-positive); the red-first
    // discipline applies to the skip arms, not the grant.
    test("a finite-positive stiffnessAng joint passes the guard (grant arm)", () => {
        const anchor = state.create();
        state.add(anchor, Body);
        Body.mass.set(anchor, 0);
        const bob = state.create();
        state.add(bob, Body);
        Body.pos.set(bob, 0, -2, 0, 0);

        const joint = state.create();
        state.add(joint, Joint);
        Joint.a.set(joint, anchor);
        Joint.b.set(joint, bob);
        Joint.stiffnessAng.set(joint, 1000);

        ConstraintSystem.update?.(state);
        expect(joints.length).toBe(1);
        expect(joints[0]).toHaveLength(1);
        expect(joints[0][0]?.stiffnessAng).toBe(1000);
    });
});
