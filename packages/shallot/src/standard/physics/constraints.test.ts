import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { load, parse, State } from "../..";
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
    springTraits,
    uninstallBackend,
} from "./index";

// Spring / Joint scene authoring (Phase 6.6): a constraint is a standalone `<a spring|joint="…">` entity
// that references its two bodies by `@name`, the consumer-shaped relation (like `Tween.target`). This is the
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
