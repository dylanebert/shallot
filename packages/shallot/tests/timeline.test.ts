import { test, expect, describe, beforeEach } from "bun:test";
import { stepFor } from "./helpers/state";
import { pair } from "../src/engine/ecs";
import {
    build,
    type State,
    TweenPlugin,
    Tween,
    TweenState,
    Sequence,
    SequenceState,
    ChildOf,
    parse,
    load,
    type Plugin,
} from "../src";
import { createTween } from "../src/extras/tween/tween";
import { clearRegistry } from "../src/engine/ecs/component";
import { Timeline, Transition, TimelinePlugin } from "../src/extras";

const Position = { x: [] as number[], y: [] as number[] };

const TestPlugin: Plugin = {
    name: "Test",
    components: { Position },
};

describe("Timeline", () => {
    let state: State;

    beforeEach(async () => {
        clearRegistry();
        state = await build({
            plugins: [TweenPlugin, TimelinePlugin, TestPlugin],
            defaults: false,
        });
    });

    test("transition plays when target changes", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);
        Timeline.step[tlEid] = 0;
        Timeline.target[tlEid] = 0;

        const seqEid = state.addEntity();
        state.addComponent(seqEid, Sequence);
        state.addComponent(seqEid, Transition);
        Transition.from[seqEid] = 0;
        Transition.to[seqEid] = 1;

        const tweenEid = createTween(state, target, "position.x", {
            to: 100,
            duration: 1,
        });
        state.addComponent(tweenEid!, pair(ChildOf.relation, seqEid));

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.5);

        expect(Sequence.state[seqEid]).toBe(SequenceState.Playing);
        expect(Tween.state[tweenEid!]).toBe(TweenState.Playing);
        expect(Position.x[target]).toBeGreaterThan(0);
    });

    test("playing transition resolves before new one starts", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const seq01 = state.addEntity();
        state.addComponent(seq01, Sequence);
        state.addComponent(seq01, Transition);
        Transition.from[seq01] = 0;
        Transition.to[seq01] = 1;
        const tween1 = createTween(state, target, "position.x", {
            to: 100,
            duration: 1,
        });
        state.addComponent(tween1!, pair(ChildOf.relation, seq01));

        const seq12 = state.addEntity();
        state.addComponent(seq12, Sequence);
        state.addComponent(seq12, Transition);
        Transition.from[seq12] = 1;
        Transition.to[seq12] = 2;
        const tween2 = createTween(state, target, "position.x", {
            to: 200,
            duration: 1,
        });
        state.addComponent(tween2!, pair(ChildOf.relation, seq12));

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.3);
        expect(Position.x[target]).toBeCloseTo(30, 1);

        Timeline.target[tlEid] = 2;
        stepFor(state, 0.5);

        expect(Position.x[target]).toBeGreaterThan(100);
        expect(Sequence.state[seq01]).toBe(SequenceState.Complete);
        expect(Sequence.state[seq12]).toBe(SequenceState.Playing);
    });

    test("standalone sequences are not affected", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;
        Position.y[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const standaloneSeq = state.addEntity();
        state.addComponent(standaloneSeq, Sequence);
        const standaloneTween = createTween(state, target, "position.y", {
            to: 50,
            duration: 1,
        });
        state.addComponent(standaloneTween!, pair(ChildOf.relation, standaloneSeq));
        Sequence.state[standaloneSeq] = SequenceState.Playing;

        const transSeq = state.addEntity();
        state.addComponent(transSeq, Sequence);
        state.addComponent(transSeq, Transition);
        Transition.from[transSeq] = 0;
        Transition.to[transSeq] = 1;
        const tween1 = createTween(state, target, "position.x", {
            to: 100,
            duration: 1,
        });
        state.addComponent(tween1!, pair(ChildOf.relation, transSeq));

        stepFor(state, 0.3);
        expect(Tween.state[standaloneTween!]).toBe(TweenState.Playing);

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.3);

        expect(Tween.state[standaloneTween!]).toBe(TweenState.Playing);
        expect(Position.y[target]).toBeGreaterThan(0);
    });

    test("no-op when step equals target", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const seqEid = state.addEntity();
        state.addComponent(seqEid, Sequence);
        state.addComponent(seqEid, Transition);
        Transition.from[seqEid] = 0;
        Transition.to[seqEid] = 1;
        const tweenEid = createTween(state, target, "position.x", {
            to: 100,
            duration: 1,
        });
        state.addComponent(tweenEid!, pair(ChildOf.relation, seqEid));

        stepFor(state, 0.5);

        expect(Sequence.state[seqEid]).toBe(SequenceState.Idle);
        expect(Position.x[target]).toBe(0);
    });

    test("step field updates after transition starts", () => {
        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const seqEid = state.addEntity();
        state.addComponent(seqEid, Sequence);
        state.addComponent(seqEid, Transition);
        Transition.from[seqEid] = 0;
        Transition.to[seqEid] = 1;

        Timeline.target[tlEid] = 1;
        state.step(0);

        expect(Timeline.step[tlEid]).toBe(1);
    });

    test("full transition completes with correct values", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const seqEid = state.addEntity();
        state.addComponent(seqEid, Sequence);
        state.addComponent(seqEid, Transition);
        Transition.from[seqEid] = 0;
        Transition.to[seqEid] = 1;
        const tweenEid = createTween(state, target, "position.x", {
            to: 100,
            duration: 0.5,
        });
        state.addComponent(tweenEid!, pair(ChildOf.relation, seqEid));

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.6);
        state.step(0);

        expect(Position.x[target]).toBe(100);
        expect(Tween.state[tweenEid!]).toBe(TweenState.Complete);
        expect(Sequence.state[seqEid]).toBe(SequenceState.Complete);
    });

    test("sequential transitions 0→1→2", () => {
        const target = state.addEntity();
        state.addComponent(target, Position);
        Position.x[target] = 0;

        const tlEid = state.addEntity();
        state.addComponent(tlEid, Timeline);

        const seq01 = state.addEntity();
        state.addComponent(seq01, Sequence);
        state.addComponent(seq01, Transition);
        Transition.from[seq01] = 0;
        Transition.to[seq01] = 1;
        const tween1 = createTween(state, target, "position.x", {
            to: 100,
            duration: 0.5,
        });
        state.addComponent(tween1!, pair(ChildOf.relation, seq01));

        const seq12 = state.addEntity();
        state.addComponent(seq12, Sequence);
        state.addComponent(seq12, Transition);
        Transition.from[seq12] = 1;
        Transition.to[seq12] = 2;
        const tween2 = createTween(state, target, "position.x", {
            to: 200,
            duration: 0.5,
        });
        state.addComponent(tween2!, pair(ChildOf.relation, seq12));

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.6);
        state.step(0);
        expect(Position.x[target]).toBe(100);

        Timeline.target[tlEid] = 2;
        stepFor(state, 0.6);
        state.step(0);
        expect(Position.x[target]).toBe(200);
    });

    test("XML scene loading with timeline + transition", () => {
        const nodes = parse(`
            <scene>
                <a id="box" position="x: 0" />
                <a timeline />
                <a
                    sequence
                    transition="from: 0; to: 1"
                >
                    <a tween="to: 100; field: position.x; duration: 0.5" target="@box" />
                </a>
            </scene>
        `);
        load(nodes, state);

        const tlEids = [...state.query([Timeline])];
        expect(tlEids.length).toBe(1);
        const tlEid = tlEids[0];

        const transEids = [...state.query([Transition])];
        expect(transEids.length).toBe(1);
        expect(Transition.from[transEids[0]]).toBe(0);
        expect(Transition.to[transEids[0]]).toBe(1);

        Timeline.target[tlEid] = 1;
        stepFor(state, 0.6);
        state.step(0);

        const boxEids = [...state.query([Position])];
        expect(boxEids.length).toBe(1);
        expect(Position.x[boxEids[0]]).toBe(100);
    });
});
