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
    Pause,
    ChildOf,
    type Plugin,
} from "../src";
import { createTween } from "../src/extras/tween/tween";
import { resolveSequence, resetSequence } from "../src/extras/tween/sequence";
import { clearRegistry } from "../src/engine/ecs/component";

const Position = { x: [] as number[], y: [] as number[] };

const TestPlugin: Plugin = {
    name: "Test",
    components: { Position },
};

describe("Sequence", () => {
    let state: State;

    beforeEach(async () => {
        clearRegistry();
        state = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false });
    });

    describe("basic behavior", () => {
        test("should create a sequence entity", () => {
            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            expect(state.hasComponent(seqEid, Sequence)).toBe(true);
        });

        test("should propagate playing to child tweens", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            expect(Tween.state[tween1!]).toBe(TweenState.Idle);
            expect(Tween.state[tween2!]).toBe(TweenState.Idle);

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.state[tween1!]).toBe(TweenState.Playing);
            expect(Tween.state[tween2!]).toBe(TweenState.Playing);
        });

        test("should mark Complete when all children complete", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.3);
            expect(Sequence.state[seqEid] === SequenceState.Playing).toBe(true);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(false);

            stepFor(state, 0.3);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);

            state.step(0);
            expect(Sequence.state[seqEid] === SequenceState.Playing).toBe(false);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(true);
        });

        test("should wait for longest child to complete", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });
            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.6);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);
            expect(Tween.state[tween2!] === TweenState.Complete).toBe(false);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(false);

            stepFor(state, 0.5);
            expect(Tween.state[tween2!] === TweenState.Complete).toBe(true);

            state.step(0);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(true);
        });
    });

    describe("edge cases", () => {
        test("should not complete when no children exist", () => {
            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.1);

            expect(Sequence.state[seqEid] === SequenceState.Playing).toBe(true);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(false);
        });

        test("should not restart already playing children", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            Tween.state[tween1!] = TweenState.Playing;
            Tween.elapsed[tween1!] = 0.5;

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.elapsed[tween1!]).toBeCloseTo(0.6, 2);
        });

        test("should not restart already complete children", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            Tween.state[tween1!] = TweenState.Playing;

            stepFor(state, 0.6);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);
            expect(Position.x[target]).toBeCloseTo(100, 1);

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.state[tween1!] === TweenState.Playing).toBe(false);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);
        });
    });

    describe("controls", () => {
        test("should restart by using reset function", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.6);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);
            state.step(0);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(true);
            expect(Position.x[target]).toBeCloseTo(100, 1);

            Sequence.state[seqEid] = SequenceState.Idle;
            Sequence.elapsed[seqEid] = 0;
            Tween.state[tween1!] = TweenState.Idle;
            Tween.elapsed[tween1!] = 0;
            Tween.to[tween1!] = 200;
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(150, 1);
        });

        test("should interpolate all children simultaneously", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;
            Position.y[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(50, 1);
            expect(Position.y[target]).toBeCloseTo(100, 1);
        });

        test("should handle resolve-reset-play pattern correctly", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seq1 = state.addEntity();
            state.addComponent(seq1, Sequence);
            const tween1 = createTween(state, target, "position.x", {
                to: 50,
                duration: 1,
            });
            state.addComponent(tween1!, pair(ChildOf.relation, seq1));

            const seq2 = state.addEntity();
            state.addComponent(seq2, Sequence);
            const tween2 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            state.addComponent(tween2!, pair(ChildOf.relation, seq2));

            Sequence.state[seq1] = SequenceState.Playing;
            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(25, 1);

            Sequence.state[seq1] = SequenceState.Complete;
            state.step(0);
            expect(Position.x[target]).toBe(50);

            Sequence.state[seq2] = SequenceState.Idle;
            Sequence.elapsed[seq2] = 0;
            Tween.state[tween2!] = TweenState.Idle;
            Tween.elapsed[tween2!] = 0;
            Sequence.state[seq2] = SequenceState.Playing;

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(75, 1);
        });
    });

    describe("pause", () => {
        test("should create pause component with duration", () => {
            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));

            expect(state.hasComponent(pauseEid, Pause)).toBe(true);
            expect(Pause.duration[pauseEid]).toBe(0.5);
        });

        test("tweens before pause should start immediately", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.state[tween1!] === TweenState.Playing).toBe(true);
            expect(Tween.state[tween2!] === TweenState.Playing).toBe(true);
        });

        test("tweens after pause should start after delay", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.state[tween1!] === TweenState.Playing).toBe(true);
            expect(Tween.state[tween2!]).toBe(TweenState.Idle);

            stepFor(state, 0.3);
            expect(Tween.state[tween2!]).toBe(TweenState.Idle);

            stepFor(state, 0.2);
            expect(Tween.state[tween2!] === TweenState.Playing).toBe(true);
        });

        test("pause at start should delay all tweens", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            expect(Tween.state[tween1!]).toBe(TweenState.Idle);

            stepFor(state, 0.3);
            expect(Tween.state[tween1!]).toBe(TweenState.Idle);

            stepFor(state, 0.2);
            expect(Tween.state[tween1!] === TweenState.Playing).toBe(true);
        });

        test("tween should account for overshoot when starting mid-frame", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.7);

            expect(Tween.state[tween1!] === TweenState.Playing).toBe(true);
            expect(Position.x[target]).toBeCloseTo(20, 1);
        });

        test("from should be captured correctly with overshoot", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 50;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.7);

            expect(Position.x[target]).toBeCloseTo(60, 1);
        });

        test("sequence should complete when all tweens complete", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;
            Position.y[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.3;

            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.6);
            expect(Tween.state[tween1!] === TweenState.Complete).toBe(true);
            expect(Tween.state[tween2!] === TweenState.Playing).toBe(true);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(false);

            stepFor(state, 0.3);
            expect(Tween.state[tween2!] === TweenState.Complete).toBe(true);

            state.step(0);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(true);
        });

        test("sequence elapsed should reset on replay", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 1.5);
            state.step(0);

            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(true);

            Sequence.state[seqEid] = SequenceState.Idle;
            Sequence.elapsed[seqEid] = 0;
            Tween.state[tween1!] = TweenState.Idle;
            Tween.elapsed[tween1!] = 0;
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.1);
            expect(Tween.state[tween1!]).toBe(TweenState.Idle);

            stepFor(state, 0.5);
            expect(Tween.state[tween1!] === TweenState.Playing).toBe(true);
        });

        test("empty sequence with only pauses should stay playing", () => {
            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 1.0);

            expect(Sequence.state[seqEid] === SequenceState.Playing).toBe(true);
            expect(Sequence.state[seqEid] === SequenceState.Complete).toBe(false);
        });
    });

    describe("skip (via setting COMPLETE)", () => {
        test("resolves active tweens to final values", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            Sequence.state[seqEid] = SequenceState.Playing;

            stepFor(state, 0.3);
            expect(Position.x[target]).toBeCloseTo(30, 1);

            Sequence.state[seqEid] = SequenceState.Complete;
            state.step(0);

            expect(Position.x[target]).toBe(100);
            expect(Tween.state[tween1!]).toBe(TweenState.Complete);
        });

        test("resolves pending tweens to final values", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;
            Position.y[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 2;

            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.3);

            expect(Tween.state[tween2!]).toBe(TweenState.Idle);

            Sequence.state[seqEid] = SequenceState.Complete;
            state.step(0);

            expect(Position.x[target]).toBe(100);
            expect(Position.y[target]).toBe(200);
            expect(Tween.state[tween1!]).toBe(TweenState.Complete);
            expect(Tween.state[tween2!]).toBe(TweenState.Complete);
        });

        test("captures from value for pending tweens", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 50;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 1;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            Sequence.state[seqEid] = SequenceState.Complete;
            state.step(0);

            expect(Position.x[target]).toBe(100);
        });

        test("resolves to last-ending tween when overlapping (long first)", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1.5,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.x", {
                to: 200,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            Sequence.state[seqEid] = SequenceState.Complete;
            state.step(0);

            expect(Position.x[target]).toBe(100);
        });

        test("resolves to last-ending tween when overlapping (long second)", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.x", {
                to: 200,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            Sequence.state[seqEid] = SequenceState.Complete;
            state.step(0);

            expect(Position.x[target]).toBe(200);
        });
    });

    describe("automatic delay computation", () => {
        test("should compute delays automatically when sequence starts playing", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            // Delays are computed automatically when sequence starts

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);

            // Tween should NOT have started yet (pause is 0.5s)
            expect(Tween.state[tween1!]).toBe(TweenState.Idle);
            expect(Position.x[target]).toBe(0);

            stepFor(state, 0.5);

            // Now tween should be playing (0.6s total, past the 0.5s pause)
            expect(Tween.state[tween1!]).toBe(TweenState.Playing);
            expect(Position.x[target]).toBeGreaterThan(0);
        });

        test("should recompute delays when sequence is reset and replayed", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            // Create pause BEFORE tween so it sorts first
            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.2;
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });
            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            // First play - delays computed automatically
            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.1);
            expect(Tween.state[tween1!]).toBe(TweenState.Idle); // Still waiting for 0.2s pause

            stepFor(state, 0.2);
            expect(Tween.state[tween1!]).toBe(TweenState.Playing);

            stepFor(state, 0.5);
            state.step(0);
            expect(Sequence.state[seqEid]).toBe(SequenceState.Complete);

            // Change pause duration for second play
            Pause.duration[pauseEid] = 0.5;

            // Reset and replay
            Sequence.state[seqEid] = SequenceState.Idle;
            Sequence.elapsed[seqEid] = 0;
            Tween.state[tween1!] = TweenState.Idle;
            Tween.elapsed[tween1!] = 0;
            Tween.to[tween1!] = 200;

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.3);

            // Tween should still be waiting (new pause is 0.5s, recomputed)
            expect(Tween.state[tween1!]).toBe(TweenState.Idle);

            stepFor(state, 0.3);
            // Now tween should be playing (past 0.5s pause)
            expect(Tween.state[tween1!]).toBe(TweenState.Playing);
        });
    });

    describe("resolveSequence helper", () => {
        test("resolves sequence that was never played", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.x", {
                to: 200,
                duration: 0.5,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            resolveSequence(state, seqEid);

            expect(Sequence.state[seqEid]).toBe(SequenceState.Complete);
            expect(Tween.state[tween1!]).toBe(TweenState.Complete);
            expect(Tween.state[tween2!]).toBe(TweenState.Complete);
            expect(Position.x[target]).toBe(200);
        });

        test("resolves with correct order based on computed delays", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1.0,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.x", {
                to: 200,
                duration: 1.0,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            resolveSequence(state, seqEid);

            expect(Position.x[target]).toBe(200);
        });

        test("is idempotent for already complete sequences", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 1.5);
            state.step(0);

            expect(Sequence.state[seqEid]).toBe(SequenceState.Complete);
            expect(Position.x[target]).toBe(100);

            Position.x[target] = 50;

            resolveSequence(state, seqEid);

            expect(Position.x[target]).toBe(50);
        });
    });

    describe("resetSequence helper", () => {
        test("resets sequence to initial state", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 0.5);

            expect(Sequence.elapsed[seqEid]).toBeGreaterThan(0);
            expect(Tween.elapsed[tween1!]).toBeGreaterThan(0);

            resetSequence(state, seqEid);

            expect(Sequence.state[seqEid]).toBe(SequenceState.Idle);
            expect(Sequence.elapsed[seqEid]).toBe(0);
            expect(Tween.state[tween1!]).toBe(TweenState.Idle);
            expect(Tween.elapsed[tween1!]).toBe(0);
        });

        test("allows replay after reset", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));

            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 1.5);
            state.step(0);

            expect(Position.x[target]).toBe(100);

            resetSequence(state, seqEid);
            Tween.to[tween1!] = 200;
            Sequence.state[seqEid] = SequenceState.Playing;
            stepFor(state, 1.5);
            state.step(0);

            expect(Position.x[target]).toBe(200);
        });
    });

    describe("resolve-reset-play workflow", () => {
        test("resolve then play next sequence captures correct from value", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seq1 = state.addEntity();
            state.addComponent(seq1, Sequence);
            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            state.addComponent(tween1!, pair(ChildOf.relation, seq1));

            const seq2 = state.addEntity();
            state.addComponent(seq2, Sequence);
            const tween2 = createTween(state, target, "position.x", {
                to: 200,
                duration: 1,
            });
            state.addComponent(tween2!, pair(ChildOf.relation, seq2));

            Sequence.state[seq1] = SequenceState.Playing;
            stepFor(state, 0.5);

            expect(Position.x[target]).toBeCloseTo(50, 1);

            resolveSequence(state, seq1);

            expect(Position.x[target]).toBe(100);

            resetSequence(state, seq2);
            Sequence.state[seq2] = SequenceState.Playing;
            stepFor(state, 0.5);

            expect(Position.x[target]).toBeCloseTo(150, 1);
        });

        test("resolution produces same final values as normal play", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;
            Position.y[target] = 0;

            const seqEid = state.addEntity();
            state.addComponent(seqEid, Sequence);

            const tween1 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            const pauseEid = state.addEntity();
            state.addComponent(pauseEid, Pause);
            Pause.duration[pauseEid] = 0.5;

            const tween2 = createTween(state, target, "position.y", {
                to: 200,
                duration: 1,
            });

            state.addComponent(tween1!, pair(ChildOf.relation, seqEid));
            state.addComponent(pauseEid, pair(ChildOf.relation, seqEid));
            state.addComponent(tween2!, pair(ChildOf.relation, seqEid));

            resolveSequence(state, seqEid);

            const resolvedX = Position.x[target];
            const resolvedY = Position.y[target];

            Position.x[target] = 0;
            Position.y[target] = 0;
            resetSequence(state, seqEid);
            Sequence.state[seqEid] = SequenceState.Playing;

            for (let i = 0; i < 200; i++) {
                stepFor(state, 0.01);
            }

            expect(Position.x[target]).toBe(resolvedX);
            expect(Position.y[target]).toBe(resolvedY);
        });

        test("multiple sequences resolve in correct order", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const seq1 = state.addEntity();
            state.addComponent(seq1, Sequence);
            const tween1 = createTween(state, target, "position.x", {
                to: 50,
                duration: 1,
            });
            state.addComponent(tween1!, pair(ChildOf.relation, seq1));

            const seq2 = state.addEntity();
            state.addComponent(seq2, Sequence);
            const tween2 = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            state.addComponent(tween2!, pair(ChildOf.relation, seq2));

            Sequence.state[seq1] = SequenceState.Playing;
            stepFor(state, 0.5);

            Sequence.state[seq2] = SequenceState.Playing;
            state.step(0);

            resolveSequence(state, seq1);
            resolveSequence(state, seq2);

            expect(Position.x[target]).toBe(100);
        });
    });
});
