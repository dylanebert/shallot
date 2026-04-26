import { test, expect, describe, beforeEach } from "bun:test";
import { stepFor } from "./helpers/state";
import {
    build,
    type State,
    TweenPlugin,
    Tween,
    TweenState,
    Target,
    Transform,
    TransformsPlugin,
    parse,
    load,
    type Plugin,
} from "../src";
import { createTween } from "../src/extras/tween/tween";
import { EASING_FUNCTIONS, getEasing, getEasingIndex } from "../src/extras/tween/easing";
import { clearRegistry } from "../src/engine/ecs/component";

const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
const Sprite = { opacity: [] as number[] };

const TestPlugin: Plugin = {
    name: "Test",
    components: { Position, Sprite },
};

describe("Tween", () => {
    let state: State;

    beforeEach(async () => {
        clearRegistry();
        state = await build({ plugins: [TweenPlugin, TestPlugin], defaults: false });
    });

    describe("createTween", () => {
        test("should create a tween entity", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            expect(tweenEid).not.toBeNull();
            expect(tweenEid).toBeGreaterThanOrEqual(0);
            expect(state.hasComponent(tweenEid!, Tween)).toBe(true);
            expect(Tween.state[tweenEid!]).toBe(TweenState.Idle);
        });

        test("should return null for invalid field path", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "invalid", { to: 100 });
            expect(tweenEid).toBeNull();
        });

        test("should return null for unknown component", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "unknown.x", { to: 100 });
            expect(tweenEid).toBeNull();
        });

        test("should capture current value as from when playing starts", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 50;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(75, 1);
        });

        test("should default duration to 1 second", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "position.x", { to: 100 });

            expect(Tween.duration[tweenEid!]).toBe(1);
        });

        test("should use specified duration", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });

            expect(Tween.duration[tweenEid!]).toBe(0.5);
        });

        test("should handle duration=0 (instant tween) without NaN", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 50;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 0,
            });

            Tween.state[tweenEid!] = TweenState.Playing;
            state.step(0);

            expect(Number.isFinite(Position.x[target])).toBe(true);
            expect(Position.x[target]).toBe(100);
            expect(Tween.state[tweenEid!]).toBe(TweenState.Complete);
        });

        test("should default to linear easing", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "position.x", { to: 100 });

            expect(Tween.easing[tweenEid!]).toBe(0);
        });

        test("should use specified easing", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                easing: "ease-out-quad",
            });

            expect(Tween.easing[tweenEid!]).toBe(getEasingIndex("ease-out-quad"));
        });
    });

    describe("TweenSystem", () => {
        test("should interpolate value over time", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(50, 1);

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(100, 1);
        });

        test("should apply easing function", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
                easing: "ease-in-quad",
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.5);
            const easeInQuad = getEasing(getEasingIndex("ease-in-quad"));
            const expected = easeInQuad(0.5) * 100;
            expect(Position.x[target]).toBeCloseTo(expected, 1);
        });

        test("should mark as Complete on finish", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            expect(Tween.state[tweenEid!] === TweenState.Playing).toBe(true);
            expect(Tween.state[tweenEid!] === TweenState.Complete).toBe(false);

            stepFor(state, 0.6);

            expect(Tween.state[tweenEid!] === TweenState.Playing).toBe(false);
            expect(Tween.state[tweenEid!] === TweenState.Complete).toBe(true);
            expect(state.entityExists(tweenEid!)).toBe(true);
        });

        test("should not update tweens without Playing state", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(0, 1);

            Tween.state[tweenEid!] = TweenState.Playing;
            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(50, 1);
        });

        test("should handle multiple tweens on different fields", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;
            Position.y[target] = 0;

            const tween1 = createTween(state, target, "position.x", { to: 100, duration: 1 });
            const tween2 = createTween(state, target, "position.y", { to: 200, duration: 1 });
            Tween.state[tween1!] = TweenState.Playing;
            Tween.state[tween2!] = TweenState.Playing;

            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(50, 1);
            expect(Position.y[target]).toBeCloseTo(100, 1);
        });
    });

    describe("State-based controls", () => {
        test("should pause by setting state to IDLE", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(25, 1);

            Tween.state[tweenEid!] = TweenState.Idle;
            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(25, 1);
        });

        test("should resume by playing again", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(0, 1);

            Tween.state[tweenEid!] = TweenState.Playing;
            stepFor(state, 0.5);
            expect(Position.x[target]).toBeCloseTo(50, 1);
        });

        test("should remove by removeEntity", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(25, 1);

            state.removeEntity(tweenEid!);
            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(25, 1);
            expect(state.entityExists(tweenEid!)).toBe(false);
        });

        test("should restart by resetting state and elapsed", () => {
            const target = state.addEntity();
            state.addComponent(target, Position);
            Position.x[target] = 0;

            const tweenEid = createTween(state, target, "position.x", {
                to: 100,
                duration: 0.5,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.6);
            expect(Tween.state[tweenEid!] === TweenState.Complete).toBe(true);
            expect(Position.x[target]).toBeCloseTo(100, 1);

            Tween.state[tweenEid!] = TweenState.Idle;
            Tween.elapsed[tweenEid!] = 0;
            Tween.to[tweenEid!] = 200;
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(state, 0.25);
            expect(Position.x[target]).toBeCloseTo(150, 1);
        });
    });

    describe("Easing functions", () => {
        test("should have 31 easing functions", () => {
            expect(EASING_FUNCTIONS.length).toBe(31);
        });

        test("linear should return input unchanged", () => {
            const linear = getEasing(0);
            expect(linear(0)).toBe(0);
            expect(linear(0.5)).toBe(0.5);
            expect(linear(1)).toBe(1);
        });

        test("ease-out-quad should ease out", () => {
            const easeOutQuad = getEasing(getEasingIndex("ease-out-quad"));
            expect(easeOutQuad(0.5)).toBeGreaterThan(0.5);
        });

        test("ease-in-quad should ease in", () => {
            const easeInQuad = getEasing(getEasingIndex("ease-in-quad"));
            expect(easeInQuad(0.5)).toBeLessThan(0.5);
        });

        test("getEasingIndex should return 0 for unknown", () => {
            expect(getEasingIndex("unknown")).toBe(0);
        });

        test("getEasing should return linear for invalid index", () => {
            const fn = getEasing(999);
            expect(fn(0.5)).toBe(0.5);
        });
    });

    describe("kebab-case field paths", () => {
        test("should handle kebab-case field names", () => {
            const TestComp = { posX: [] as number[] };
            const TestCompPlugin: Plugin = { name: "TestComp", components: { TestComp } };
            state.register(TestCompPlugin);

            const target = state.addEntity();
            state.addComponent(target, TestComp);
            TestComp.posX[target] = 0;

            const tweenEid = createTween(state, target, "test-comp.pos-x", {
                to: 100,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            expect(tweenEid).not.toBeNull();

            stepFor(state, 0.5);
            expect(TestComp.posX[target]).toBeCloseTo(50, 1);
        });
    });

    describe("XML declarative loading", () => {
        test("should load tween from XML with field and target relation", () => {
            const nodes = parse(`
                <scene>
                    <a id="ball" position="x: 0" />
                    <a tween="to: 100; field: position.x; easing: ease-out-quad; duration: 0.5" target="@ball" />
                </scene>
            `);
            load(nodes, state);

            const tweenEids = [...state.query([Tween])];
            expect(tweenEids.length).toBe(1);
            const tweenEid = tweenEids[0];

            expect(Tween.to[tweenEid]).toBe(100);
            expect(Tween.easing[tweenEid]).toBe(getEasingIndex("ease-out-quad"));
            expect(Tween.duration[tweenEid]).toBe(0.5);
            expect(Tween.field[tweenEid]).toBe("position.x");

            const ballEids = [...state.query([Position])];
            expect(ballEids.length).toBe(1);
            const ballEid = ballEids[0];
            expect(state.hasRelation(tweenEid, Target, ballEid)).toBe(true);

            Tween.state[tweenEid] = TweenState.Playing;
            stepFor(state, 0.5);
            expect(Position.x[ballEid]).toBeCloseTo(100, 1);
        });
    });

    describe("proxy field accessors", () => {
        test("should tween euler values through quaternion proxy", async () => {
            clearRegistry();
            const transformState = await build({
                plugins: [TransformsPlugin, TweenPlugin],
                defaults: false,
            });

            const target = transformState.addEntity();
            transformState.addComponent(target, Transform);
            Transform.rotY[target] = 0;

            const tweenEid = createTween(transformState, target, "transform.rot-y", {
                to: 90,
                duration: 1,
            });
            Tween.state[tweenEid!] = TweenState.Playing;

            stepFor(transformState, 0.5);
            expect(Transform.rotY[target]).toBeCloseTo(45, 1);

            stepFor(transformState, 0.5);
            expect(Transform.rotY[target]).toBeCloseTo(90, 1);
        });
    });
});
