import { beforeEach, describe, expect, test } from "bun:test";
import { f32, load, parse, State, serialize, sparse, stringify } from "../..";
import { clear, register } from "../../engine/ecs/core";
import { AnimationPlugin, AnimationState, AnimationSystem, Animator, Playables, script } from ".";

const Position = { value: sparse(f32) };

function initialize() {
    clear();
    AnimationPlugin.initialize?.(new State());
    register("position", Position, { defaults: () => ({ value: 0 }) });
    for (const [name, component] of Object.entries(AnimationPlugin.components ?? {})) {
        register(name, component, AnimationPlugin.traits?.[name]);
    }
    Playables.register({
        name: "bounce",
        ...script((t, pose) => pose.set("position.value", t * t), 2),
    });
}

function step(state: State) {
    state.addSystem(AnimationSystem);
    state.step(0);
}

describe("AnimationPlugin", () => {
    beforeEach(initialize);

    test("initialize clears stale playable registrations", () => {
        expect(Playables.has("bounce")).toBe(true);
        AnimationPlugin.initialize?.(new State());
        expect(Playables.size).toBe(0);
    });

    test("an animator whose clip resolves to nothing warns once, in the form verify promotes to an error", () => {
        // the visualization and clips-recipe scenes shipped `animator="loop: 1; target: @x"` with no
        // `clip:`; the system skipped them silently and every gate stayed green on a static frame
        const warned: string[] = [];
        const original = console.warn;
        console.warn = (...args: unknown[]) => warned.push(args.map(String).join(" "));
        try {
            const state = new State();
            load(
                parse(
                    `<scene><a animator="target: @coin" /><a animator="clip: nope; target: @coin" /><a id="coin" position /></scene>`,
                ),
                state,
            );
            state.addSystem(AnimationSystem);
            state.step(0);
            state.step(0);
        } finally {
            console.warn = original;
        }
        // stepping twice proves the per-animator once
        expect(warned).toHaveLength(2);
        expect(warned.some((w) => /names no clip/.test(w))).toBe(true);
        expect(warned.some((w) => /"nope" is not registered/.test(w))).toBe(true);
    });

    test("scene clip names and @name targets bind a seeked pose", () => {
        const state = new State();
        load(
            parse(
                `<scene><a id="driver" animator="clip: bounce; target: @coin; time: 1.5; state: idle" /><a id="coin" position /></scene>`,
            ),
            state,
        );

        step(state);

        const coin = state.only([Position]);
        expect(Position.value.get(coin)).toBe(2.25);
    });

    test("seek → serialize → rebuild → seek produces the same component value", () => {
        const first = new State();
        load(
            parse(
                `<scene><a id="driver" animator="clip: bounce; target: @coin; state: idle" /><a id="coin" position /></scene>`,
            ),
            first,
        );
        const firstAnimator = first.only([Animator]);
        Animator.time.set(firstAnimator, 1.25);
        step(first);
        const firstCoin = first.only([Position]);
        const expected = Position.value.get(firstCoin);
        const saved = stringify(serialize(first));

        initialize();
        const rebuilt = new State();
        load(parse(saved), rebuilt);
        const rebuiltAnimator = rebuilt.only([Animator]);
        Animator.time.set(rebuiltAnimator, 1.25);
        step(rebuilt);
        const rebuiltCoin = rebuilt.only([Position]);

        expect(Position.value.get(rebuiltCoin)).toBe(expected);
        expect(Animator.state.get(rebuiltAnimator)).toBe(AnimationState.Idle);
    });

    test("serialize round-trips the animator target by scene name", () => {
        const state = new State();
        load(
            parse(
                `<scene><a id="driver" animator="clip: bounce; target: @coin; time: 0.5; scale: 2; loop: 1" /><a id="coin" position /></scene>`,
            ),
            state,
        );
        const xml = stringify(serialize(state));
        expect(xml).toContain("target: @coin");
        // the clip survives as its NAME, not an id: the interned id is process-order and the formatter
        // (`scripts/format.ts`) round-trips scenes with no playable registered at all
        expect(xml).toContain("clip: bounce");

        const reloaded = new State();
        load(parse(xml), reloaded);
        const animator = reloaded.only([Animator]);
        const coin = reloaded.only([Position]);
        expect(Animator.target.get(animator)).toBe(coin);
        expect(Animator.clip.get(animator)).toBe(Animator.clip.get(state.only([Animator])));
        expect(Animator.time.get(animator)).toBe(0.5);
        expect(Animator.scale.get(animator)).toBe(2);
        expect(Animator.loop.get(animator)).toBe(1);
    });
});
