import { expect } from "bun:test";
import { TICK_RATE } from "../../brand";
import { check } from "../../harness/check";
import {
    initialPresentation,
    type PresentationState,
    transitionPresentation,
} from "./presentation";

function step(
    state: PresentationState,
    type: Parameters<typeof transitionPresentation>[1]["type"],
): {
    state: PresentationState;
    effects: readonly string[];
} {
    return transitionPresentation(state, { type } as Parameters<typeof transitionPresentation>[1]);
}

check(
    "responsive readiness before grace never mounts branded content",
    {
        claim: "responsive readiness before grace leaves the overlay ground-only and cancels the delayed brand mount",
        subject: "src/standard/loading/presentation.ts",
    },
    () => {
        expect(TICK_RATE).toBe(35);
        let current = initialPresentation("responsive");
        let result = step(current, "show");
        current = result.state;
        expect(current.phase).toBe("grace");
        expect(result.effects).toEqual([]);

        result = step(current, "ready");
        current = result.state;
        expect(current.phase).toBe("ready");
        expect(current.branded).toBe(false);
        expect(result.effects).toEqual([]);

        result = step(current, "grace");
        expect(result.effects).toEqual([]);
        expect(result.state.branded).toBe(false);
    },
);

check(
    "reduced motion preserves responsive grace",
    {
        claim: "reduced-motion responsive readiness before grace never mounts branded content",
        subject: "src/standard/loading/presentation.ts",
    },
    () => {
        let current = initialPresentation("responsive", true);
        let result = step(current, "show");
        current = result.state;
        expect(current.phase).toBe("grace");
        expect(result.effects).toEqual([]);

        result = step(current, "ready");
        expect(result.state.branded).toBe(false);
        expect(result.effects).toEqual([]);
    },
);

check(
    "responsive mounts one complete static lockup after grace",
    {
        claim: "responsive grace mounts one static brand and readiness starts its whole-overlay exit without animation completion",
        subject: "src/standard/loading/presentation.ts",
    },
    () => {
        let current = initialPresentation("responsive");
        current = step(current, "show").state;
        const mounted = step(current, "grace");
        current = mounted.state;
        expect(mounted.effects).toEqual(["mount-static"]);
        expect(current.branded).toBe(true);
        expect(current.animated).toBe(false);
        expect(current.animationFinished).toBe(false);

        const ready = step(current, "ready");
        expect(ready.effects).toEqual(["start-exit"]);
        expect(ready.state.phase).toBe("exiting");
    },
);

check(
    "cinematic waits for readiness and only unspent lockup rest",
    {
        claim: "cinematic requires readiness plus finished animation and lockup rest before its single exit",
        subject: "src/standard/loading/presentation.ts",
    },
    () => {
        const orders: ReadonlyArray<readonly ("ready" | "animation-finished" | "rest-finished")[]> =
            [
                ["ready", "animation-finished", "rest-finished"],
                ["ready", "rest-finished", "animation-finished"],
                ["animation-finished", "ready", "rest-finished"],
                ["animation-finished", "rest-finished", "ready"],
                ["rest-finished", "ready", "animation-finished"],
                ["rest-finished", "animation-finished", "ready"],
            ];

        for (const order of orders) {
            let current = initialPresentation("cinematic");
            const shown = step(current, "show");
            current = shown.state;
            expect(shown.effects).toEqual(["mount-animation"]);

            for (const event of order.slice(0, -1)) {
                const result = step(current, event);
                current = result.state;
                expect(result.effects).toEqual([]);
            }
            expect(step(current, order[2] as (typeof order)[number]).effects).toEqual([
                "start-exit",
            ]);
        }
    },
);

check(
    "responsive and terminal mutations cannot leave live callbacks",
    {
        claim: "responsive never starts animation and completion, error, and cleanup transition paths cancel pending effects",
        subject: "src/standard/loading/presentation.ts",
    },
    () => {
        let current = initialPresentation("responsive");
        current = step(current, "show").state;
        const mounted = step(current, "grace");
        current = mounted.state;
        expect(mounted.effects).toEqual(["mount-static"]);
        expect(current.animated).toBe(false);

        const errored = step(current, "error");
        expect(errored.state.phase).toBe("error");
        expect(errored.effects).toEqual(["cancel"]);
        expect(step(errored.state, "cleanup").effects).toEqual(["cancel"]);

        const cleaned = step(current, "cleanup");
        expect(cleaned.state.phase).toBe("cleaned");
        expect(cleaned.effects).toEqual(["cancel"]);
        expect(step(cleaned.state, "ready").effects).toEqual([]);
    },
);
