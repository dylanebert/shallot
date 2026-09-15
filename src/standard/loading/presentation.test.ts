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
        subject: [
            "src/standard/loading/presentation.ts",
            "src/standard/loading/presentation.test.ts",
        ],
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
    "responsive exits without awaiting the canonical animation",
    {
        claim: "responsive readiness after grace starts one whole-overlay exit without waiting for animation completion",
        subject: [
            "src/standard/loading/presentation.ts",
            "src/standard/loading/presentation.test.ts",
        ],
    },
    () => {
        let current = initialPresentation("responsive");
        current = step(current, "show").state;
        const mounted = step(current, "grace");
        current = mounted.state;
        expect(mounted.effects).toEqual(["mount-animation"]);

        const ready = step(current, "ready");
        expect(ready.effects).toEqual(["start-exit"]);
        expect(ready.state.phase).toBe("exiting");
        expect(ready.state.animationFinished).toBe(false);
    },
);

check(
    "cinematic waits for readiness and only unspent lockup rest",
    {
        claim: "cinematic requires readiness plus finished animation and lockup rest before its single exit",
        subject: [
            "src/standard/loading/presentation.ts",
            "src/standard/loading/presentation.test.ts",
        ],
    },
    () => {
        let current = initialPresentation("cinematic");
        const shown = step(current, "show");
        current = shown.state;
        expect(shown.effects).toEqual(["mount-animation"]);

        let result = step(current, "ready");
        current = result.state;
        expect(result.effects).toEqual([]);
        result = step(current, "animation-finished");
        current = result.state;
        expect(result.effects).toEqual([]);
        result = step(current, "rest-finished");
        expect(result.effects).toEqual(["start-exit"]);
    },
);

check(
    "compact and terminal mutations cannot leave live callbacks",
    {
        claim: "compact never starts animation and completion, error, and cleanup transition paths cancel pending effects",
        subject: [
            "src/standard/loading/presentation.ts",
            "src/standard/loading/presentation.test.ts",
        ],
    },
    () => {
        let current = initialPresentation("compact");
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
