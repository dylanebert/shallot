/** The explicit presentation intent for a branded startup screen. */
export type SplashProfile = "responsive" | "cinematic" | "compact";

export type PresentationPhase =
    | "ground"
    | "grace"
    | "brand"
    | "ready"
    | "exiting"
    | "error"
    | "cleaned";

export interface PresentationState {
    readonly profile: SplashProfile;
    readonly reducedMotion: boolean;
    readonly phase: PresentationPhase;
    readonly ready: boolean;
    readonly branded: boolean;
    readonly animated: boolean;
    readonly animationFinished: boolean;
    readonly restFinished: boolean;
}

export type PresentationEvent =
    | { type: "show" }
    | { type: "grace" }
    | { type: "ready" }
    | { type: "animation-finished" }
    | { type: "rest-finished" }
    | { type: "error" }
    | { type: "cleanup" };

export type PresentationEffect = "mount-static" | "mount-animation" | "start-exit" | "cancel";

export function initialPresentation(
    profile: SplashProfile,
    reducedMotion = false,
): PresentationState {
    return {
        profile,
        reducedMotion,
        phase: "ground",
        ready: false,
        branded: false,
        animated: false,
        animationFinished: false,
        restFinished: false,
    };
}

/**
 * Advance the owner of splash policy without touching the DOM or a clock. Effects are the only
 * points where the Loading implementation may start or cancel an external callback.
 */
export function transitionPresentation(
    current: PresentationState,
    event: PresentationEvent,
): { state: PresentationState; effects: readonly PresentationEffect[] } {
    if (current.phase === "cleaned") return { state: current, effects: [] };
    if (event.type === "cleanup") {
        return { state: { ...current, phase: "cleaned" }, effects: ["cancel"] };
    }
    if (event.type === "error") {
        return {
            state: { ...current, phase: "error" },
            effects: ["cancel"],
        };
    }
    if (current.phase === "error") return { state: current, effects: [] };

    let state = current;
    const effects: PresentationEffect[] = [];
    const exit = (): void => {
        if (state.phase === "exiting") return;
        state = { ...state, phase: "exiting" };
        effects.push("start-exit");
    };

    switch (event.type) {
        case "show":
            if (current.profile === "cinematic" && current.reducedMotion) {
                state = { ...current, phase: "brand", branded: true };
                effects.push("mount-static");
            } else if (current.profile === "cinematic") {
                state = { ...current, phase: "brand", branded: true, animated: true };
                effects.push("mount-animation");
            } else {
                state = { ...current, phase: "grace" };
            }
            break;
        case "grace":
            if (current.ready) {
                state = { ...current, phase: "ready" };
            } else if (current.profile === "compact" || current.reducedMotion) {
                state = { ...current, phase: "brand", branded: true };
                effects.push("mount-static");
            } else {
                state = { ...current, phase: "brand", branded: true, animated: true };
                effects.push("mount-animation");
            }
            break;
        case "ready":
            state = { ...current, ready: true };
            if (current.profile !== "cinematic" && current.branded) exit();
            else if (
                current.profile === "cinematic" &&
                current.animationFinished &&
                current.restFinished
            )
                exit();
            else if (current.profile === "cinematic" && current.reducedMotion) exit();
            else if (!current.branded) state = { ...state, phase: "ready" };
            break;
        case "animation-finished":
            state = { ...current, animationFinished: true };
            if (current.profile === "cinematic" && current.ready && current.restFinished) exit();
            break;
        case "rest-finished":
            state = { ...current, restFinished: true };
            if (current.profile === "cinematic" && current.ready && current.animationFinished)
                exit();
            break;
    }

    return { state, effects };
}
