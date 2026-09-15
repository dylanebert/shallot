import { shallotDark } from "@dylanebert/shallot";
import { END_TICK } from "@dylanebert/shallot/brand";
import type { Check, Verdict } from "@dylanebert/shallot/harness";

let now = 0;
let nextTimer = 1;
const timers = new Map<number, { at: number; callback: () => void }>();

Object.defineProperty(window.performance, "now", { configurable: true, value: () => now });
window.setTimeout = ((callback: TimerHandler, delay = 0, ...args: unknown[]) => {
    const id = nextTimer++;
    timers.set(id, {
        at: now + Math.max(0, Number(delay) || 0),
        callback: () => {
            if (typeof callback === "function") callback(...args);
        },
    });
    return id;
}) as typeof window.setTimeout;
window.clearTimeout = ((id: number) => {
    timers.delete(id);
}) as typeof window.clearTimeout;
window.requestAnimationFrame = ((callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(now), 1000 / 60)) as typeof window.requestAnimationFrame;
window.cancelAnimationFrame = window.clearTimeout;

function settle(): Promise<void> {
    return Promise.resolve().then(() => Promise.resolve());
}

async function advance(ms: number): Promise<void> {
    const target = now + ms;
    while (true) {
        const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > target) break;
        now = next[1].at;
        timers.delete(next[0]);
        next[1].callback();
        await settle();
    }
    now = target;
    await settle();
}

function mount(profile?: "responsive" | "cinematic" | "compact") {
    const host = document.createElement("div");
    host.style.position = "relative";
    document.body.append(host);
    const loading = shallotDark({ container: host, ...(profile === undefined ? {} : { profile }) });
    const cleanup = loading.show();
    if (!cleanup) throw new Error("loading did not mount in Chromium");
    return { host, loading, cleanup };
}

function hasBrand(host: HTMLElement): boolean {
    return host.querySelector('svg[shape-rendering="crispEdges"]') !== null;
}

window.__harness = {
    ready: true,
    async run(): Promise<Verdict> {
        const checks: Check[] = [];
        const record = (name: string, ok: boolean, detail: string) => {
            checks.push({ name, ok, detail });
        };

        const fast = mount();
        record(
            "ground mounts before branded content",
            fast.host.firstElementChild !== null && !hasBrand(fast.host),
            "the overlay ground exists before the responsive grace",
        );
        fast.loading.update(0.4);
        record(
            "progress remains build-driven",
            [...fast.host.querySelectorAll("div")].some((element) => element.style.width === "40%"),
            "the progress bar reflects the authored progress value",
        );
        await advance(149);
        record("default equals responsive", !hasBrand(fast.host), "default remains in grace");
        const fastDone = fast.loading.complete?.();
        if (fastDone) await fastDone;
        record("ready before grace skips brand", !hasBrand(fast.host), "responsive has no flash");
        fast.cleanup();
        await advance(1000);
        record(
            "cleanup cancels delayed mount",
            fast.host.firstElementChild === null,
            "no overlay returned",
        );

        const responsive = mount("responsive");
        await advance(150);
        record("grace mounts responsive animation", hasBrand(responsive.host), "one branded mount");
        const responsiveDone = responsive.loading.complete?.();
        if (!responsiveDone) throw new Error("responsive Loading omitted complete");
        record(
            "responsive exits as a whole overlay",
            (responsive.host.firstElementChild as HTMLElement).style.opacity === "0",
            "the game is beneath a 150ms opacity handoff",
        );
        await advance(149);
        record(
            "fade remains cancellable",
            responsive.host.firstElementChild !== null,
            "fade is not a hold",
        );
        await advance(1);
        await responsiveDone;
        responsive.cleanup();
        record(
            "responsive cleanup removes overlay",
            responsive.host.firstElementChild === null,
            "overlay removed",
        );

        const cinematic = mount("cinematic");
        const cinematicDone = cinematic.loading.complete?.();
        if (!cinematicDone) throw new Error("cinematic Loading omitted complete");
        await advance((END_TICK + 2) * (1000 / 35));
        record(
            "cinematic waits for finished lockup rest",
            (cinematic.host.firstElementChild as HTMLElement).style.opacity !== "0",
            "readiness alone does not dismiss cinematic",
        );
        await advance(150);
        record(
            "cinematic starts its exit after rest",
            (cinematic.host.firstElementChild as HTMLElement).style.opacity === "0",
            "single exit",
        );
        await advance(150);
        await cinematicDone;
        cinematic.cleanup();

        const error = mount();
        error.loading.error?.(new Error("controlled startup failure"));
        record(
            "errors bypass grace",
            error.host.textContent?.includes("controlled startup failure") === true,
            "error stays readable",
        );
        await advance(1000);
        record("error does not remount brand", !hasBrand(error.host), "error owns the overlay");
        error.cleanup();

        const previousMatchMedia = window.matchMedia;
        window.matchMedia = (() => ({ matches: true })) as unknown as typeof window.matchMedia;
        const reduced = mount("responsive");
        record(
            "reduced motion is static",
            hasBrand(reduced.host),
            "finished mark mounts without grace animation",
        );
        const reducedDone = reduced.loading.complete?.();
        if (reducedDone) await reducedDone;
        record(
            "reduced motion has no animated fade",
            (reduced.host.firstElementChild as HTMLElement).style.opacity !== "0",
            "no decorative dwell",
        );
        reduced.cleanup();
        window.matchMedia = previousMatchMedia;

        const ok = checks.every((check) => check.ok);
        return { ok, checks, tick: Math.round(now) };
    },
};
