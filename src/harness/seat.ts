// Seat policy: which capability a requirement tag actually names, and why a host that lacks it refuses.
// Pure and host-independent — every fact reaches these functions as data, so the same policy resolves a
// fixture and a host nobody has run yet. Nothing here launches, probes or reads the environment; `./launch`
// holds the per-host launch declarations.

import { type AdapterFacts, classifyAdapter } from "../engine/runtime/adapter";
import type { LaunchPlan } from "./launch";

/** the seats this policy resolves. `cpu` is the absence of a requirement, not a tag. */
export type Seat = "cpu" | "gpu" | "display";

/** a declared headed display and the headed browser observed on it. */
export interface DisplayFacts {
    /** the host's own declaration of its display, from `SHALLOT_DISPLAY_SEAT`. */
    source: string;
    /** the headed browser launched on that display: its launch plan and the adapter it reached. */
    browser?: { launch?: LaunchPlan; adapter?: AdapterFacts };
}

/**
 * every fact a seat resolution may read. Each seat reads only its own field, so a real in-process device
 * cannot satisfy `display` and a headed browser cannot satisfy `gpu`.
 */
export interface SeatFacts {
    /** the in-process (Bun) WebGPU adapter, for the `gpu` seat. */
    device?: AdapterFacts;
    /** the declared display and its headed browser, for `display`. */
    display?: DisplayFacts;
}

/** a seat resolution: available, or refused with the reason a verdict prints. */
export type SeatResolution = { ok: true; detail: string } | { ok: false; reason: string };

function refuse(seat: Seat, reason: string): SeatResolution {
    return { ok: false, reason: `${seat} seat unavailable: ${reason}` };
}

/**
 * Resolve one seat from observed facts.
 *
 * - `cpu` — no requirement; always available.
 * - `gpu` — a real in-process WebGPU device. Reads no browser fact, so a browser claim never grants it.
 * - `display` — a host-declared display and a headed launch on it that reaches a positively identified
 *   real adapter. The declaration alone never grants it, and neither does a headed browser alone.
 *
 * @example const seat = resolveSeat("gpu", { device: { present: true, info } });
 */
export function resolveSeat(seat: Seat, facts: SeatFacts): SeatResolution {
    if (seat === "cpu") return { ok: true, detail: "cpu" };
    if (seat === "gpu") {
        if (facts.device === undefined)
            return refuse(seat, "no in-process WebGPU device was probed");
        const adapter = classifyAdapter(facts.device);
        if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
        return { ok: true, detail: `real device ${adapter.identity}` };
    }
    const display = facts.display;
    if (display === undefined) return refuse(seat, "no headed display is declared");
    if (display.browser?.launch === undefined) {
        return refuse(
            seat,
            `no headed Chromium launch was observed on the declared display ${display.source}`,
        );
    }
    if (display.browser.adapter === undefined) {
        return refuse(seat, "the headed browser reported no adapter observation");
    }
    const adapter = classifyAdapter(display.browser.adapter);
    if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
    return {
        ok: true,
        detail: `headed chromium on real adapter ${adapter.identity} via ${display.source}`,
    };
}
