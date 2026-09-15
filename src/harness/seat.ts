// Seat policy: which capability a requirement tag actually names, and why a host that lacks it refuses.
// Pure and host-independent — every fact reaches these functions as data, so the same policy resolves a
// macOS run, a fixture and a host nobody has run yet. Nothing here launches, probes or reads the
// environment; `./launch` holds the per-host launch declarations and `./driver` does the observing.

import { type AdapterFacts, classifyAdapter } from "../engine/runtime/adapter";
import { type CaptureIdentity, captureIdentityLabel, captureIdentityMatches } from "./capture";
import { LAUNCH_MODES, type LaunchPlan } from "./launch";

// Keep the harness seat module's public imports stable while the engine owns adapter policy.
export {
    type AdapterClass,
    type AdapterFacts,
    type AdapterInfoFacts,
    type AdapterVerdict,
    adapterIdentity,
    classifyAdapter,
} from "../engine/runtime/adapter";

/** the seats a requirement tag names. `cpu` is the absence of a requirement, not a tag. */
export const SEATS = ["cpu", "gpu", "chromium", "display"] as const;
export type Seat = (typeof SEATS)[number];

/** the capture geometry an observer actually saw, against which the contract is checked. */
export interface CaptureFacts {
    identity: CaptureIdentity;
}

/** the browser seat's observed facts: how it was launched, what adapter it reached, what it captured. */
export interface BrowserFacts {
    launch?: LaunchPlan;
    adapter?: AdapterFacts;
    capture?: CaptureFacts;
}

/** a declared headed display and the headed browser observed on it. */
export interface DisplayFacts {
    /** the host's own declaration of its display, from `SHALLOT_DISPLAY_SEAT`. */
    source: string;
    /** the headed browser launched on that display: its launch plan and the adapter it reached. */
    browser?: BrowserFacts;
}

/**
 * every fact a seat resolution may read. Each seat reads only its own field, which is what keeps one seat
 * from silently standing in for another: a real `device` adapter cannot satisfy `chromium`, a headless
 * browser cannot satisfy `display`, and a headed browser cannot satisfy `chromium`.
 */
export interface SeatFacts {
    /** the in-process (Bun) WebGPU adapter, for the `gpu` seat. */
    device?: AdapterFacts;
    /** the headless browser seat, for `chromium`. */
    browser?: BrowserFacts;
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
 * - `chromium` — a declared headless launch, a positively identified real adapter inside that browser, and
 *   a capture at the one declared identity. A headed launch never grants it.
 * - `display` — a host-declared display and a headed launch on it that reaches a positively identified
 *   real adapter. The declaration alone never grants it, and neither does a headed browser alone.
 *
 * @example const seat = resolveSeat("gpu", { device: { present: true, info } });
 */
export function resolveSeat(
    seat: Seat,
    facts: SeatFacts,
    contract?: CaptureIdentity,
): SeatResolution {
    if (seat === "cpu") return { ok: true, detail: "cpu" };
    if (seat === "gpu") {
        if (facts.device === undefined)
            return refuse(seat, "no in-process WebGPU device was probed");
        const adapter = classifyAdapter(facts.device);
        if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
        return { ok: true, detail: `real device ${adapter.identity}` };
    }
    if (seat === "display") {
        const display = facts.display;
        if (display === undefined) return refuse(seat, "no headed display is declared");
        const launch = display.browser?.launch;
        if (launch === undefined || LAUNCH_MODES[launch.seat] !== "headed") {
            return refuse(
                seat,
                `no headed Chromium launch was observed on the declared display ${display.source}`,
            );
        }
        if (display.browser?.adapter === undefined) {
            return refuse(seat, "the headed browser reported no adapter observation");
        }
        const adapter = classifyAdapter(display.browser.adapter);
        if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
        return {
            ok: true,
            detail: `headed chromium on real adapter ${adapter.identity} via ${display.source}`,
        };
    }
    const browser = facts.browser;
    if (browser?.launch === undefined) {
        return refuse(seat, "no declared headless Chromium launch path for this host");
    }
    const mode = LAUNCH_MODES[browser.launch.seat];
    if (mode !== "headless") {
        return refuse(seat, `a ${mode} launch never grants the chromium seat, which runs headless`);
    }
    if (browser.adapter === undefined) {
        return refuse(seat, "the browser reported no adapter observation");
    }
    const adapter = classifyAdapter(browser.adapter);
    if (adapter.class !== "real") return refuse(seat, adapter.reason ?? adapter.class);
    if (contract !== undefined) {
        if (browser.capture === undefined) {
            return refuse(seat, "the browser reported no capture at the declared capture contract");
        }
        if (!captureIdentityMatches(browser.capture.identity, contract)) {
            return refuse(
                seat,
                `capture identity ${captureIdentityLabel(browser.capture.identity)} is not the declared contract ${captureIdentityLabel(contract)}`,
            );
        }
    }
    return {
        ok: true,
        detail: `${mode} chromium on real adapter ${adapter.identity}`,
    };
}
