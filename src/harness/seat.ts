// Seat policy: which capability a requirement tag actually names, and why a host that lacks it refuses.
// Pure and host-independent — every fact reaches these functions as data, so the same policy resolves a
// macOS run, a fixture and a host nobody has run yet. Nothing here launches, probes or reads the
// environment; `./launch` holds the per-host launch declarations and `./driver` does the observing.

import { type AdapterFacts, classifyAdapter } from "../engine/runtime/adapter";
import { type CaptureIdentity, captureIdentityLabel, captureIdentityMatches } from "./capture";
import { HIDDEN_WINDOW_CLASS, type LaunchPlan, launchMode } from "./launch";

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

/** one window as the compositor reports it, for the hidden read-back. */
export interface HiddenWindow {
    address: string;
    class: string;
    /** the workspace's name; a special workspace's name starts with `special:`. */
    workspace: string;
}

/** what the compositor reported once a headed `chromium` launch opened its page. */
export interface HiddenFacts {
    /** every window the compositor reports. */
    windows: readonly HiddenWindow[];
    /** the class of the window that holds focus, empty when none does. */
    activeClass: string;
}

/** the browser seat's observed facts: how it was launched, what adapter it reached, what it captured. */
export interface BrowserFacts {
    launch?: LaunchPlan;
    /** the compositor read-back, for a headed launch. */
    hidden?: HiddenFacts;
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
 * browser cannot satisfy `display`, and a visible headed browser cannot satisfy `chromium`.
 */
export interface SeatFacts {
    /** the in-process (Bun) WebGPU adapter, for the `gpu` seat. */
    device?: AdapterFacts;
    /** the browser seat that takes nothing from the desktop, for `chromium`. */
    browser?: BrowserFacts;
    /** the declared display and its headed browser, for `display`. */
    display?: DisplayFacts;
}

/**
 * Why a headed launch's windows are not hidden from the person's desktop, or undefined when they are. A
 * rule is a request: every window of {@link HIDDEN_WINDOW_CLASS} must be on a special workspace and none
 * may hold focus, and each way it is not reads back as its own reason.
 */
export function hiddenRefusal(facts: HiddenFacts | undefined): string | undefined {
    const missing = `the compositor must hold a rule sending class ${HIDDEN_WINDOW_CLASS} to a silent special workspace with no focus`;
    if (facts === undefined)
        return `no compositor read-back of the headed window of class ${HIDDEN_WINDOW_CLASS}; ${missing}`;
    const windows = facts.windows.filter((window) => window.class === HIDDEN_WINDOW_CLASS);
    if (windows.length === 0)
        return `the compositor reports no window of class ${HIDDEN_WINDOW_CLASS}; ${missing}`;
    const shown = windows.filter((window) => !window.workspace.startsWith("special:"));
    if (shown.length > 0)
        return `the compositor reports ${shown.length} of ${windows.length} windows of class ${HIDDEN_WINDOW_CLASS} off a special workspace (${shown.map((window) => `${window.address} on ${window.workspace}`).join(", ")}); ${missing}`;
    if (facts.activeClass === HIDDEN_WINDOW_CLASS)
        return `the compositor reports the active window is of class ${HIDDEN_WINDOW_CLASS}; ${missing}`;
    return undefined;
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
 * - `chromium` — a declared launch in the mode its host's evidence gives, a positively identified real
 *   adapter inside that browser, and a capture at the one declared identity. A headed launch grants it only
 *   carrying the hidden window class and read back hidden by the compositor; a `display` launch never.
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
        if (launch === undefined || launch.seat !== "display") {
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
        return refuse(seat, "no declared Chromium launch path for this host");
    }
    if (browser.launch.seat !== "chromium") {
        return refuse(seat, `a ${browser.launch.seat} launch never grants the chromium seat`);
    }
    const mode = launchMode(browser.launch.host, browser.launch.seat);
    if (mode === "headed") {
        const hidden = hiddenRefusal(browser.hidden);
        if (hidden !== undefined) return refuse(seat, hidden);
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
