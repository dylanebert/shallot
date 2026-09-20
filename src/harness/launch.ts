// Operational launch dispatch, kept separate from seat policy. The display seat is always headed,
// because presenting on a monitor is its premise. Adapter support is observed by the run and resolved by
// seat.ts; this module only chooses how a supported OS launches Chromium.

import floor from "./browser.json" with { type: "json" };

function hasLaunchPath(host: string): boolean {
    switch (host) {
        case "darwin":
        case "linux":
        case "win32":
            return true;
        default:
            return false;
    }
}

/** a resolved headed launch of the display seat, with only operational browser data. */
export interface LaunchPlan {
    host: string;
    seat: "display";
    channel: "chromium";
    args: readonly string[];
}

/** Resolve a supported host's display launch plan, or refuse an unsupported host. */
export function launchPlan(host: string): LaunchPlan | { refused: string } {
    if (!hasLaunchPath(host)) {
        return {
            refused: `no operational headed Chromium launch path for host ${host}; supported hosts are darwin, linux, win32`,
        };
    }
    return {
        host,
        seat: "display",
        channel: floor.channel as "chromium",
        args: floor.args,
    };
}
