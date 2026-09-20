// Chromium launch configuration, separate from seat policy. Display launches are headed because
// presenting on a monitor is their premise. A configured OS does not establish adapter or placement
// support; those require observation by the run.

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

/** Resolve display launch parameters for a configured OS, or refuse an unconfigured platform. */
export function launchPlan(host: string): LaunchPlan | { refused: string } {
    if (!hasLaunchPath(host)) {
        return {
            refused: `no headed Chromium launch configuration for platform ${host}; configured platforms are darwin, linux, win32`,
        };
    }
    return {
        host,
        seat: "display",
        channel: floor.channel as "chromium",
        args: floor.args,
    };
}
