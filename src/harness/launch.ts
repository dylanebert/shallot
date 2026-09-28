// A configured OS permits a headed Chromium display launch, not a claim about adapter identity or
// monitor placement; the run observes both.

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
        channel: "chromium",
    };
}
