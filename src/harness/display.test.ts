import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import {
    declaredMonitor,
    parseMonitors,
    parseWindows,
    placementRefusal,
    ruleScript,
    SEAT_RULE_NAME,
    SEAT_WINDOW_CLASS,
    type SeatMonitor,
    type SeatWindow,
} from "./display";

// One live two-monitor desktop as the compositor reports it: the same pair whose refresh rates made one
// 120-frame window 500 ms on DP-1 and 833 ms on DP-2.
const MONITORS = JSON.stringify([
    {
        id: 0,
        name: "DP-1",
        width: 2560,
        height: 1440,
        refreshRate: 239.97,
        x: 2560,
        y: 0,
        scale: 1,
        activeWorkspace: { id: 1, name: "1" },
    },
    {
        id: 1,
        name: "DP-2",
        width: 2560,
        height: 1440,
        refreshRate: 143.912,
        x: 0,
        y: 0,
        scale: 1,
        activeWorkspace: { id: 2, name: "2" },
    },
]);

const WINDOWS = JSON.stringify([
    {
        address: "0xpage",
        class: "shallot-display-seat",
        monitor: 0,
        workspace: { id: 1, name: "1" },
        size: [640, 360],
        floating: true,
        mapped: true,
        hidden: false,
    },
    {
        address: "0xother",
        class: "discord",
        monitor: 1,
        workspace: { id: 2, name: "2" },
        size: [2536, 1390],
        floating: false,
        mapped: true,
        hidden: false,
    },
]);

const monitor = (name: string): SeatMonitor => {
    const found = declaredMonitor(parseMonitors(MONITORS), name);
    if ("refused" in found) throw new Error(found.refused);
    return found;
};
const page = (): SeatWindow => {
    const found = parseWindows(WINDOWS).find((row) => row.address === "0xpage");
    if (found === undefined) throw new Error("no page window in the fixture");
    return found;
};

check(
    "the display seat reads its monitor from the compositor",
    {
        claim: "the declared display resolves to the compositor's own geometry and refresh rate, and a declaration naming no live monitor refuses with the live names",
    },
    () => {
        expect(monitor("DP-1")).toEqual({
            name: "DP-1",
            id: 0,
            x: 2560,
            y: 0,
            width: 2560,
            height: 1440,
            refreshRate: 239.97,
            scale: 1,
            activeWorkspace: 1,
        });
        expect(monitor("DP-2").refreshRate).toBe(143.912);
        // A declaration the compositor cannot place is a premise failure, not a measurement: the run must
        // not measure some other display and print this name beside it.
        const refused = declaredMonitor(parseMonitors(MONITORS), "HDMI-A-1");
        expect("refused" in refused && refused.refused).toBe(
            "SHALLOT_DISPLAY_SEAT declares HDMI-A-1, which is not a live monitor; the compositor reports DP-1, DP-2",
        );
        const none = declaredMonitor(parseMonitors("[]"), "DP-1");
        expect("refused" in none && none.refused).toContain("the compositor reports none");
    },
);

check(
    "a page that did not land on the declared monitor refuses by name",
    {
        claim: "the placement check passes only a mapped, unhidden window on the declared monitor's shown workspace, and names which of those failed otherwise",
    },
    () => {
        const dp1 = monitor("DP-1");
        // The one passing shape: the page is where the declaration says, on the workspace that monitor is
        // showing, mapped and visible.
        expect(placementRefusal(dp1, page(), "after being placed")).toBeUndefined();
        // Tiled onto the other monitor, which is what happened before the page was placed at all.
        expect(placementRefusal(dp1, { ...page(), monitor: 1 }, "after being placed")).toBe(
            "the page's window is on monitor 1 after being placed, not the declared DP-1 (monitor 0), so the window it presented on is not the one this seat declares",
        );
        // Still on the declared monitor, but the person switched that monitor to another workspace, so the
        // page is no longer presenting on it.
        expect(placementRefusal(dp1, { ...page(), workspace: 7 }, "after the windows")).toBe(
            "the page's window is on workspace 7 after the windows while DP-1 is showing workspace 1, so the page is not presenting on the declared monitor",
        );
        expect(placementRefusal(dp1, { ...page(), hidden: true }, "after the windows")).toContain(
            "hidden on DP-1",
        );
        expect(placementRefusal(dp1, { ...page(), mapped: false }, "after the windows")).toContain(
            "unmapped on DP-1",
        );
        expect(placementRefusal(dp1, undefined, "after the windows")).toContain("is gone");
    },
);

check(
    "the placement opens the page on the declared monitor",
    {
        claim: "the placement rule sends only this harness's own window class to the declared monitor, as a floating window at the capture contract's size, and one rule of a stable name replaces the last rather than accumulating",
    },
    () => {
        const script = ruleScript(monitor("DP-1"), [1280, 720]);
        expect(script).toContain(`name = "${SEAT_RULE_NAME}"`);
        // Only this harness's Chromium carries this class, so no window of the person's can match it.
        expect(script).toContain(`match = { class = "^(${SEAT_WINDOW_CLASS})$" }`);
        expect(script).toContain('monitor = "DP-1"');
        // Floating at the measured size: the page is shown as it is measured, and the workspace it lands on
        // is not retiled around it.
        expect(script).toContain("float = true");
        expect(script).toContain('size = "1280 720"');
    },
);
