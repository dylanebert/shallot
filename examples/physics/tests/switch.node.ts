import { setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { type Scene, scenes } from "../src/scenes";
import { expectSwitchesEqualFresh } from "./switch";

setDefaultTimeout(TIMEOUT.node);

// Bridge, the costliest destination, is held by switch-bridge.node.ts so each file fits its budget.
test("every other switched scene equals a fresh scene through 600 steps after a changed, grabbed Bridge", () =>
    expectSwitchesEqualFresh(
        (Object.keys(scenes) as Scene[]).filter((scene) => scene !== "bridge"),
    ));
