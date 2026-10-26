import { setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { categories } from "../src/scenes";
import { expectSwitchesEqualFresh } from "./switch";

setDefaultTimeout(TIMEOUT.node);

// Benchmark scenes hold scene parity by manual oracle; this check holds the switcher.
// Bridge is held separately by switch-bridge.node.ts.
test("switching the critical set equals fresh scenes through 600 steps after a changed, grabbed Bridge", () =>
    expectSwitchesEqualFresh(
        categories
            .filter((category) => category.name !== "Benchmark")
            .flatMap((category) => category.scenes.map((scene) => scene.id))
            .filter((scene) => scene !== "bridge"),
    ));
