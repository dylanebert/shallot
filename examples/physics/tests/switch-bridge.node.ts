import { setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expectSwitchesEqualFresh } from "./switch";

setDefaultTimeout(TIMEOUT.node);

test("a reselected Bridge equals a fresh Bridge through 600 steps after it was changed and grabbed", () =>
    expectSwitchesEqualFresh(["bridge"]));
