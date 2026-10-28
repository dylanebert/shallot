import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bodyMoverTOIRows } from "./fixtures/body-mover-toi";

setDefaultTimeout(TIMEOUT.node);

test("body mover TOI matches native for convex shapes, earliest hit, miss, initial overlap, filters and both sweeps", async () => {
    expect(await bodyMoverTOIRows()).toBe(
        await Bun.file(new URL("./body-mover-toi.gold.txt", import.meta.url)).text(),
    );
});
