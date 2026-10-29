import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(0);
test("dynamic tree gold is Box3D's independent tree construction, rebuilds, queries, reports and scaled file load", async () => {
    expect(nativeSseOutput("tree.c", "")).toBe(await Bun.file(new URL("../../src/standard/physics/tree.gold.txt", import.meta.url)).text());
});
