import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "../../src/project/testing.js";
import { nativeEdgeInput } from "../../src/standard/physics/standalone-collision-edge.fixture";
import { nativeCollisionInput } from "../../src/standard/physics/standalone-collision.fixture";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(TIMEOUT.node);
test("standalone collision records equal native Box3D including GJK history", async () => {
    const native = nativeSseOutput("functions.c", nativeCollisionInput(), ["standalone collision"]);
    const gold = await Bun.file(new URL("../../src/standard/physics/standalone-collision.gold.txt", import.meta.url)).text();
    expect(native).toBe(gold);
    const edgeNative = nativeSseOutput("functions.c", nativeEdgeInput(), ["standalone collision edges"]);
    const edgeGold = await Bun.file(new URL("../../src/standard/physics/standalone-collision-edge.gold.txt", import.meta.url)).text();
    expect(edgeNative).toBe(edgeGold);
});
