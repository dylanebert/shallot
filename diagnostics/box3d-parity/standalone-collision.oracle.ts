import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "../../src/project/testing.js";
import { nativeCollisionInput } from "../../src/standard/physics/standalone-collision.fixture";
import { nativeSseOutput } from "./native-evidence";

setDefaultTimeout(TIMEOUT.node);
test("standalone collision records equal native Box3D including GJK history", async () => {
    const native = nativeSseOutput("functions.c", nativeCollisionInput(), ["standalone collision"]);
    const gold = await Bun.file(new URL("../../src/standard/physics/standalone-collision.gold.txt", import.meta.url)).text();
    expect(native).toBe(gold);
});
