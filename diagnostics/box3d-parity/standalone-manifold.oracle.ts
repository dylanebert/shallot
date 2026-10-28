import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { TIMEOUT } from "../../src/project/testing.js";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { manifoldRecords } from "../../src/standard/physics/standalone-manifold.fixture";
import { nativeSseOutput } from "./native-evidence";
setDefaultTimeout(TIMEOUT.node);
test("public manifold fields and warm caches equal native Box3D", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const records = manifoldRecords(app.world);
        const native = nativeSseOutput("functions.c", records.input);
        expect(records.output).toBe(native);
        expect(native).toBe(await Bun.file(new URL("../../src/standard/physics/standalone-manifold.gold.txt", import.meta.url)).text());
    } finally { app.dispose(); }
});
