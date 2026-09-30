import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { Color, createApp } from "@dylanebert/shallot";
import { ColorTraits } from "./part";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const ColorOwner = {
    name: "ColorAuthoringOwner",
    components: { Color },
    traits: { Color: ColorTraits },
};

test("Color loads canonical rgba field syntax and rejects CSS-function syntax at the scene owner, so an authored invalid color cannot reach runtime silently", async () => {
    const valid = await createApp({
        defaults: false,
        plugins: [ColorOwner],
        scene: `<scene><a id="valid" color="rgba: 0.22 0.24 0.26" /></scene>`,
    });
    try {
        const eid = [...valid.world.query([Color])][0];
        if (eid === undefined) throw new Error("valid Color scene did not create a Color entity");
        const lanes = valid.world.storage(Color).rgba.read(eid, new Float32Array(4));
        const expected = [0.22, 0.24, 0.26, 1];
        if (lanes.some((value, index) => Math.abs(value - expected[index]) > 1e-6))
            throw new Error(`canonical rgba lanes were ${Array.from(lanes).join(", ")}`);
    } finally {
        valid.dispose();
    }

    await createApp({
        defaults: false,
        plugins: [ColorOwner],
        scene: `<scene><a color="rgba(0.22 0.24 0.26)" /></scene>`,
    }).then(
        (app) => {
            app.dispose();
            throw new Error("CSS-function Color syntax was accepted");
        },
        (error: unknown) => {
            if (
                !(error instanceof Error) ||
                !error.message.includes('expected "field: value" syntax')
            )
                throw error;
        },
    );
});
