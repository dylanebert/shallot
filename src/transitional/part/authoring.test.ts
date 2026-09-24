import { build, Color, SlabPlugin } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { ColorTraits } from "./part";

const ColorOwner = {
    name: "ColorAuthoringOwner",
    components: { Color },
    traits: { Color: ColorTraits },
    dependencies: [SlabPlugin],
};

check(
    "Color scene grammar accepts canonical rgba fields",
    {
        claim: "Color loads canonical rgba field syntax and rejects CSS-function syntax at the scene owner, so an authored invalid color cannot reach runtime silently",
    },
    async () => {
        const valid = await build({
            defaults: false,
            plugins: [ColorOwner],
            scene: `<scene><a id="valid" color="rgba: 0.22 0.24 0.26" /></scene>`,
        });
        try {
            const eid = [...valid.state.query([Color])][0];
            if (eid === undefined)
                throw new Error("valid Color scene did not create a Color entity");
            const lanes = Color.rgba.read(eid, new Float32Array(4));
            const expected = [0.22, 0.24, 0.26, 1];
            if (lanes.some((value, index) => Math.abs(value - expected[index]) > 1e-6))
                throw new Error(`canonical rgba lanes were ${Array.from(lanes).join(", ")}`);
        } finally {
            valid.dispose();
        }

        await build({
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
    },
);
