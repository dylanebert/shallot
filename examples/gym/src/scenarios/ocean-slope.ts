import {
    Camera,
    GlazePlugin,
    InputPlugin,
    MirrorPlugin,
    OrbitPlugin,
    PartPlugin,
    ProfilePlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { getSlopeTexture, OceanPlugin, SLOPE_MIP_LEVELS } from "@dylanebert/shallot-ocean";
import { type Check, frames, register, type Scenario } from "../gym";

const scene = `<scene>
    <a camera="clear-color: 0x14202a" sear orbit="distance: 8; yaw: 0.5; pitch: 0.2" transform />
</scene>`;

const scenario: Scenario = {
    name: "ocean-slope",
    noRender: true,
    params: [],
    async build() {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                RenderPlugin,
                PartPlugin,
                InputPlugin,
                OrbitPlugin,
                SearPlugin,
                GlazePlugin,
                OceanPlugin,
            ],
            scene,
        });
        await frames(3);
        return { state, dispose };
    },
    async assert(state): Promise<Check[]> {
        const camera = [...state.query([Camera, Sear, Transform])].length;
        const texture = getSlopeTexture();
        const ok = camera === 1 && texture !== null && texture.mipLevelCount === SLOPE_MIP_LEVELS;
        return [
            {
                name: "ocean slope product path",
                pass: ok,
                detail: `camera=${camera}, mips=${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`,
            },
        ];
    },
    live(): string {
        const texture = getSlopeTexture();
        return `ocean-slope\nmips ${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`;
    },
};

register(scenario);
