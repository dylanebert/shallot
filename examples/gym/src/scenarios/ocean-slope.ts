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
import {
    generateH0,
    getSlopeTexture,
    OceanPlugin,
    readSlopeMips,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipAgreement,
    slopeMipSize,
    slopeSourceErrorBound,
} from "@dylanebert/shallot-ocean";
import { type Check, frames, register, type Scenario } from "../gym";

const scene = `<scene>
    <a camera="clear-color: 0x14202a" sear orbit="distance: 8; yaw: 0.5; pitch: 0.2" transform />
</scene>`;

let gpuMips: Float32Array[] | null = null;

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
        gpuMips = await readSlopeMips(SLOPE_CASCADE_CONFIGS[0], 0.73);
        return { state, dispose };
    },
    async assert(state): Promise<Check[]> {
        const camera = [...state.query([Camera, Sear, Transform])].length;
        const texture = getSlopeTexture();
        const checks: Check[] = [
            {
                name: "ocean slope product path",
                pass:
                    camera === 1 && texture !== null && texture.mipLevelCount === SLOPE_MIP_LEVELS,
                detail: `camera=${camera}, mips=${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`,
            },
        ];
        if (!texture || !gpuMips) return checks;

        const config = SLOPE_CASCADE_CONFIGS[0];
        const cpu = runSlopeCpuPipeline(generateH0(config, 0), config, 0.73);
        const sourceError = Math.max(
            slopeSourceErrorBound(cpu.x, config.N),
            slopeSourceErrorBound(cpu.z, config.N),
        );
        const level0 = new Float32Array(config.N * config.N * 4);
        for (let i = 0; i < config.N * config.N; i++) {
            const x = cpu.xField[i * 2];
            const z = cpu.zField[i * 2];
            level0[i * 4] = x;
            level0[i * 4 + 1] = z;
            level0[i * 4 + 2] = x * x + z * z;
        }
        const expected: Float32Array[] = [level0];
        for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
            expected.push(reduceSlopeMip(expected[level - 1], slopeMipSize(config, level - 1)));
        }
        const comparison = slopeMipAgreement(gpuMips, expected, sourceError);
        checks.push({
            name: "ocean slope CPU/GPU mip and residual agreement",
            pass: comparison.pass,
            detail: `format=rgba16float, sourceError=${sourceError}, ${comparison.detail}`,
        });
        return checks;
    },
    live(): string {
        const texture = getSlopeTexture();
        return `ocean-slope\nmips ${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`;
    },
};

register(scenario);
