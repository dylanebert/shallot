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
    slopeFftErrorBound,
    slopeMipSize,
    slopeMipTolerance,
} from "@dylanebert/shallot-ocean";
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
        const checks: Check[] = [
            {
                name: "ocean slope product path",
                pass:
                    camera === 1 && texture !== null && texture.mipLevelCount === SLOPE_MIP_LEVELS,
                detail: `camera=${camera}, mips=${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`,
            },
        ];
        if (!texture) return checks;

        const config = SLOPE_CASCADE_CONFIGS[0];
        const cpu = runSlopeCpuPipeline(generateH0(config, 0), config, 0.73);
        const sourceError = Math.max(
            slopeFftErrorBound(cpu.x, config.N),
            slopeFftErrorBound(cpu.z, config.N),
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
        const actual = await readSlopeMips(config, 0.73);
        let worstRatio = 0;
        let worstLabel = "none";
        let worstError = 0;
        let worstTolerance = 0;
        let worstExpected = 0;
        let worstActual = 0;
        let vacuity = "";
        let comparisonPass = actual.length === expected.length;
        for (let level = 0; level < expected.length; level++) {
            for (let channel = 0; channel < 4; channel++) {
                const offset = channel;
                let error = 0;
                let zeroError = 0;
                let channelExpected = 0;
                let channelActual = 0;
                let channelTolerance = 0;
                for (let i = offset; i < expected[level].length; i += 4) {
                    const tolerance = slopeMipTolerance(
                        expected[level][i],
                        level,
                        channel,
                        sourceError,
                    );
                    const ratio = Math.abs(actual[level][i] - expected[level][i]) / tolerance;
                    if (ratio > error) {
                        error = ratio;
                        channelExpected = expected[level][i];
                        channelActual = actual[level][i];
                        channelTolerance = tolerance;
                    }
                    zeroError = Math.max(zeroError, Math.abs(expected[level][i]) / tolerance);
                }
                if (error > worstRatio) {
                    worstRatio = error;
                    worstLabel = `level=${level},channel=${channel}`;
                    worstExpected = channelExpected;
                    worstActual = channelActual;
                    worstError = Math.abs(worstActual - worstExpected);
                    worstTolerance = channelTolerance;
                }
                const isVacuous = zeroError < 1;
                vacuity += `zeroPayload level=${level},channel=${channel}:${isVacuous ? "vacuous" : "witnessed"} `;
                comparisonPass &&= isVacuous ? zeroError < 1 : zeroError >= 1;
                comparisonPass &&= error < 1;
            }
        }
        checks.push({
            name: "ocean slope CPU/GPU mip and residual agreement",
            pass: comparisonPass,
            detail: `format=rgba16float, sourceError=${sourceError}, worst=${worstLabel}:${worstRatio} (${worstActual}-${worstExpected}=${worstError}, tol=${worstTolerance}), ${vacuity.trim()}`,
        });
        return checks;
    },
    live(): string {
        const texture = getSlopeTexture();
        return `ocean-slope\nmips ${texture?.mipLevelCount ?? 0}/${SLOPE_MIP_LEVELS}`;
    },
};

register(scenario);
