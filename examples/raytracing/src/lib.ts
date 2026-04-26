import {
    RasterPlugin,
    RenderPlugin,
    Skylab,
    Part,
    surfaceRegistry,
    property,
    traits,
    type Plugin,
    type Config,
    type System,
    type State,
} from "@dylanebert/shallot";
import {
    BloomPlugin,
    OrbitPlugin,
    RaytracingPlugin,
    SkylabPlugin,
} from "@dylanebert/shallot/extras";
import { WaterPlugin } from "./water";
import { registerContent } from "./content";
import { raytracingUI } from "./ui";

export const DayNight = {
    hour: [] as number[],
    speed: [] as number[],
};
traits(DayNight, { requires: [Skylab], defaults: () => ({ hour: 12, speed: 0 }) });

const DayNightSystem: System = {
    group: "simulation",
    update(state: State) {
        const dt = state.time.deltaTime;
        for (const eid of state.query([DayNight])) {
            let hour = DayNight.hour[eid] + dt * DayNight.speed[eid];
            if (hour >= 24) hour -= 24;
            if (hour < 0) hour += 24;
            DayNight.hour[eid] = hour;
            Skylab.elevation[eid] = 55 * Math.sin(((hour - 6) / 12) * Math.PI);
            Skylab.azimuth[eid] = 180 + (hour / 24) * 360;
        }
    },
};

export const DayNightPlugin: Plugin = {
    name: "DayNight",
    systems: [DayNightSystem],
    components: { DayNight },
};

const WAVE_HEIGHTS = [0.5, 0.1];

const WaveBlobSystem: System = {
    group: "simulation",
    update(state: State) {
        const waveHeight = property("waveHeight");
        if (!waveHeight) return;
        const surfaceId = surfaceRegistry.getByName("wave-blob");
        if (surfaceId === undefined) return;
        let idx = 0;
        for (const eid of state.query([Part])) {
            if (Part.surface[eid] !== surfaceId) continue;
            waveHeight[eid] = WAVE_HEIGHTS[idx] ?? WAVE_HEIGHTS[WAVE_HEIGHTS.length - 1];
            idx++;
        }
    },
};

export const RaytracingExamplePlugin: Plugin = {
    name: "RaytracingExample",
    dependencies: [RenderPlugin],
    systems: [WaveBlobSystem],
    initialize() {
        registerContent();
    },
};

export const config: Config = {
    plugins: [
        RaytracingPlugin,
        OrbitPlugin,
        DayNightPlugin,
        SkylabPlugin,
        BloomPlugin,
        WaterPlugin,
        RaytracingExamplePlugin,
    ],
    exclude: [RasterPlugin],
    scene: "/scenes/raytracing.scene",
    ui: raytracingUI,
};
