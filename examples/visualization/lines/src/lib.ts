import {
    Transform,
    rotate,
    traits,
    minimalLight,
    type State,
    type System,
    type Plugin,
    type Config,
} from "@dylanebert/shallot";
import { Line, Arrow, LinesPlugin, ArrowsPlugin, OrbitPlugin } from "@dylanebert/shallot/extras";

export const WaveOffset = {
    amplitudeX: [] as number[],
    amplitudeY: [] as number[],
    amplitudeZ: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
    baseOffsetX: [] as number[],
    baseOffsetY: [] as number[],
    baseOffsetZ: [] as number[],
};
traits(WaveOffset, {
    defaults: () => ({
        amplitudeX: 0,
        amplitudeY: 0,
        amplitudeZ: 0,
        frequency: 1,
        phase: 0,
        baseOffsetX: 1,
        baseOffsetY: 0,
        baseOffsetZ: 0,
    }),
});

export const PulseThickness = {
    minThickness: [] as number[],
    maxThickness: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
};
traits(PulseThickness, {
    defaults: () => ({ minThickness: 1, maxThickness: 4, frequency: 1, phase: 0 }),
});

export const PulseArrow = {
    minSize: [] as number[],
    maxSize: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
};
traits(PulseArrow, {
    defaults: () => ({ minSize: 6, maxSize: 18, frequency: 0.8, phase: 0 }),
});

export const RotateOrigin = {
    radius: [] as number[],
    speed: [] as number[],
    axisY: [] as number[],
    phase: [] as number[],
};
traits(RotateOrigin, {
    defaults: () => ({ radius: 1, speed: 45, axisY: 0, phase: 0 }),
});

export const FadeOpacity = {
    minOpacity: [] as number[],
    maxOpacity: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
};
traits(FadeOpacity, {
    defaults: () => ({ minOpacity: 0.3, maxOpacity: 1, frequency: 0.5, phase: 0 }),
});

export const SpinY = { speed: [] as number[] };
traits(SpinY, { defaults: () => ({ speed: 45 }) });

const WaveOffsetSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([WaveOffset, Line])) {
            const freq = WaveOffset.frequency[eid];
            const phase = WaveOffset.phase[eid];
            const wave = Math.sin(t * freq * Math.PI * 2 + phase);

            Line.offsetX[eid] = WaveOffset.baseOffsetX[eid] + wave * WaveOffset.amplitudeX[eid];
            Line.offsetY[eid] = WaveOffset.baseOffsetY[eid] + wave * WaveOffset.amplitudeY[eid];
            Line.offsetZ[eid] = WaveOffset.baseOffsetZ[eid] + wave * WaveOffset.amplitudeZ[eid];
        }
    },
};

const PulseThicknessSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([PulseThickness, Line])) {
            const min = PulseThickness.minThickness[eid];
            const max = PulseThickness.maxThickness[eid];
            const freq = PulseThickness.frequency[eid];
            const phase = PulseThickness.phase[eid];
            const wave = (Math.sin(t * freq * Math.PI * 2 + phase) + 1) * 0.5;
            Line.thickness[eid] = min + wave * (max - min);
        }
    },
};

const PulseArrowSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([PulseArrow, Arrow])) {
            const min = PulseArrow.minSize[eid];
            const max = PulseArrow.maxSize[eid];
            const freq = PulseArrow.frequency[eid];
            const phase = PulseArrow.phase[eid];
            const wave = (Math.sin(t * freq * Math.PI * 2 + phase) + 1) * 0.5;
            Arrow.size[eid] = min + wave * (max - min);
        }
    },
};

const RotateOriginSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([RotateOrigin, Transform])) {
            const radius = RotateOrigin.radius[eid];
            const speed = RotateOrigin.speed[eid];
            const axisY = RotateOrigin.axisY[eid];
            const phase = RotateOrigin.phase[eid];
            const angle = ((t * speed + phase) * Math.PI) / 180;

            Transform.posX[eid] = Math.cos(angle) * radius;
            Transform.posY[eid] = axisY;
            Transform.posZ[eid] = Math.sin(angle) * radius;
        }
    },
};

const FadeOpacitySystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([FadeOpacity, Line])) {
            const min = FadeOpacity.minOpacity[eid];
            const max = FadeOpacity.maxOpacity[eid];
            const freq = FadeOpacity.frequency[eid];
            const phase = FadeOpacity.phase[eid];
            const wave = (Math.sin(t * freq * Math.PI * 2 + phase) + 1) * 0.5;
            Line.opacity[eid] = min + wave * (max - min);
        }
    },
};

const SpinYSystem: System = {
    group: "simulation",
    update(state: State) {
        const dt = state.time.deltaTime;
        for (const eid of state.query([SpinY, Transform])) {
            const q = rotate(
                Transform.quatX[eid],
                Transform.quatY[eid],
                Transform.quatZ[eid],
                Transform.quatW[eid],
                0,
                SpinY.speed[eid] * dt,
                0,
            );
            Transform.quatX[eid] = q.x;
            Transform.quatY[eid] = q.y;
            Transform.quatZ[eid] = q.z;
            Transform.quatW[eid] = q.w;
        }
    },
};

export const LinesDemoPlugin: Plugin = {
    name: "LinesDemo",
    systems: [
        WaveOffsetSystem,
        PulseThicknessSystem,
        PulseArrowSystem,
        RotateOriginSystem,
        FadeOpacitySystem,
        SpinYSystem,
    ],
    components: { WaveOffset, PulseThickness, PulseArrow, RotateOrigin, FadeOpacity, SpinY },
};

export const config: Config = {
    plugins: [LinesPlugin, ArrowsPlugin, OrbitPlugin, LinesDemoPlugin],
    scene: "/lines/scenes/lines.scene",
    loading: minimalLight(),
};
