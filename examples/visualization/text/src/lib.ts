import {
    Transform,
    traits,
    minimalLight,
    type State,
    type System,
    type Plugin,
    type Config,
} from "@dylanebert/shallot";
import { Text, TextPlugin, OrbitPlugin, font } from "@dylanebert/shallot/extras";

export const WaveText = {
    amplitude: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
    baseY: [] as number[],
};
traits(WaveText, {
    defaults: () => ({
        amplitude: 0.3,
        frequency: 1,
        phase: 0,
        baseY: 0,
    }),
});

export const PulseSize = {
    minSize: [] as number[],
    maxSize: [] as number[],
    frequency: [] as number[],
    phase: [] as number[],
};
traits(PulseSize, {
    defaults: () => ({
        minSize: 0.3,
        maxSize: 0.6,
        frequency: 0.5,
        phase: 0,
    }),
});

export const ColorCycle = {
    hueSpeed: [] as number[],
    saturation: [] as number[],
    lightness: [] as number[],
};
traits(ColorCycle, {
    defaults: () => ({
        hueSpeed: 30,
        saturation: 0.7,
        lightness: 0.6,
    }),
});

export const RotateText = {
    speed: [] as number[],
    radius: [] as number[],
    axisY: [] as number[],
    phase: [] as number[],
};
traits(RotateText, {
    defaults: () => ({
        speed: 30,
        radius: 2,
        axisY: 0,
        phase: 0,
    }),
});

function hslToRgb(h: number, s: number, l: number): number {
    h = h / 360;
    let r: number, g: number, b: number;

    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p: number, q: number, t: number): number => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }

    return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
}

const WaveTextSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([WaveText, Transform])) {
            const amp = WaveText.amplitude[eid];
            const freq = WaveText.frequency[eid];
            const phase = WaveText.phase[eid];
            const baseY = WaveText.baseY[eid];
            const wave = Math.sin(t * freq * Math.PI * 2 + phase);
            Transform.posY[eid] = baseY + wave * amp;
        }
    },
};

const PulseSizeSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([PulseSize, Text])) {
            const min = PulseSize.minSize[eid];
            const max = PulseSize.maxSize[eid];
            const freq = PulseSize.frequency[eid];
            const phase = PulseSize.phase[eid];
            const wave = (Math.sin(t * freq * Math.PI * 2 + phase) + 1) * 0.5;
            Text.fontSize[eid] = min + wave * (max - min);
        }
    },
};

const ColorCycleSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([ColorCycle, Text])) {
            const speed = ColorCycle.hueSpeed[eid];
            const sat = ColorCycle.saturation[eid];
            const light = ColorCycle.lightness[eid];
            const hue = (t * speed) % 360;
            Text.color[eid] = hslToRgb(hue, sat, light);
        }
    },
};

const RotateTextSystem: System = {
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([RotateText, Transform])) {
            const speed = RotateText.speed[eid];
            const radius = RotateText.radius[eid];
            const axisY = RotateText.axisY[eid];
            const phase = RotateText.phase[eid];
            const angle = ((t * speed + phase) * Math.PI) / 180;

            Transform.posX[eid] = Math.cos(angle) * radius;
            Transform.posY[eid] = axisY;
            Transform.posZ[eid] = Math.sin(angle) * radius;
        }
    },
};

export const TextDemoPlugin: Plugin = {
    name: "TextDemo",
    systems: [WaveTextSystem, PulseSizeSystem, ColorCycleSystem, RotateTextSystem],
    components: { WaveText, PulseSize, ColorCycle, RotateText },
    dependencies: [TextPlugin],
    initialize() {
        font("/Inter-Regular.ttf", "inter");
        font(
            "https://fonts.gstatic.com/s/pressstart2p/v16/e3t4euO8T-267oIAQAu6jDQyK0nS.ttf",
            "pixel",
        );
    },
};

export const config: Config = {
    plugins: [TextPlugin, OrbitPlugin, TextDemoPlugin],
    scene: "/text/scenes/text.scene",
    loading: minimalLight(),
};
