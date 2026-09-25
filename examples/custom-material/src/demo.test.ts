import { resolve } from "node:path";
import {
    build,
    Color,
    Compute,
    GlazePlugin,
    Part,
    probeTexture,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";
import { check } from "@dylanebert/shallot/harness/check";
import { Camera, computeViewProj, type View, Views } from "@dylanebert/shallot/rendering";
import { SearPlugin } from "@dylanebert/shallot/standard/rendering";
import ShallotSurfaces from "./surfaces";

const SCENE = resolve(import.meta.dir, "../public/scenes/custom-material.scene");
const WIDTH = 256;
const HEIGHT = 256;

function decodeUnsignedFloat(value: number, mantissaBits: number): number {
    const mantissaMask = (1 << mantissaBits) - 1;
    const mantissa = value & mantissaMask;
    const exponent = (value >> mantissaBits) & 0x1f;
    if (exponent === 0) return mantissa * 2 ** (1 - 15 - mantissaBits);
    if (exponent === 0x1f) return mantissa === 0 ? Number.POSITIVE_INFINITY : Number.NaN;
    return (1 + mantissa / 2 ** mantissaBits) * 2 ** (exponent - 15);
}

function project(eid: number, point: readonly [number, number, number]): [number, number] {
    const matrix = new Float32Array(16);
    computeViewProj(eid, WIDTH / HEIGHT, matrix);
    const [x, y, z] = point;
    const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    return [((clipX / clipW) * 0.5 + 0.5) * WIDTH, (0.5 - (clipY / clipW) * 0.5) * HEIGHT];
}

function colorFromWord(word: number): [number, number, number] {
    return [
        decodeUnsignedFloat(word & 0x7ff, 6),
        decodeUnsignedFloat((word >> 11) & 0x7ff, 6),
        decodeUnsignedFloat((word >> 22) & 0x3ff, 5),
    ];
}

function colorAt(bytes: Uint8Array, x: number, y: number): [number, number, number] {
    const word = new DataView(bytes.buffer, bytes.byteOffset).getUint32((y * WIDTH + x) * 4, true);
    return colorFromWord(word);
}

async function probePixel(
    device: GPUDevice,
    target: GPUTexture,
    [x, y]: readonly [number, number],
): Promise<[number, number, number]> {
    const probe = await probeTexture(device, target, {
        origin: [Math.round(x), Math.round(y), 0],
        size: [1, 1],
    });
    return colorFromWord(new DataView(probe.bytes).getUint32(0, true));
}

check(
    "custom materials reach the rendered scene target",
    {
        claim: "custom-material renders both checker tones, blended glass, and its gradient in their scene regions",
        size: "integration",
        requires: ["gpu"],
        subject: "examples/custom-material/src/surfaces.ts",
    },
    async () => {
        const peerModule = "bun-webgpu";
        const peer = (await import(peerModule)) as { setupGlobals(): Promise<void> };
        await peer.setupGlobals();

        const app = await build({
            defaults: false,
            plugins: [OrbitPlugin, GlazePlugin, SearPlugin, ShallotSurfaces],
            scene: SCENE,
        });
        let present: GPUTexture | undefined;
        let offscreen: GPUTexture | undefined;
        const device = Compute.device;
        if (!device) {
            app.dispose();
            throw new Error("custom-material has no GPU device");
        }
        const camera = app.state.only([Camera]);
        const previousCreateTexture = Object.getOwnPropertyDescriptor(device, "createTexture");
        const createTexture = device.createTexture.bind(device);
        Object.defineProperty(device, "createTexture", {
            configurable: true,
            value(descriptor: GPUTextureDescriptor) {
                const texture = createTexture(descriptor);
                if (descriptor.label?.startsWith("shallot-offscreen-")) offscreen = texture;
                return texture;
            },
        });

        try {
            present = device.createTexture({
                label: "custom-material-test-present",
                size: [WIDTH, HEIGHT],
                format: "bgra8unorm",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
            });
            Views.set(camera, {
                canvas: null,
                context: { getCurrentTexture: () => present! } as unknown as GPUCanvasContext,
                width: WIDTH,
                height: HEIGHT,
                clientWidth: WIDTH,
                clientHeight: HEIGHT,
                viewportIndex: 0,
                framebuffer: null,
                present: null,
                depth: null,
                tag: null,
                slot: 0,
                observer: null,
                stamp: app.state.stamp(camera),
            } satisfies View);

            app.state.step(Time.FIXED_DT);
            app.state.step(Time.FIXED_DT);
            if (!offscreen) throw new Error("custom-material rendered no scene target");
            const probe = await probeTexture(device, offscreen);
            const bytes = new Uint8Array(probe.bytes);
            const checkerPoints: [number, number][] = [];
            for (const y of [-0.375, -0.125, 0.125, 0.375]) {
                for (const x of [-0.375, -0.125, 0.125, 0.375]) {
                    checkerPoints.push(project(camera, [2.6 + x, 0.2 + y, 0.51]));
                }
            }
            const checker = checkerPoints.map(([x, y]) =>
                colorAt(bytes, Math.round(x), Math.round(y)),
            );
            const dimTone = checker.filter((color) => Math.max(...color) < 0.75).length;
            const brightTone = checker.filter((color) => Math.max(...color) > 1.25).length;
            if (dimTone === 0 || brightTone === 0) {
                throw new Error(
                    `checker did not render both tones (dim=${dimTone}, bright=${brightTone})`,
                );
            }

            const glass = [...app.state.query([Part, Transform])].find(
                (eid) =>
                    Math.abs(Transform.pos.x.get(eid) - 2.6) < 1e-4 &&
                    Math.abs(Transform.pos.z.get(eid) - 1.8) < 1e-4,
            );
            if (glass === undefined) throw new Error("custom-material scene has no glass sphere");
            const glassCenter = project(camera, [2.6, 0.2, 1.8]);
            const alpha = Color.rgba.w.get(glass);
            if (Math.abs(alpha - 0.5) > 1e-5) throw new Error(`glass opacity is ${alpha}, not 0.5`);
            const blended = colorAt(bytes, Math.round(glassCenter[0]), Math.round(glassCenter[1]));

            Color.rgba.w.set(glass, 1);
            app.state.step(Time.FIXED_DT);
            const opaque = await probePixel(device, offscreen, glassCenter);
            Color.rgba.w.set(glass, 0);
            app.state.step(Time.FIXED_DT);
            const clear = await probePixel(device, offscreen, glassCenter);
            const blendError = Math.max(
                ...blended.map((channel, lane) =>
                    Math.abs(channel - (opaque[lane] + clear[lane]) * 0.5),
                ),
            );
            const glassEffect = Math.max(
                ...blended.map((channel, lane) => Math.abs(channel - clear[lane])),
            );
            if (blendError > 0.12 || glassEffect < 0.05) {
                throw new Error(
                    `glass did not blend its source over the scene (error=${blendError}, effect=${glassEffect})`,
                );
            }

            const skyHigh = colorAt(bytes, 10, 16);
            const skyLow = colorAt(bytes, 10, 96);
            if (skyHigh[2] - skyLow[2] < 0.015) {
                throw new Error(
                    `gradient backdrop did not brighten toward the sky (${skyHigh} vs ${skyLow})`,
                );
            }
        } finally {
            if (previousCreateTexture)
                Object.defineProperty(device, "createTexture", previousCreateTexture);
            else Reflect.deleteProperty(device, "createTexture");
            present?.destroy();
            app.dispose();
        }
    },
);
