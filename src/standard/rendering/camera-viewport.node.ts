import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachCanvas,
    attachTexture,
    Camera,
    CameraViewport,
    DirectionalLight,
    Resolution,
    Views,
} from "../../core/rendering";
import { renderFrameKey } from "../../core/rendering/frame-state";
import { Transform } from "../../core/transform";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { Clusters } from "./cluster";
import { StandardRenderer, StandardRenderingPlugin } from "./index";
import { cascadeCovers, updateCascades } from "./shadows";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, pixelRatio: 0.8, plugins: [StandardRenderingPlugin] },
]);
class TestCanvas {}
for (const binding of ["attached", "auto-bound", "late-mounted"])
    for (const pinned of [true, false])
        for (const consumer of ["clusters", "shadows"]) {
            if (binding !== "attached" && (pinned || consumer !== "shadows")) continue;
            test(`${consumer} use the ${binding} camera rectangle aspect with ${pinned ? "Resolution" : "backing-size rounding"} on its first visible frame`, () => {
                const { world } = subjects()[0];
                const camera = world.create();
                const reference = world.create();
                const sun = world.create();
                const previousDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
                const previous = Object.getOwnPropertyDescriptor(globalThis, "HTMLCanvasElement");
                Object.defineProperty(globalThis, "HTMLCanvasElement", {
                    configurable: true,
                    value: TestCanvas,
                });
                let context: CanvasContext;
                const canvas = Object.assign(Object.create(TestCanvas.prototype), {
                    width: 64,
                    height: 48,
                    clientWidth: 64,
                    clientHeight: 48,
                    style: { imageRendering: "auto" },
                    getContext: () => context,
                    getBoundingClientRect: () => ({ width: 64, height: 48 }),
                }) as HTMLCanvasElement;
                context = new CanvasContext(canvas, 64, 48);
                let mounted = binding !== "late-mounted";
                Object.defineProperty(globalThis, "document", {
                    configurable: true,
                    value: {
                        querySelector: () => (mounted ? canvas : null),
                        querySelectorAll: () => (mounted ? [canvas] : []),
                    },
                });
                try {
                    for (const eid of [camera, reference]) {
                        world.add(eid, Transform, { translation: [0, 2, 10, 0] });
                        world.add(eid, Camera, { antialias: 0 });
                    }
                    world.add(camera, StandardRenderer);
                    world.add(camera, CameraViewport, { x: 40, y: 4, width: 16, height: 32 });
                    if (pinned) world.add(camera, Resolution, { width: 32, height: 32 });
                    if (binding === "attached") attachCanvas(world, camera, canvas);
                    attachTexture(world, reference, { width: 16, height: 32 });
                    world.add(sun, Transform);
                    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
                    if (!mounted) {
                        world.step(0);
                        expect(world.resource(Views).has(camera)).toBe(false);
                        mounted = true;
                    }
                    world.step(0);
                    const view = world.resource(Views).get(camera)!;
                    const matrix = world.resource(renderFrameKey).clipFromViews[view.slot]!;
                    const cameraAspect = matrix[5]! / matrix[0]!;
                    expect(cameraAspect).toBeCloseTo(0.5, 6);
                    expect(view.width / view.height).not.toBeCloseTo(cameraAspect, 3);
                    if (consumer === "clusters") {
                        const staging = world.resource(Clusters).staging;
                        expect(staging[view.slot * 8]! / staging[view.slot * 8 + 1]!).toBeCloseTo(
                            cameraAspect,
                            6,
                        );
                    } else {
                        const actual = [...cascadeCovers(world)];
                        updateCascades(world, reference);
                        expect(actual).toEqual([...cascadeCovers(world)]);
                    }
                } finally {
                    for (const eid of [camera, reference, sun]) world.destroy(eid);
                    mounted = false;
                    world.step(0);
                    if (previousDocument)
                        Object.defineProperty(globalThis, "document", previousDocument);
                    else Reflect.deleteProperty(globalThis, "document");
                    if (previous) Object.defineProperty(globalThis, "HTMLCanvasElement", previous);
                    else Reflect.deleteProperty(globalThis, "HTMLCanvasElement");
                }
            });
        }
