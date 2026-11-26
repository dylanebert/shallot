import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { Time, type World } from "../../engine";
import { drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "../../extras/lines";
import { internText, registerFont, Text, TextPlugin } from "../../extras/text";
import { isolationFont } from "../../extras/text/font.fixture";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { Meshes, MeshInstance } from "../mesh";
import { GlobalTransform, Transform } from "../transform";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
    TransparentPhases,
    VisualCorrection,
} from ".";
import {
    GlobalTransformHistory,
    RenderTransformHistory,
    readRenderGlobalTransform,
} from "./global-transform";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
let mover = -1;
const subject = gpuScene(
    import.meta.path,
    {
        defaults: false,
        plugins: [
            TextPlugin,
            LinesPlugin,
            MeshRenderPlugin,
            StandardRenderingPlugin,
            {
                name: "Mover",
                recovery: "stateless",
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            if (mover < 0) return;
                            world
                                .storage(Transform)
                                .translation.set(mover, world.time.fixedTick * 10, 0, 0);
                        },
                    },
                ],
            },
        ],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
    (world) => {
        const camera = createCamera(world);
        const farMaterial = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [1, 0.2, 0.1, 0.25], unlit: true }), {
                alphaMode: AlphaMode.Blend,
            });
        const nearMaterial = world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [0.1, 0.2, 1, 0.5], unlit: true }), {
                alphaMode: AlphaMode.Blend,
            });
        return { camera, farMaterial, nearMaterial };
    },
);

function createCamera(world: World) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 5, 0] });
    world.add(eid, Camera);
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    world.storage(Camera).mode.set(eid, CameraMode.Orthographic);
    world.storage(Camera).size.set(eid, 3);
    world.storage(Camera).clearColor.set(eid, 0);
    world.storage(Camera).antialias.set(eid, 0);
    attachTexture(world, eid, { width: 64, height: 64 });
    return eid;
}

async function readFrame(world: World, camera: number) {
    world.step(1 / 60);
    world.step(1 / 60);
    return (await captureTexture(world, camera)).rgba;
}

function differsAt(first: Uint8ClampedArray, second: Uint8ClampedArray, offset: number): boolean {
    for (let channel = 0; channel < 4; channel++)
        if (first[offset + channel] !== second[offset + channel]) return true;
    return false;
}

test("standard and text sort together; depth-tested retained lines follow sorted transparency", async () => {
    const {
        world,
        scene: { camera, farMaterial, nearMaterial },
    } = subject();
    const blank = await readFrame(world, camera);

    const label = world.create();
    world.add(label, Transform, { translation: [-2, 0.25, -1, 0] });
    const content = internText(world, "i");
    world.add(label, Text, {
        content,
        font,
        fontSize: 0.8,
        color: 0xff3030,
    });
    const textOnly = await readFrame(world, camera);

    const offscreenLine = world.create();
    world.add(offscreenLine, Transform, { translation: [4, 0, -1.2, 0] });
    const offscreenAsset = new GizmoAsset();
    drawLine(offscreenAsset, [0, 0, 0], [0.4, 0, 0], 0xffffff);
    world.add(offscreenLine, Gizmo, { handle: registerGizmo(world, offscreenAsset), width: 8 });

    const line = world.create();
    world.add(line, Transform, { translation: [-2, -0.6, -1, 0] });
    const asset = new GizmoAsset();
    drawLine(asset, [0, 0, 0], [1, 0, 0], 0xffffff);
    world.add(line, Gizmo, { handle: registerGizmo(world, asset), width: 12 });
    const textAndLines = await readFrame(world, camera);

    const farPane = world.create();
    world.add(farPane, Transform, {
        translation: [2, 0, -1.5, 0],
        scale: [0.45, 2, 0.02, 0],
    });
    world.add(farPane, MeshInstance);
    world.add(farPane, MeshMaterial, farMaterial);

    const slope = 0.35;
    const vertices = new Float32Array([
        -2.5,
        -2.5,
        -2.5 * slope,
        0,
        0,
        0,
        1,
        0,
        2.5,
        -2.5,
        2.5 * slope,
        1,
        0,
        0,
        1,
        0,
        2.5,
        2.5,
        2.5 * slope,
        1,
        0,
        0,
        1,
        1,
        -2.5,
        2.5,
        -2.5 * slope,
        0,
        0,
        0,
        1,
        1,
    ]);
    const mesh = world.resource(Meshes).add({
        name: "transparent-depth-probe-slope",
        vertices,
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const nearPane = world.create();
    world.add(nearPane, Transform, { translation: [0, 0, -0.7, 0] });
    world.add(nearPane, MeshInstance, { mesh });
    world.add(nearPane, MeshMaterial, nearMaterial);
    const composed = await readFrame(world, camera);

    let textCoverage = 0;
    let textOverdraw = 0;
    let lineCoverage = 0;
    let lineOverdraw = 0;
    for (let offset = 0; offset < blank.length; offset += 4) {
        if (differsAt(blank, textOnly, offset)) {
            textCoverage++;
            if (differsAt(textOnly, composed, offset)) textOverdraw++;
        }
        if (differsAt(textOnly, textAndLines, offset)) {
            lineCoverage++;
            if (differsAt(textAndLines, composed, offset)) lineOverdraw++;
        }
    }
    expect(textCoverage).toBeGreaterThan(0);
    expect(lineCoverage).toBeGreaterThan(0);
    expect(textOverdraw).toBeGreaterThan(0);
    expect(lineOverdraw).toBeGreaterThan(0);

    const phase = world.resource(TransparentPhases).get(camera)!;
    expect(phase.length).toBe(5);
    expect(phase.item(0).stableId).toBe(farPane);
    expect(phase.item(1).stableId).toBe(label);
    expect(phase.item(2).stableId).toBe(nearPane);
    expect(phase.item(0).renderer).toBe(phase.item(2).renderer);
    expect(phase.item(1).renderer).not.toBe(phase.item(2).renderer);
    expect(phase.item(3).batch).toBe(world.storage(Gizmo).handle.get(offscreenLine));
    expect(phase.item(4).batch).toBe(world.storage(Gizmo).handle.get(line));
    expect(phase.item(3).renderer).toBe(phase.item(4).renderer);
});

test("transparent mesh culling and mesh/text sorting use corrected rendered placement", async () => {
    const {
        world,
        scene: { camera, nearMaterial },
    } = subject();
    for (const eid of world.query([Gizmo])) world.remove(eid, Gizmo);
    for (const eid of world.query([Text])) world.remove(eid, Text);
    for (const eid of world.query([MeshInstance])) world.remove(eid, MeshInstance);
    const corrected = world.create();
    world.add(corrected, Transform, { translation: [100, 0, 0] });
    world.add(corrected, MeshInstance);
    world.add(corrected, MeshMaterial, nearMaterial);
    world.add(corrected, VisualCorrection, { translation: [-100, 0, -3] });
    const middle = world.create();
    world.add(middle, Transform, { translation: [0, 0, -1] });
    world.add(middle, MeshInstance);
    world.add(middle, MeshMaterial, nearMaterial);
    const culled = world.create();
    world.add(culled, Transform);
    world.add(culled, MeshInstance);
    world.add(culled, MeshMaterial, nearMaterial);
    world.add(culled, VisualCorrection, { translation: [100, 0, 0] });
    const label = world.create();
    world.add(label, Transform, { translation: [0, 0, -10] });
    world.add(label, Text, { content: internText(world, "i"), font, fontSize: 0.8 });
    world.add(label, VisualCorrection, { translation: [0, 0, 10] });
    await readFrame(world, camera);
    const phase = world.resource(TransparentPhases).get(camera)!;
    expect(phase.length).toBe(3);
    expect(phase.item(0).stableId).toBe(corrected);
    expect(phase.item(1).stableId).toBe(middle);
    expect(phase.item(2).stableId).toBe(label);
    world.remove(corrected, VisualCorrection);
    world.step(0);
    expect(phase.length).toBe(2);
    expect(phase.item(0).stableId).toBe(middle);
    expect(phase.item(1).stableId).toBe(label);
});

test("transparent history retires on opacity or unreadability without ending explicit enrollment", () => {
    const {
        world,
        scene: { nearMaterial },
    } = subject();
    const opaque = world.resource(Materials).add(StandardMaterial({ unlit: true }));
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, nearMaterial);
    world.step(1 / 60);
    const history = world.resource(GlobalTransformHistory).cpuHistory;
    expect(history.has(eid)).toBe(true);
    world.add(eid, MeshMaterial, opaque);
    world.step(0);
    expect(history.has(eid)).toBe(false);
    const opaqueSnapshot = world.snapshot();
    world.add(eid, MeshMaterial, nearMaterial);
    world.step(1 / 60);
    expect(history.has(eid)).toBe(true);
    world.restore(opaqueSnapshot);
    expect(history.has(eid)).toBe(false);
    world.add(eid, MeshMaterial, nearMaterial);
    world.step(1 / 60);
    expect(history.has(eid)).toBe(true);
    world.remove(eid, GlobalTransform);
    expect(history.has(eid)).toBe(false);
    world.add(eid, GlobalTransform);
    world.step(1 / 60);
    expect(history.has(eid)).toBe(true);
    world.remove(eid, MeshInstance);
    expect(history.has(eid)).toBe(false);
    world.add(eid, MeshInstance);
    world.add(eid, RenderTransformHistory);
    world.step(1 / 60);
    world.add(eid, MeshMaterial, opaque);
    world.step(0);
    expect(history.has(eid)).toBe(true);
    world.remove(eid, RenderTransformHistory);
    expect(history.has(eid)).toBe(false);
});

test("an Add mesh interpolates on its first read after fixed capture and releases its history when opaque", () => {
    const { world } = subject();
    const add = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 1, 1, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Add,
        });
    const opaque = world.resource(Materials).add(StandardMaterial({ unlit: true }));
    world.step(Time.FIXED_DT * ((1.5 - world.time.fixedAlpha) % 1));
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial, add);
    mover = eid;
    try {
        // One fixed tick moves it from the origin; its frame's reads follow that capture.
        world.step(Time.FIXED_DT);
        const alpha = world.time.fixedAlpha;
        expect(alpha).toBeGreaterThan(0.1);
        expect(alpha).toBeLessThan(0.9);
        const current = world.storage(GlobalTransform).translation.x.get(eid);
        expect(current).toBeGreaterThan(0);
        const out = new Float32Array(12);
        readRenderGlobalTransform(world, eid, out);
        expect(out[0]).toBeCloseTo(alpha * current, 4);
        const history = world.resource(GlobalTransformHistory).cpuHistory;
        expect(history.has(eid)).toBe(true);
        world.add(eid, MeshMaterial, opaque);
        world.step(0);
        expect(history.has(eid)).toBe(false);
    } finally {
        mover = -1;
        world.destroy(eid);
    }
});

test("gizmos of one asset draw once after a transparent mesh regardless of instance depth", async () => {
    const {
        world,
        scene: { camera, nearMaterial },
    } = subject();
    for (const eid of world.query([Gizmo])) world.remove(eid, Gizmo);
    for (const eid of world.query([Text])) world.remove(eid, Text);
    for (const eid of world.query([MeshInstance])) world.remove(eid, MeshInstance);
    const asset = new GizmoAsset();
    drawLine(asset, [-0.8, 0, 2], [0.8, 0, 2], 0xffffff);
    drawLine(asset, [0, -0.8, 2], [0, 0.8, 2], 0xff3030);
    const handle = registerGizmo(world, asset);
    const far = world.create();
    world.add(far, Transform, { translation: [0, 0, -3] });
    world.add(far, Gizmo, { handle, width: 8, opacity: 0.5 });
    const near = world.create();
    world.add(near, Transform, { translation: [0, 0, -1] });
    world.add(near, Gizmo, { handle, width: 4, opacity: 0.75 });
    const combined = await readFrame(world, camera);
    const phase = world.resource(TransparentPhases).get(camera)!;
    expect(phase.length).toBe(1);
    expect(phase.item(0).batch).toBe(handle);
    expect(phase.metrics.draws).toBe(1);
    const pane = world.create();
    world.add(pane, Transform, { scale: [0.7, 0.7, 0.02] });
    world.add(pane, MeshInstance);
    world.add(pane, MeshMaterial, nearMaterial);
    const interleaved = await readFrame(world, camera);
    expect(phase.length).toBe(2);
    expect(phase.item(0).stableId).toBe(pane);
    expect(phase.item(1).batch).toBe(handle);
    expect(phase.metrics.draws).toBe(2);
    expect(interleaved).not.toEqual(combined);
    world.remove(pane, MeshInstance);
    // Appending shared geometry invalidates its upload without changing its handle.
    drawLine(asset, [-0.8, 0.5, 2], [0.8, 0.5, 2], 0x30ff30);
    const changed = await readFrame(world, camera);
    expect(phase.metrics.draws).toBe(1);
    expect(changed).not.toEqual(combined);
});
