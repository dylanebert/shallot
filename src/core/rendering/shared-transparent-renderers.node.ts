import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import type { World } from "../../engine";
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
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
    TransparentPhases,
} from ".";

setDefaultTimeout(TIMEOUT.node);
let font = 0;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [TextPlugin, LinesPlugin, MeshRenderPlugin, StandardRenderingPlugin],
        setup(world) {
            font = registerFont(
                world,
                `data:font/ttf;base64,${Buffer.from(isolationFont()).toString("base64")}`,
            );
        },
    },
]);

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

test("standard, text and line renderers alternate in one phase; text and lines do not write transparent depth", async () => {
    const { world } = subjects()[0];
    const camera = createCamera(world);
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

    const farMaterial = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [1, 0.2, 0.1, 0.25], unlit: true }), {
            alphaMode: AlphaMode.Blend,
        });
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
    const nearMaterial = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.1, 0.2, 1, 0.5], unlit: true }), {
            alphaMode: AlphaMode.Blend,
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
    expect(phase.item(1).stableId).toBe(offscreenLine);
    expect(phase.item(phase.length - 1).stableId).toBe(nearPane);
    const middle = [phase.item(2), phase.item(3)];
    expect(middle.map((item) => item.stableId).sort((a, b) => a - b)).toEqual(
        [label, line].sort((a, b) => a - b),
    );
    expect(phase.item(0).renderer).toBe(phase.item(phase.length - 1).renderer);
    expect(middle[0]!.renderer).not.toBe(middle[1]!.renderer);
    const labelItem = middle.find((item) => item.stableId === label)!;
    const lineItem = middle.find((item) => item.stableId === line)!;
    expect(labelItem.draw).not.toBe(label);
    expect(lineItem.draw).toBe(1);
    expect(lineItem.stablePart).toBe(0);
    expect(lineItem.draw).not.toBe(lineItem.stablePart);
});
