import { World } from "../../src/engine";
import { Camera, TransparentPhase, type View } from "../../src/core/rendering";
import { GlobalTransform } from "../../src/core/transform";
import { initializeSegmentState, push, resetCount } from "../../src/extras/lines/segments";

const ITEM_COUNT = 64;
const world = new World();
const camera = world.create();
world.add(camera, Camera, {
    mode: 0,
    fov: 60,
    near: 0.1,
    far: 1000,
    size: 5,
    clearColor: 0,
    antialias: 0,
});
world.add(camera, GlobalTransform, {
    translation: [0, 0, 5, 0],
    rotation: [0, 0, 0, 1],
    scale: [1, 1, 1, 0],
    linearVelocity: [0, 0, 0, 0],
});
initializeSegmentState(world);

const phase = new TransparentPhase();
const view = { width: 64, height: 64 } as View;
const drawStorage = new Uint32Array(ITEM_COUNT);

function stageLines(): void {
    resetCount(world);
    for (let i = 0; i < ITEM_COUNT; i++)
        push(world, i, 0, 0, i + 1, 0, 0, 2, 0xffffffff, i + 1, i);
}

function stagePhase(): void {
    phase.begin(world, camera, view);
    phase.useRenderer(0);
    for (let i = 0; i < ITEM_COUNT; i++) {
        const z = ((i * 17) % ITEM_COUNT) / 16 - 2;
        phase.add(i, i % 3, (i % 7) * 0.1, 0, z, i + 1, i % 5);
    }
    phase.sort();
    let index = 0;
    while (index < phase.length) {
        const item = phase.item(index);
        let end = index + 1;
        while (end < phase.length) {
            const next = phase.item(end);
            if (next.renderer !== item.renderer || next.batch !== item.batch) break;
            end++;
        }
        for (let i = index; i < end; i++) drawStorage[i] = phase.item(i).stableId;
        index = end;
    }
}

export let controlSink: object;
export function control() {
    controlSink = { frame: 0 };
}

export default async function create(mode = "phase") {
    const includeLines = mode === "segments" || mode === "stage";
    const includePhase = mode !== "segments";
    return {
        step() {
            if (includeLines) stageLines();
            if (includePhase) stagePhase();
        },
        dispose() {},
    };
}
