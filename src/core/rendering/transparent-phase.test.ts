import { expect, test } from "bun:test";
import { World } from "../../engine";
import { component } from "../../engine/ecs/component";
import { GlobalTransform } from "../transform";
import { Camera } from "./camera";
import { TransparentPhase } from "./phases";
import type { View } from "./view";

function view(width = 100, height = 100): View {
    return { width, height } as View;
}

function worldWithCamera(
    position: [number, number, number],
    rotation: [number, number, number, number] = [0, 0, 0, 1],
) {
    const world = new World();
    world.registry.register(component("Camera", { ...Camera }));
    world.registry.register(component("GlobalTransform", { ...GlobalTransform }));
    const camera = world.create();
    world.add(camera, Camera, {
        mode: 0,
        fov: 60,
        near: 0.1,
        far: 100,
        size: 5,
        clearColor: 0,
        antialias: 0,
    });
    world.add(camera, GlobalTransform, {
        translation: [...position, 0],
        rotation,
        scale: [1, 1, 1, 0],
        linearVelocity: [0, 0, 0, 0],
    });
    return { world, camera };
}

test("transparent items sort by view-space Z, not radial distance, and break ties by stable identity", () => {
    const { world, camera } = worldWithCamera([0, 0, 0]);
    try {
        const phase = new TransparentPhase();
        phase.begin(world, camera, view());
        phase.useRenderer(0);
        phase.add(11, 1, 0, 0, -10, 11); // contributed first, but the larger stable identity
        phase.add(10, 1, 20, 0, -10, 10); // radial distance 22.4, same view Z, smaller identity
        phase.useRenderer(1);
        phase.add(20, 2, 0, 0, -20, 20); // radial distance 20, view Z -20
        phase.sort();

        expect(Array.from({ length: phase.length }, (_, i) => phase.item(i).draw)).toEqual([
            20, 10, 11,
        ]);
        expect(phase.item(0).sortKey).toBe(-20);
        expect(phase.item(1).stableId).toBe(10);
        expect(phase.item(2).stableId).toBe(11);
        expect(phase.metrics.sortComparisons).toBeGreaterThan(0);
    } finally {
        world.dispose();
    }
});

test("two views own opposite transparent orders for the same pair of panes", () => {
    const { world, camera: first } = worldWithCamera([0, 0, 5]);
    const second = world.create();
    world.add(second, Camera, {
        mode: 0,
        fov: 60,
        near: 0.1,
        far: 100,
        size: 5,
        clearColor: 0,
        antialias: 0,
    });
    world.add(second, GlobalTransform, {
        translation: [0, 0, -5, 0],
        rotation: [0, 1, 0, 0],
        scale: [1, 1, 1, 0],
        linearVelocity: [0, 0, 0, 0],
    });

    try {
        const a = new TransparentPhase();
        const b = new TransparentPhase();
        a.begin(world, first, view());
        a.useRenderer(0);
        a.add(100, 1, 0, 0, 0, 100);
        a.useRenderer(1);
        a.add(200, 2, 0, 0, 10, 200);
        a.sort();

        b.begin(world, second, view());
        b.useRenderer(0);
        b.add(100, 1, 0, 0, 0, 100);
        b.useRenderer(1);
        b.add(200, 2, 0, 0, 10, 200);
        b.sort();
        expect([a.item(0).draw, a.item(1).draw]).toEqual([100, 200]);
        expect([b.item(0).draw, b.item(1).draw]).toEqual([200, 100]);
    } finally {
        world.dispose();
    }
});

test("a phase reused across frames as a prompt's parts come and go queues each item exactly once", () => {
    const { world, camera } = worldWithCamera([0, 0, 0]);
    try {
        const phase = new TransparentPhase();
        // Three far name labels and a near prompt's key and action text from one renderer, its key cap
        // and ring from another; the ring shows only while the held thing is focused.
        const frame = (ring: boolean) => {
            phase.begin(world, camera, view());
            phase.useRenderer(0);
            phase.add(0, 0, -1.8, 0, -3, 10);
            phase.add(1, 0, 0, 0, -3.1, 11);
            phase.add(2, 0, 1.8, 0, -3.2, 12);
            phase.add(3, 0, 0, 0, -1, 13);
            phase.add(4, 0, 0.1, 0, -1.01, 14);
            phase.useRenderer(1);
            phase.add(20, 1, 0, 0, -1.02, 20);
            if (ring) phase.add(23, 1, 0, 0, -1.03, 23);
            phase.sort();
            return Array.from(
                { length: phase.length },
                (_, i) => `${phase.item(i).renderer}:${phase.item(i).draw}`,
            );
        };
        const held = ["0:2", "0:1", "0:0", "1:23", "1:20", "0:4", "0:3"];
        expect(frame(true)).toEqual(held);
        expect(frame(false)).toEqual(held.filter((item) => item !== "1:23"));
        expect(frame(true)).toEqual(held);
    } finally {
        world.dispose();
    }
});
