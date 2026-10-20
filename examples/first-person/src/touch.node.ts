import { expect, test } from "bun:test";
import {
    cancelTouch,
    focusCanvas,
    InputPlugin,
    pointerLockChanged,
    resizeViewport,
    touchPoint,
    World,
} from "@dylanebert/shallot";
import { LocalPlayer, Player, PlayerInput, PlayerPlugin } from "./player";

test("floating stick clamps, ignores its dead zone, and coexists with right drag; release and cancel neutralize it", () => {
    const world = new World();
    for (const system of InputPlugin.systems ?? []) world.addSystem(system, InputPlugin.name);
    world.addSystem(
        PlayerPlugin.systems!.find((system) => system.name === "local-input")!,
        "Player",
    );
    const eid = world.create();
    world.add(eid, Player);
    world.add(eid, LocalPlayer, { sensitivity: 1.5 });
    focusCanvas(world);
    resizeViewport(world, 0, 390, 844, 1);
    pointerLockChanged(world, true);
    touchPoint(world, 1, 80, 600);
    touchPoint(world, 1, 83, 600);
    world.step(0);
    const input = world.storage(PlayerInput);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 0]);
    touchPoint(world, 1, 115, 600);
    world.step(0);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))[0]).toBeCloseTo(0.5);
    touchPoint(world, 1, 80, 390);
    touchPoint(world, 2, 300, 500);
    touchPoint(world, 2, 330, 520);
    world.step(0);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 1]);
    expect(input.yaw.get(eid)).toBeCloseTo((-30 * 1.5) / 1080);
    expect(input.pitch.get(eid)).toBeCloseTo((-20 * 1.5) / 1080);
    touchPoint(world, 1, 80, 390, false);
    world.step(0);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 0]);
    const yaw = input.yaw.get(eid);
    cancelTouch(world, 2);
    touchPoint(world, 3, 50, 500);
    touchPoint(world, 3, 50, 400);
    cancelTouch(world, 3);
    world.step(0);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 0]);
    expect(input.yaw.get(eid)).toBe(yaw);
    world.dispose();
});
