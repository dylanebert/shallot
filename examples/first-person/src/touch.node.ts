import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import {
    blurCanvas,
    cancelTouch,
    Devices,
    focusCanvas,
    InputPlugin,
    pointerLockChanged,
    resizeViewport,
    setInputEnabled,
    touchPoint,
    World,
} from "@dylanebert/shallot";
import { LocalPlayer, mountTouchButton, Player, PlayerInput, PlayerPlugin } from "./player";

test("floating stick clamps, ignores its dead zone, and coexists with right drag; release and cancel neutralize it", () => {
    const world = new World();
    for (const system of InputPlugin.systems ?? []) world.addSystem(system, InputPlugin.name);
    world.addSystem(
        PlayerPlugin.systems!.find((system) => system.name === "local-input")!,
        "Player",
    );
    const eid = world.create();
    world.add(eid, Player);
    world.add(eid, LocalPlayer, { sensitivity: 1.5, touchSensitivity: 0.5 });
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
    expect(input.yaw.get(eid)).toBeCloseTo((-30 * 0.5 * Math.PI) / 180);
    expect(input.pitch.get(eid)).toBeCloseTo((-20 * 0.5 * Math.PI) / 180);
    touchPoint(world, 1, 300, 600);
    touchPoint(world, 2, 100, 520);
    world.step(0);
    expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([1, 0]);
    expect(input.yaw.get(eid)).toBeCloseTo((200 * 0.5 * Math.PI) / 180);
    touchPoint(world, 1, 300, 600, false);
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

for (const captureThrows of [false, true]) {
    test(`button cancel releases its key and a tap jumps once when capture ${captureThrows ? "throws" : "succeeds"}`, () => {
        class Button extends EventTarget {
            textContent = "";
            style = { cssText: "" };
            pressedAtCapture = false;
            setPointerCapture() {
                this.pressedAtCapture = world.resource(Devices).keys.pressed.has("Space");
                if (captureThrows) throw new DOMException("Unknown pointer", "NotFoundError");
            }
            remove() {}
        }
        const prior = Object.getOwnPropertyDescriptor(globalThis, "document");
        const button = new Button();
        Object.defineProperty(globalThis, "document", {
            configurable: true,
            value: { createElement: () => button },
        });
        const world = new World();
        try {
            for (const system of InputPlugin.systems ?? [])
                world.addSystem(system, InputPlugin.name);
            world.addSystem(
                PlayerPlugin.systems!.find((system) => system.name === "local-input")!,
                "Player",
            );
            const eid = world.create();
            world.add(eid, Player);
            world.add(eid, LocalPlayer);
            mountTouchButton({ append() {} } as unknown as HTMLElement, world, "Jump", "Space");
            button.dispatchEvent(new Event("pointerdown"));
            expect(button.pressedAtCapture).toBe(true);
            expect(world.resource(Devices).keys.held.has("Space")).toBe(true);
            button.dispatchEvent(new Event("pointercancel"));
            expect(world.resource(Devices).keys.held.has("Space")).toBe(false);
            world.step(0);
            const input = world.storage(PlayerInput);
            const before = input.jumpPresses.get(eid);
            button.dispatchEvent(new Event("pointerdown"));
            button.dispatchEvent(new Event("pointerup"));
            expect(world.resource(Devices).keys.held.has("Space")).toBe(false);
            world.step(0);
            expect(input.jumpPresses.get(eid)).toBe(before + 1);
            world.step(0);
            expect(input.jumpPresses.get(eid)).toBe(before + 1);
        } finally {
            world.dispose();
            if (prior) Object.defineProperty(globalThis, "document", prior);
            else Reflect.deleteProperty(globalThis, "document");
        }
    });
}

for (const transition of ["blur", "disable"] as const) {
    test(`${transition} neutralizes touch movement and pending look without replaying them`, () => {
        const world = new World();
        for (const system of InputPlugin.systems ?? []) world.addSystem(system, InputPlugin.name);
        world.addSystem(
            PlayerPlugin.systems!.find((system) => system.name === "local-input")!,
            "Player",
        );
        const eid = world.create();
        world.add(eid, Player);
        world.add(eid, LocalPlayer, { sensitivity: 1.5, touchSensitivity: 0.5 });
        focusCanvas(world);
        resizeViewport(world, 0, 390, 844, 1);
        touchPoint(world, 1, 80, 600);
        touchPoint(world, 1, 80, 530);
        touchPoint(world, 2, 300, 500);
        touchPoint(world, 2, 330, 520);
        world.step(0);
        const input = world.storage(PlayerInput);
        expect(input.move.column[eid * 2 + 1]).toBe(1);
        const yaw = input.yaw.get(eid);
        const pitch = input.pitch.get(eid);
        touchPoint(world, 2, 350, 540);
        if (transition === "blur") blurCanvas(world);
        else setInputEnabled(world, false);
        world.step(0);
        expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 0]);
        expect(input.yaw.get(eid)).toBe(yaw);
        expect(input.pitch.get(eid)).toBe(pitch);
        setInputEnabled(world, true);
        focusCanvas(world);
        world.step(0);
        expect(Array.from(input.move.column.slice(eid * 2, eid * 2 + 2))).toEqual([0, 0]);
        expect(input.yaw.get(eid)).toBe(yaw);
        expect(input.pitch.get(eid)).toBe(pitch);
        world.dispose();
    });
}
