import { expect, test } from "bun:test";
import { World } from "./world";

test("no system updates after its world disposes", () => {
    const world = new World();
    const log: string[] = [];
    const first = {
        name: "first",
        update: () => log.push("first"),
        dispose: () => log.push("first.dispose"),
    };
    world.addSystem(first, "Game");
    world.addSystem(
        {
            name: "second",
            after: [first],
            update: () => log.push("second"),
            dispose: () => log.push("second.dispose"),
        },
        "Game",
    );
    world.addSystem({ name: "fixed", group: "fixed", update: () => log.push("fixed") }, "Game");
    world.step(1 / 60);
    world.dispose();
    world.step(1 / 60);
    world.tick();
    expect(log).toEqual(["fixed", "first", "second", "first.dispose", "second.dispose"]);
    expect(world.hasSystem(first)).toBe(false);
});

test("a system that disposes its world ends the pass", () => {
    const world = new World();
    const log: string[] = [];
    const quit = {
        name: "quit",
        update: (w: World) => {
            log.push("quit");
            w.dispose();
        },
    };
    world.addSystem(quit, "Game");
    world.addSystem(
        {
            name: "later",
            after: [quit],
            update: () => log.push("later"),
            dispose: () => log.push("later.dispose"),
        },
        "Game",
    );
    world.step(1 / 60);
    expect(log).toEqual(["quit", "later.dispose"]);
});
