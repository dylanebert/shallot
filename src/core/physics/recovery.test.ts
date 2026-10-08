import { expect, test } from "bun:test";
import { World } from "../../engine";
import { Hulls } from "./hull";
import { PhysicsPlugin } from "./index";

test("core physics recovers mutable hull geometry, deleted ID reservations and future registration with reusable images", () => {
    const world = new World();
    if (typeof PhysicsPlugin.recovery !== "function")
        throw new Error("Physics must own hull recovery");
    world.registerRecovery(PhysicsPlugin.name, PhysicsPlugin.recovery(world));
    const hulls = world.resource(Hulls);
    const cube = structuredClone(hulls.get(hulls.name(0)!)!);
    expect(hulls.register({ ...structuredClone(cube), name: "deleted" })).toBe(1);
    expect(hulls.register({ ...structuredClone(cube), name: "live" })).toBe(2);
    hulls.delete("deleted");
    const saved = world.snapshot();
    hulls.get("live")!.verts[0][0] = 88;
    hulls.delete(cube.name);
    hulls.register({ ...structuredClone(cube), name: "future" });
    hulls.register({ ...structuredClone(cube), name: "deleted" });
    world.restore(saved);
    expect(world.resource(Hulls)).toBe(hulls);
    expect(hulls.get("live")).toEqual({ ...cube, name: "live" });
    expect(hulls.get(cube.name)).toEqual(cube);
    expect(hulls.has("deleted")).toBe(false);
    expect(hulls.id("deleted")).toBe(1);
    expect(hulls.id("future")).toBeUndefined();
    expect(hulls.size).toBe(2);
    hulls.get("live")!.faces[0].verts[0] = 99;
    world.restore(saved);
    expect(hulls.get("live")).toEqual({ ...cube, name: "live" });
    expect(hulls.register({ ...cube, name: "replayed" })).toBe(3);
    expect(hulls.register({ ...cube, name: "deleted" })).toBe(1);
    world.dispose();
});
