import { expect, test } from "bun:test";
import {
    GlobalTransform,
    initializeGlobalTransform,
    registerGlobalTransform,
    Transform,
} from "./global-transform";
import { World } from "./world";

test("placement initialization installs processing on a bare World for frames and exact ticks", () => {
    const world = new World();
    registerGlobalTransform(world);
    initializeGlobalTransform(world);
    const eid = world.create();
    world.add(eid, Transform);
    const source = world.storage(Transform).translation;
    const global = world.storage(GlobalTransform).translation;
    source.x.set(eid, 3);
    world.step(0);
    expect(global.x.get(eid)).toBe(3);
    source.x.set(eid, 7);
    world.tick();
    expect(global.x.get(eid)).toBe(7);
    world.dispose();
});
