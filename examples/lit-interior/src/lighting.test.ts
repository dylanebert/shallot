import { expect, test } from "bun:test";
import {
    DirectionalLight,
    NotShadowCaster,
    PointLight,
    SpotLight,
    World,
} from "@dylanebert/shallot";
import { SkyPlugin } from "@dylanebert/shallot/extras";
import { Room, RoomPlugin } from "./room";

function scene() {
    const world = new World();
    SkyPlugin.initialize!(world);
    RoomPlugin.initialize!(world);
    return { world, room: world.resource(Room) };
}
test("the window sun casts cascaded shadows", () => {
    const { world, room } = scene();
    expect(world.storage(DirectionalLight).shadowMapsEnabled.get(room.sun)).toBe(1);
    expect(world.storage(DirectionalLight).illuminance.get(room.sun)).toBeGreaterThan(0);
});
test("two warm shadowed point pendants and a fixed shadowed spot downlight light the far end", () => {
    const { world, room } = scene();
    expect(room.pendants).toHaveLength(2);
    for (const eid of room.pendants) {
        expect(world.storage(PointLight).shadowMapsEnabled.get(eid)).toBe(1);
        expect(world.storage(PointLight).intensity.get(eid)).toBeGreaterThan(0);
        expect(world.storage(PointLight).color.get(eid)).toBe(0xffd49a);
    }
    expect(world.storage(SpotLight).shadowMapsEnabled.get(room.downlight)).toBe(1);
    expect(world.storage(SpotLight).intensity.get(room.downlight)).toBeGreaterThan(0);
});
test("all three fixture bulbs do not cast shadows", () => {
    const { world, room } = scene();
    expect(room.bulbs).toHaveLength(3);
    for (const eid of room.bulbs) expect(world.has(eid, NotShadowCaster)).toBe(true);
});
