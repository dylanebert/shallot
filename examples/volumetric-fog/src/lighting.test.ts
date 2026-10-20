import { expect, test } from "bun:test";
import {
    DirectionalLight,
    NotShadowCaster,
    PointLight,
    SpotLight,
    VolumetricLight,
    World,
} from "@dylanebert/shallot";
import { Fog, FogPlugin } from "@dylanebert/shallot/extras";
import { Room, RoomPlugin } from "./room";

function scene() {
    const world = new World();
    FogPlugin.initialize!(world);
    RoomPlugin.initialize!(world);
    return { world, room: world.resource(Room) };
}
test("the hall contains scattering fog without self-illumination", () => {
    const { world } = scene();
    const fog = world.only([Fog]);
    expect(world.storage(Fog).density.get(fog)).toBeGreaterThan(0);
    expect(world.storage(Fog).brightness.get(fog)).toBe(0);
});
test("the window sun casts cascaded shadows", () => {
    const { world, room } = scene();
    expect(world.storage(DirectionalLight).shadowMapsEnabled.get(room.sun)).toBe(1);
    expect(world.storage(DirectionalLight).illuminance.get(room.sun)).toBeGreaterThan(0);
});
test("two warm shadowed point pendants and a fixed shadowed spot downlight light the far end", () => {
    const { world, room } = scene();
    for (const eid of [room.sun, room.downlight, ...room.pendants])
        expect(world.has(eid, VolumetricLight)).toBe(true);
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
