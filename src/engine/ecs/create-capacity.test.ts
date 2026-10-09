import { expect, test } from "bun:test";
import { component, f32, vec2 } from "./component";
import { World } from "./world";

function expectColumnsCover(world: World, eid: number): void {
    for (const entry of world.storageEntries()) {
        for (const field of entry.fields.values()) {
            expect(field.column.length).toBeGreaterThanOrEqual((eid + 1) * field.type.lanes);
            field.column[eid * field.type.lanes] = 17;
            expect(field.column[eid * field.type.lanes]).toBe(17);
        }
    }
}

test("create grows every stored scalar and vector column past initial capacity", () => {
    const world = new World();
    const Scalar = component("CreateCapacityScalar", { value: f32 });
    const Vector = component("CreateCapacityVector", { value: vec2 });
    world.storage(Scalar);
    world.storage(Vector);

    let eid = 0;
    for (let i = 0; i < 20; i++) eid = world.create();

    expect(eid).toBeGreaterThanOrEqual(16);
    expectColumnsCover(world, eid);
    world.dispose();
});

test("create grows a field stored after another field's capacity grew", () => {
    const world = new World();
    const Existing = component("CreateCapacityExisting", { value: f32 });
    const Late = component("CreateCapacityLate", { value: vec2 });
    const existing = world.storage(Existing).value;
    world.create();
    const snapshot = world.snapshot();
    for (let i = 0; i < 20; i++) world.create();
    world.restore(snapshot);
    const late = world.storage(Late).value;

    expect(existing.column.length / existing.type.lanes).toBeGreaterThan(
        late.column.length / late.type.lanes,
    );
    expect(late.column.length).toBe(16 * late.type.lanes);

    let eid = 0;
    for (let i = 0; i < 20; i++) eid = world.create();

    expectColumnsCover(world, eid);
    world.dispose();
});

test("create grows columns again after restoring a lower entity high-water mark", () => {
    const world = new World();
    const Scalar = component("CreateCapacityRestoredScalar", { value: f32 });
    const Vector = component("CreateCapacityRestoredVector", { value: vec2 });
    world.storage(Scalar);
    world.storage(Vector);
    world.create();
    const snapshot = world.snapshot();

    for (let i = 0; i < 20; i++) world.create();
    world.restore(snapshot);
    let eid = 0;
    for (let i = 0; i < 40; i++) eid = world.create();

    expect(eid).toBeGreaterThanOrEqual(32);
    expectColumnsCover(world, eid);
    world.dispose();
});
