import { expect, test } from "bun:test";
import { World } from "../../../engine";
import { ecsShapeReference } from "../kernel/kernel";
import {
    bindShapeEntity,
    createShapeEntities,
    liveShapeEntity,
    pruneShapeEntities,
    retireShapeEntity,
    type ShapeEntity,
    shapeEntityAt,
} from "./shape-entities";
import type { WorldState } from "./world";

test("a slot retired and rebound at the same generation before its events are read is live again with the new entities", () => {
    const world = new World();
    const [shape, body, next, nextBody] = [1, 2, 3, 4].map(() => world.ref(world.create()));
    const entities = createShapeEntities();
    const out: ShapeEntity = { shape: 0 as never, body: 0 as never };
    bindShapeEntity(entities, 3, 5, shape, body);
    retireShapeEntity(entities, 3, 5);
    expect(liveShapeEntity(entities, 3, out)).toBe(false);
    bindShapeEntity(entities, 3, 5, next, nextBody);
    expect(liveShapeEntity(entities, 3, out)).toBe(true);
    expect(out).toEqual({ shape: next, body: nextBody });
    // Its events resolve to the live binding, ahead of the retired one.
    expect(shapeEntityAt(entities, 3, 5, out)).toBe(true);
    expect(out).toEqual({ shape: next, body: nextBody });
    world.dispose();
});

test("a kernel callback names a Shape retired since the last event read by its EntityRef, which does not alias the eid's next entity", () => {
    const world = new World();
    const eid = world.create();
    const shape = world.ref(eid);
    const body = world.ref(world.create());
    const state = { shapeEntities: createShapeEntities() } as WorldState;
    bindShapeEntity(state.shapeEntities, 0, 2, shape, body);
    retireShapeEntity(state.shapeEntities, 0, 2);
    world.destroy(eid);
    const recycled = world.create();
    expect(recycled).toBe(eid);
    // The slot's next shape is the recycled eid's, at the next generation.
    bindShapeEntity(state.shapeEntities, 0, 3, world.ref(recycled), body);
    const reference = ecsShapeReference(state, { index1: 1, world0: 0, generation: 2 });
    expect(reference).toBe(shape);
    expect(reference).not.toBe(world.ref(recycled));
    expect(world.resolve(reference)).toBe(0);
    pruneShapeEntities(state.shapeEntities);
    expect(() => ecsShapeReference(state, { index1: 1, world0: 0, generation: 2 })).toThrow(
        /stale/,
    );
    world.dispose();
});
