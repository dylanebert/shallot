import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { contactIds } from "../collision/contact";
import gold from "../collision/recycle.gold.json";
import { BodyField, bodyField } from "./bodyrecords";
import { SIM_STRIDE } from "./columns";
import * as layout from "./contact-layout";
import { kernel } from "./kernel";

const scratch = new Uint32Array(1);
const float = new Float32Array(scratch.buffer);
function decode(hex: string) {
    scratch[0] = Number.parseInt(hex.slice(2), 16);
    return float[0];
}
// The frozen pool's header preceded its inline points; resident manifolds put points first.
const map = [
    56,
    57,
    58,
    60,
    61,
    62,
    59,
    63,
    64,
    65,
    66,
    ...Array.from({ length: 56 }, (_, i) => i),
];
function collide(c: (typeof gold.cases)[number], recycle: boolean) {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody({ type: BodyType.Dynamic });
        a.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 2 });
        b.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 2 });
        world.step(1 / 60, 1);
        const k = kernel(world.state.ecsState);
        k.bodySetActiveWorld(world.state.worldId);
        const id = Array.from(contactIds(world.state))[0];
        expect(id).toBeDefined();
        k.allocateManifolds(id, c.manifoldCount);
        k.reserveCollide(1, 1, 1, decode(c.tol));
        world.state.bodyStore.refreshViews();
        world.state.manifoldStore.refreshViews();
        const bodies = world.state.bodyStore;
        const contactA =
            world.state.manifoldStore.dirU[id * layout.DIR_STRIDE + layout.ContactField.bodyIdA];
        const ordered = contactA === a.id.index1 - 1 ? [a, b] : [b, a];
        for (const [body, xf, center, extent] of [
            [ordered[0], c.xfA, c.centerA, c.maxExtentA],
            [ordered[1], c.xfB, c.centerB, c.maxExtentB],
        ] as const) {
            const local = bodyField(world.state, body.id.index1 - 1, BodyField.localIndex);
            const o = local * SIM_STRIDE;
            bodies.simF.set(xf.map(decode), o);
            bodies.simF.set(center.map(decode), o + 7);
            bodies.simF.set(extent.map(decode), o + 46);
        }
        const store = world.state.manifoldStore;
        const d = id * layout.DIR_STRIDE;
        store.dirU[d + layout.DIR_FLAGS] = 0x10 | 0x00010000 | (recycle ? 0x00800000 : 0);
        store.dirF.set(c.cachedRotA.map(decode), d + 20);
        store.dirF.set(c.cachedRotB.map(decode), d + 24);
        store.dirF.set(c.cachedRelPose.map(decode), d + 28);
        const base = store.dirU[d + layout.DIR_BLOCK] >>> 2;
        for (let m = 0; m < c.manifoldCount; ++m) {
            for (let i = 0; i < map.length; ++i)
                store.poolU[base + m * layout.MANIFOLD_STRIDE + map[i]] = Number.parseInt(
                    c.poolIn[(c.base + m) * 67 + i].slice(2),
                    16,
                );
        }
        new Uint32Array(k.memory.buffer, k.collideListPtr(), 1)[0] = id;
        k.dispatchContacts(1);
        store.refreshViews();
        const count = store.dirU[d + layout.DIR_COUNT];
        const after = store.dirU[d + layout.DIR_BLOCK] >>> 2;
        return {
            count,
            words: Array.from(store.poolU.slice(after, after + count * layout.MANIFOLD_STRIDE)),
        };
    } finally {
        world.destroy();
    }
}

for (const [index, c] of gold.cases.entries()) {
    test(`recycle native gold case ${index} through completed collide task`, () => {
        const result = collide(c, true);
        if (c.recycled === 0) {
            // Clearing has-manifold-cache disables recycling without changing the prior manifold
            // supplied to narrowphase; a partial gate write must not affect that path's result.
            expect(result).toEqual(collide(c, false));
        } else {
            expect(result.count).toBe(c.manifoldCount);
            for (let m = 0; m < c.manifoldCount; ++m) {
                for (let i = 0; i < map.length; ++i)
                    expect(result.words[m * 67 + map[i]]).toBe(
                        Number.parseInt(c.poolOut[(c.base + m) * 67 + i].slice(2), 16),
                    );
            }
        }
    });
}
