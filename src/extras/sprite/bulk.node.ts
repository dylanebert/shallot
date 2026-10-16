import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { Transform } from "../../core/transform";
import { MaterialAssets, MeshMaterial } from "../../standard/rendering";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [SpritePlugin] }]);

test("changing one field on 10,000 sprites makes no per-sprite JavaScript write", () => {
    const { world } = subjects()[0];
    const eids: number[] = [];
    for (let i = 0; i < 10_000; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Sprite);
        eids.push(eid);
    }
    world.step(0);
    const setAt = spyOn(MaterialAssets.prototype, "setAt");
    for (const eid of eids) world.storage(Sprite).opacity.set(eid, 0.5);
    const material = world.storage(MeshMaterial);
    const bounds = world.storage(MeshInstance).cullBounds;
    const fields = [
        material.type,
        material.material,
        material.alphaMode,
        material.alphaCutoff,
        bounds.x,
        bounds.y,
        bounds.z,
        bounds.w,
    ];
    const setters = fields.map((field) => field.set);
    let fieldWrites = 0;
    for (const field of fields) {
        const set = field.set.bind(field);
        field.set = (eid, value) => {
            fieldWrites++;
            set(eid, value);
        };
    }
    try {
        world.step(0);
        console.log(
            `10,000 changed sprites: ${setAt.mock.calls.length} MaterialAssets.setAt calls, ${fieldWrites} mesh field setters`,
        );
        expect(setAt.mock.calls.length).toBe(0);
        expect(fieldWrites).toBe(0);
    } finally {
        setAt.mockRestore();
        for (let i = 0; i < fields.length; i++) fields[i].set = setters[i];
    }
});
