import { afterEach, expect, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    InputPlugin,
    Outline,
    pressKey,
    releaseKey,
    Shape,
    StandardPhysicsPlugin,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { Action, createCrate, Game, InteractionInput, InteractPlugin, Usable } from "./interact";
import { Player } from "./player";
import { newGame, populate } from "./room";
import {
    decode,
    encode,
    Kind,
    migrate,
    type PreviousSave,
    persisted,
    readSave,
    Saved,
    Wallet,
} from "./save";

const apps: Awaited<ReturnType<typeof createApp>>[] = [];
afterEach(() => {
    for (const app of apps) app.dispose();
    apps.length = 0;
});
async function scene(action = Action.Coin as number) {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "GameCheck",
                recovery: "stateless",
                dependencies: [StandardPhysicsPlugin, InputPlugin, TransformPlugin],
                components: [...InteractPlugin.components!, ...Object.values(persisted), Outline],
                systems: InteractPlugin.systems,
            },
        ],
    });
    apps.push(app);
    const w = app.world;
    const camera = w.create();
    w.add(camera, Transform, { translation: [0, 1.4, 0] });
    w.add(camera, InteractionInput);
    const player = w.create();
    w.add(player, Wallet);
    const game = w.resource(Game);
    Object.assign(game, { camera, player });
    const target = w.create();
    w.add(target, Transform, { translation: [0, 1.4, -2], scale: [0.5, 0.5, 0.5] });
    w.add(target, Body, { type: BodyType.Static });
    w.add(target, Shape);
    w.add(target, Usable, { action });
    w.tick();
    w.step(1 / 60);
    const press = () => {
        pressKey(w, "KeyE");
        w.step(1 / 60);
        releaseKey(w, "KeyE");
        w.step(1 / 60);
    };
    return { w, camera, player, target, press };
}

test("E on a coin adds one to the money and removes it", async () => {
    const { w, player, target, press } = await scene();
    press();
    expect(w.storage(Wallet).money.get(player)).toBe(1);
    expect(w.has(target, Usable)).toBe(false);
});
test("E on the machine with three coins spends them and creates a crate", async () => {
    const { w, player, press } = await scene(Action.Machine);
    w.storage(Wallet).money.set(player, 3);
    press();
    expect(w.storage(Wallet).money.get(player)).toBe(0);
    expect([...w.query([Saved])].map((eid) => w.storage(Saved).kind.get(eid))).toEqual([
        Kind.Crate,
    ]);
});
test("E on the machine with fewer than three coins does nothing", async () => {
    const { w, player, press } = await scene(Action.Machine);
    w.storage(Wallet).money.set(player, 2);
    press();
    expect(w.storage(Wallet).money.get(player)).toBe(2);
    expect([...w.query([Saved])]).toEqual([]);
});
test("E on a crate carries it and E again sets it down", async () => {
    const { w, player, target, press } = await scene(Action.Crate);
    press();
    expect(w.storage(Wallet).carried.get(player)).toBe(target);
    expect(w.has(target, Body)).toBe(false);
    press();
    expect(w.storage(Wallet).carried.get(player)).toBe(0);
    expect(w.has(target, Body)).toBe(true);
    expect(w.storage(Transform).translation.y.get(target)).toBeCloseTo(0.35);
});

for (const [look, pitch] of [
    ["level", 0],
    ["up", 0.65],
    ["down", -0.65],
] as const) {
    test(`the view-centre aim ray misses the carried crate's box when looking ${look}`, async () => {
        const { w, camera, player } = await scene();
        const crate = createCrate(w);
        w.remove(crate, Shape);
        w.remove(crate, Body);
        w.storage(Wallet).carried.set(player, crate);
        const t = w.storage(Transform);
        t.rotation.set(camera, Math.sin(pitch / 2), 0, 0, Math.cos(pitch / 2));
        w.step(0);

        const q = t.rotation.read(crate, new Float32Array(4));
        const centre = t.translation.read(crate, new Float32Array(4));
        const eye = t.translation.read(camera, new Float32Array(4));
        const scale = t.scale.read(crate, new Float32Array(4));
        // Inverse crate rotation puts the ray in its box's local axes.
        const local = (v: number[]) => {
            const [x, y, z, s] = [-q[0], -q[1], -q[2], q[3]];
            const cross = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]];
            return [
                v[0] + 2 * (s * cross[0] + y * cross[2] - z * cross[1]),
                v[1] + 2 * (s * cross[1] + z * cross[0] - x * cross[2]),
                v[2] + 2 * (s * cross[2] + x * cross[1] - y * cross[0]),
            ];
        };
        const origin = local([eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]]);
        const direction = local([0, Math.sin(pitch), -Math.cos(pitch)]);
        let near = 0;
        let far = Number.POSITIVE_INFINITY;
        for (let axis = 0; axis < 3; axis++) {
            // Include the boundary despite f32 pose rounding: grazing still obscures aim.
            const half = scale[axis] / 2 + 1e-6;
            if (Math.abs(direction[axis]) < 1e-8) {
                if (Math.abs(origin[axis]) > half) far = -1;
            } else {
                const a = (-half - origin[axis]) / direction[axis];
                const b = (half - origin[axis]) / direction[axis];
                near = Math.max(near, Math.min(a, b));
                far = Math.min(far, Math.max(a, b));
            }
        }
        expect(near <= far).toBe(false);
        expect([...scale.slice(0, 3)]).toEqual([
            Math.fround(0.7),
            Math.fround(0.7),
            Math.fround(0.7),
        ]);
    });
}

test("save loaded into an offset fresh World preserves every field and remaps the carried crate", async () => {
    const { w } = await scene();
    const [camera, player] = newGame(w);
    const crate = w.create();
    w.add(crate, Saved, { kind: Kind.Crate });
    w.add(crate, Transform, {
        translation: [2, 1, -3],
        rotation: [0, 0.6, 0, 0.8],
        scale: [0.7, 0.7, 0.7],
    });
    w.storage(Wallet).carried.set(player, crate);
    w.storage(Wallet).money.set(player, 7);
    w.storage(Player).yaw.set(player, 0.7);
    w.storage(Player).pitch.set(player, -0.2);
    w.storage(Transform).translation.set(player, 1, 2, 3);
    const save = encode(w);
    const { w: fresh } = await scene();
    for (let i = 0; i < 20; i++) fresh.create();
    const eids = decode(fresh, JSON.parse(JSON.stringify(save)));
    expect(encode(fresh)).toEqual(save);
    const loadedPlayer = eids[1];
    const loadedCrate = eids.at(-1)!;
    expect(loadedCrate).not.toBe(crate);
    expect(fresh.storage(Wallet).carried.get(loadedPlayer)).toBe(loadedCrate);
    expect(fresh.storage(Player).camera.get(loadedPlayer)).toBe(eids[0]);
    expect(eids[0]).not.toBe(camera);
});

test("previous-version coins field migrates and loads equal to the current encoding", async () => {
    const { w } = await scene();
    const [, player] = newGame(w);
    w.storage(Wallet).money.set(player, 5);
    const current = encode(w);
    const previous: PreviousSave = {
        version: 1,
        entities: current.entities.map((entry) => {
            if (!entry.Wallet) return entry;
            const { money, ...wallet } = entry.Wallet;
            return { ...entry, Wallet: { ...wallet, coins: money } };
        }),
    };
    expect(migrate(previous)).toEqual(current);
    const { w: fresh } = await scene();
    decode(fresh, previous);
    expect(encode(fresh)).toEqual(current);
});
test("no save starts a new game", async () => {
    const { w } = await scene();
    const save = readSave({ getItem: () => null });
    expect(save).toBeNull();
    const eids = populate(w, save);
    expect(eids).toHaveLength(5);
    expect(w.storage(Wallet).money.get(eids[1])).toBe(0);
    expect(w.storage(Wallet).carried.get(eids[1])).toBe(0);
    expect(w.storage(Transform).translation.z.get(eids[1])).toBe(2);
});
