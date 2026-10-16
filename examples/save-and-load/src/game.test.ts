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
import { Action, Game, InteractionInput, InteractPlugin, Usable } from "./interact";
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
