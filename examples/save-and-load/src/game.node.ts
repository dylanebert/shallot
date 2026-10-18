import { afterAll, beforeAll, beforeEach, expect, setDefaultTimeout, test } from "bun:test";
import { type createApp, internText, Outline, Text, Transform } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import {
    pointerMove,
    pressButton,
    pressKey,
    releaseButton,
    releaseKey,
} from "@dylanebert/shallot/input";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { addCoin, count, Game, loadGame, newGame, saveGame, saveLabel } from "./game";
import { COIN_CAP, COIN_DIAMETER, Coin, encode, placement, SAVE_KEY } from "./save";
import { Scene } from "./scene";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof createApp>>;
const records = new Map<string, string>();
beforeAll(async () => {
    app = await openGpuApp({ plugins: [Scene] });
    const world = app.world;
    attachTexture(world, world.resource(Game).camera, { width: 320, height: 180 });
    world.resource(Game).storage = {
        getItem: (key) => records.get(key) ?? null,
        setItem: (key, value) => {
            records.set(key, value);
        },
    };
    world.step(1 / 60);
    world.step(1 / 60);
});
afterAll(() => app?.dispose());
beforeEach(() => {
    newGame(app.world);
    records.clear();
    app.world.resource(Game).saved = false;
    pointerMove(app.world, { x: 160, y: 90, hover: true });
});
function key(code: string) {
    pressKey(app.world, code);
    app.world.step(0);
    releaseKey(app.world, code);
    app.world.step(0);
}

test("hovering the cube outlines it and leaving removes the outline", () => {
    const w = app.world;
    w.step(0);
    expect(w.has(w.resource(Game).cube, Outline)).toBe(true);
    pointerMove(w, { x: 0, y: 0, hover: false });
    w.step(0);
    expect(w.has(w.resource(Game).cube, Outline)).toBe(false);
});
test("a click on the cube adds one coin entity to the pile and count", () => {
    const w = app.world;
    pressButton(w, "left");
    releaseButton(w, "left");
    w.step(0);
    expect(count(w)).toBe(1);
    const eid = [...w.query([Coin])][0]!;
    expect(w.storage(Transform).translation.z.get(eid)).toBeGreaterThan(0.5);
    expect(w.storage(Coin).height.get(eid)).toBeCloseTo(placement(0)[1]);
});
test("Save then New game then Load restores every coin's saved fields in the running World", () => {
    const w = app.world;
    for (let i = 0; i < 3; i++) addCoin(w);
    const saved = encode(w);
    saveGame(w);
    newGame(w);
    expect(count(w)).toBe(0);
    loadGame(w);
    expect(encode(w).entities).toEqual(expect.arrayContaining(saved.entities));
    expect(w.has(w.resource(Game).cube, Transform)).toBe(true);
});
test("each shortcut does what its button does", () => {
    const w = app.world;
    addCoin(w);
    key("KeyS");
    expect(w.resource(Game).feedback).toBe(2);
    const saved = encode(w);
    key("KeyN");
    expect(count(w)).toBe(0);
    key("KeyL");
    expect(encode(w)).toEqual(saved);
});
test("a previous-version bare count loads as that many coin entities", () => {
    records.set(SAVE_KEY, JSON.stringify({ version: 1, count: 13 }));
    loadGame(app.world);
    expect(count(app.world)).toBe(13);
    const spots = encode(app.world).entities.map((e) => e.Transform!.translation as number[]);
    for (let i = 0; i < 13; i++)
        expect(
            spots.some((p) => Math.hypot(p[0]! - placement(i)[0], p[2]! - placement(i)[2]) < 1e-6),
        ).toBe(true);
});
test("with no save Load does nothing", () => {
    expect(app.world.resource(Game).saved).toBe(false);
    loadGame(app.world);
    expect(count(app.world)).toBe(0);
});
test("a drag past the threshold that starts on the cube orbits and adds none", () => {
    const w = app.world,
        camera = w.resource(Game).camera;
    const yaw = w.storage(Orbit).yaw.get(camera);
    pressButton(w, "left");
    w.step(1 / 60);
    pointerMove(w, { x: 190, y: 90, deltaX: 30, hover: true });
    w.step(1 / 60);
    releaseButton(w, "left");
    w.step(1 / 60);
    expect(count(w)).toBe(0);
    expect(w.storage(Orbit).yaw.get(camera)).not.toBe(yaw);
});

function checkScatter() {
    const w = app.world;
    const positions = [...w.query([Coin])].map((eid) => {
        const t = w.storage(Transform).translation;
        return [t.x.get(eid), w.storage(Coin).height.get(eid), t.z.get(eid)];
    });
    expect(positions).toHaveLength(COIN_CAP);
    for (const [i, p] of positions.entries()) {
        expect(Math.abs(p[0]!) + COIN_DIAMETER / 2).toBeLessThan(2);
        expect(Math.abs(p[2]!) + COIN_DIAMETER / 2).toBeLessThan(2);
        expect(p[1]).toBeCloseTo(placement(i)[1]);
        for (const q of positions.slice(i + 1))
            expect(Math.hypot(p[0]! - q[0]!, p[2]! - q[2]!)).toBeGreaterThan(COIN_DIAMETER);
    }
}
test("no two coins overlap and every coin lies on the platform at the cap", () => {
    for (let i = 0; i < COIN_CAP; i++) addCoin(app.world);
    checkScatter();
});
test("no two coins overlap and every coin lies on the platform in a migrated record", () => {
    records.set(SAVE_KEY, JSON.stringify({ version: 1, count: COIN_CAP }));
    loadGame(app.world);
    checkScatter();
});
test("a ray from each coin's centre toward the sun misses the cube", () => {
    // Slab intersection with the cube [-0.5, 0.5]^3; sun direction is (-1, 2, 0.35).
    for (let i = 0; i < COIN_CAP; i++) {
        const p = placement(i),
            direction = [-1, 2, 0.35];
        let near = 0,
            far = Infinity;
        for (let axis = 0; axis < 3; axis++) {
            const a = (-0.5 - p[axis]!) / direction[axis]!;
            const b = (0.5 - p[axis]!) / direction[axis]!;
            near = Math.max(near, Math.min(a, b));
            far = Math.min(far, Math.max(a, b));
        }
        expect(near > far).toBe(true);
    }
});
test("at the cap a click adds none and the cube text says full", () => {
    const w = app.world;
    for (let i = 0; i < COIN_CAP; i++) addCoin(w);
    pressButton(w, "left");
    releaseButton(w, "left");
    w.step(0);
    expect(count(w)).toBe(COIN_CAP);
    expect(w.storage(Text).content.get(w.resource(Game).label)).toBe(
        internText(w, "The pile is full"),
    );
});
test("Save's label says saved and returns", () => {
    saveGame(app.world);
    expect(saveLabel(app.world)).toBe("Saved");
    for (let i = 0; i < 130; i++) app.world.step(1 / 60);
    expect(saveLabel(app.world)).toBe("Save");
});
