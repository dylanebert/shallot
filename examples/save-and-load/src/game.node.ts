import { afterAll, beforeAll, beforeEach, expect, setDefaultTimeout, test } from "bun:test";
import { type createApp, Outline, resizeViewport, Transform } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import {
    pointerMove,
    pressButton,
    pressKey,
    releaseButton,
    releaseKey,
} from "@dylanebert/shallot/input";
import { attachTexture, Views } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "../tests/gpu-app";
import { addCoin, count, Game, loadGame, newGame, saveGame } from "./game";
import { Coin, encode, SAVE_KEY } from "./save";
import { Scene } from "./scene";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof createApp>>;
const records = new Map<string, string>();
beforeAll(async () => {
    app = await openGpuApp({ plugins: [Scene] });
    const world = app.world;
    attachTexture(world, world.resource(Game).camera, { width: 320, height: 180 });
    world.resource(Views).get(world.resource(Game).camera)!.viewportIndex = 0;
    resizeViewport(world, 0, 320, 180, 1);
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
    expect(w.storage(Transform).translation.x.get(eid)).toBeGreaterThan(0.5);
    expect(w.storage(Coin).height.get(eid)).toBeCloseTo(-0.46);
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
    for (const eid of app.world.query([Coin])) {
        expect(app.world.storage(Transform).translation.x.get(eid)).toBeLessThan(1.7);
        expect(Math.abs(app.world.storage(Transform).translation.z.get(eid))).toBeLessThan(0.5);
    }
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
