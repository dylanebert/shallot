import { test, expect, describe, beforeEach } from "bun:test";
import { build, State } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Player, PlayerPlugin } from "../src/standard/player";
import { Transform, TransformsPlugin } from "../src/standard/transforms";
import { count, first, spawn } from "./helpers/state";

describe("Player", () => {
    let state: State;

    beforeEach(() => {
        clearRegistry();
        state = new State();
        state.register(PlayerPlugin);
        state.register(TransformsPlugin);
    });

    describe("component defaults", () => {
        test("yaw defaults to 0", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(Player.yaw[eid]).toBe(0);
        });

        test("pitch defaults to 0", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(Player.pitch[eid]).toBe(0);
        });

        test("speed defaults to 6", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(Player.speed[eid]).toBe(6);
        });

        test("sensitivity defaults to 1.5", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(Player.sensitivity[eid]).toBe(1.5);
        });
    });

    describe("plugin registration", () => {
        test("Player component is queryable after registration", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(count(state, [Player])).toBe(1);
            expect(first(state, [Player])).toBe(eid);
        });

        test("entity has both Player and Transform after spawn", () => {
            const eid = spawn(state, [Player], [Transform]);
            expect(state.hasComponent(eid, Player)).toBe(true);
            expect(state.hasComponent(eid, Transform)).toBe(true);
        });
    });

    describe("component CRUD", () => {
        test("remove Player", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            expect(count(state, [Player])).toBe(1);

            state.removeComponent(eid, Player);
            expect(count(state, [Player])).toBe(0);
        });

        test("custom values persist on component", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Player);
            Player.yaw[eid] = 1.5;
            Player.pitch[eid] = -0.3;
            Player.speed[eid] = 7;
            Player.sensitivity[eid] = 2.0;

            expect(Player.yaw[eid]).toBe(1.5);
            expect(Player.pitch[eid]).toBe(-0.3);
            expect(Player.speed[eid]).toBe(7);
            expect(Player.sensitivity[eid]).toBe(2.0);
        });

        test("multiple players have independent values", () => {
            const a = state.addEntity();
            const b = state.addEntity();
            state.addComponent(a, Player);
            state.addComponent(b, Player);

            Player.speed[a] = 5;
            Player.speed[b] = 10;

            expect(Player.speed[a]).toBe(5);
            expect(Player.speed[b]).toBe(10);
            expect(count(state, [Player])).toBe(2);
        });
    });

    describe("Player + Transform query", () => {
        test("entity with both components is queryable", () => {
            const eid = spawn(state, [Player], [Transform]);
            expect(count(state, [Player, Transform])).toBe(1);
            expect(first(state, [Player, Transform])).toBe(eid);
        });

        test("entity with only Player is not in joint query", () => {
            const playerOnly = state.addEntity();
            state.addComponent(playerOnly, Player);
            expect(count(state, [Player, Transform])).toBe(0);
        });

        test("entity with only Transform is not in joint query", () => {
            const transformOnly = state.addEntity();
            state.addComponent(transformOnly, Transform);
            expect(count(state, [Player, Transform])).toBe(0);
        });
    });

    describe("system without DOM", () => {
        test("stepping without DOM does not crash", async () => {
            clearRegistry();
            const s = await build({
                plugins: [PlayerPlugin, TransformsPlugin],
                defaults: false,
            });
            spawn(s, [Player], [Transform]);
            s.step(1 / 60);
            s.step(1 / 60);
        });

        test("transform unchanged when system has no DOM resources", async () => {
            clearRegistry();
            const s = await build({
                plugins: [PlayerPlugin, TransformsPlugin],
                defaults: false,
            });
            const eid = spawn(s, [Player], [Transform]);
            Transform.posX[eid] = 5;
            Transform.posY[eid] = 10;
            Transform.posZ[eid] = 15;

            s.step(1 / 60);

            expect(Transform.posX[eid]).toBe(5);
            expect(Transform.posY[eid]).toBe(10);
            expect(Transform.posZ[eid]).toBe(15);
        });
    });
});
