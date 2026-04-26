import { test, expect, describe, beforeEach } from "bun:test";
import { State } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Outline, OutlinePlugin } from "../src/extras/outline";
import { RenderPlugin } from "../src/standard/render";

describe("Outline", () => {
    describe("resource set/get", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            state = new State();
        });

        test("set and read back resource", () => {
            const value: Outline = {
                getEntities: () => [1, 2, 3],
                color: 0xff0000,
                thickness: 2,
            };
            state.setResource(Outline, value);

            const result = Outline.from(state);
            expect(result).toBeDefined();
            expect(result!.color).toBe(0xff0000);
            expect(result!.thickness).toBe(2);
            expect(result!.getEntities()).toEqual([1, 2, 3]);
        });

        test("from returns undefined when not set", () => {
            expect(Outline.from(state)).toBeUndefined();
        });
    });

    describe("resource lifecycle", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            state = new State();
        });

        test("delete removes resource", () => {
            state.setResource(Outline, {
                getEntities: () => [],
                color: 0xffffff,
                thickness: 1,
            });
            expect(Outline.from(state)).toBeDefined();

            state.deleteResource(Outline);
            expect(Outline.from(state)).toBeUndefined();
        });

        test("set overwrites previous value", () => {
            state.setResource(Outline, {
                getEntities: () => [],
                color: 0xff0000,
                thickness: 1,
            });
            state.setResource(Outline, {
                getEntities: () => [5],
                color: 0x00ff00,
                thickness: 3,
            });

            const result = Outline.from(state);
            expect(result!.color).toBe(0x00ff00);
            expect(result!.thickness).toBe(3);
            expect(result!.getEntities()).toEqual([5]);
        });
    });

    describe("getEntities dynamic callback", () => {
        test("returns different values on successive calls", () => {
            const state = new State();
            let entities = [1, 2];

            state.setResource(Outline, {
                getEntities: () => entities,
                color: 0xffffff,
                thickness: 1,
            });

            const outline = Outline.from(state)!;
            expect(outline.getEntities()).toEqual([1, 2]);

            entities = [3, 4, 5];
            expect(outline.getEntities()).toEqual([3, 4, 5]);

            entities = [];
            expect(outline.getEntities()).toEqual([]);
        });
    });

    describe("plugin registration", () => {
        test("registers without error when Render absent", () => {
            clearRegistry();
            const state = new State();
            expect(() => state.register(OutlinePlugin)).not.toThrow();
        });

        test("registers with RenderPlugin without error", () => {
            clearRegistry();
            const state = new State();
            state.register(RenderPlugin);
            expect(() => state.register(OutlinePlugin)).not.toThrow();
        });
    });
});
