import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../ecs";
import { Preloads, preload } from "./preload";
import { parse } from "./xml";

describe("scene preload", () => {
    beforeEach(() => {
        Preloads.delete("a");
        Preloads.delete("b");
    });

    test("runs every registered resolver over the parsed nodes, awaited", async () => {
        const state = new State();
        const seen: string[] = [];
        Preloads.register({
            name: "a",
            resolve: async (nodes, s) => {
                await Promise.resolve();
                expect(s).toBe(state);
                seen.push(`a:${nodes.length}`);
            },
        });
        Preloads.register({
            name: "b",
            resolve: (nodes) => {
                seen.push(`b:${nodes.length}`);
            },
        });

        await preload(parse(`<scene><a /><a /></scene>`), state);
        // the async resolver completed before preload resolved — the pass is awaited, not fire-and-forget
        expect(seen).toEqual(["a:2", "b:2"]);
    });

    test("a deleted resolver no longer runs (the disabled-plugin path)", async () => {
        const state = new State();
        let ran = 0;
        Preloads.register({
            name: "a",
            resolve: () => {
                ran++;
            },
        });
        await preload([], state);
        Preloads.delete("a");
        await preload([], state);
        expect(ran).toBe(1);
    });

    test("re-registering under one name replaces, not stacks (idempotent initialize)", async () => {
        const state = new State();
        const seen: string[] = [];
        Preloads.register({ name: "a", resolve: () => void seen.push("old") });
        Preloads.register({ name: "a", resolve: () => void seen.push("new") });
        await preload([], state);
        expect(seen).toEqual(["new"]);
    });
});
