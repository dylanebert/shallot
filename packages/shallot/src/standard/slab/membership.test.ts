import { beforeEach, describe, expect, test } from "bun:test";
import { capacity, type Plugin, State } from "../..";
import { clear, register } from "../../engine/ecs/core";

// Pure-CPU ECS membership logic (no device). The membership→GPU word mirror a pack scan reads is
// real-GPU behavior, so its truth lives in the gym `render` scenario (load-bearing: no survivors
// without those bits) — `bun bench --scenario render`.

const A = {};
const B = {};
const MarkerPlugin: Plugin = { name: "Markers", components: { A, B } };

describe("component membership", () => {
    beforeEach(() => {
        clear();
        for (const [n, c] of Object.entries(MarkerPlugin.components ?? {})) register(n, c);
    });

    test("bit assigns a stable, distinct slot per component", () => {
        const state = new State();
        const a = state.membership.bit(A);
        expect(state.membership.bit(A)).toEqual(a); // stable across calls
        expect(state.membership.bit(B).mask).not.toBe(a.mask); // distinct bit
    });

    test("drain reports adds, clears the bit on remove and the word on destroy", () => {
        const state = new State();
        const a = state.membership.bit(A);
        const b = state.membership.bit(B);
        const words = new Uint32Array(state.membership.generations * capacity);
        const sink = (eid: number, gen: number, word: number) => {
            words[gen * capacity + eid] = word;
        };

        const eid = state.create();
        state.add(eid, A);
        state.add(eid, B);
        expect(state.membership.drain(sink)).toBe(true);
        expect(words[a.gen * capacity + eid] & a.mask).not.toBe(0);
        expect(words[b.gen * capacity + eid] & b.mask).not.toBe(0);

        expect(state.membership.drain(sink)).toBe(false);

        state.remove(eid, A);
        state.membership.drain(sink);
        expect(words[a.gen * capacity + eid] & a.mask).toBe(0); // A cleared
        expect(words[b.gen * capacity + eid] & b.mask).not.toBe(0); // B kept

        state.destroy(eid);
        state.membership.drain(sink);
        expect(words[b.gen * capacity + eid]).toBe(0); // whole word cleared
    });
});
