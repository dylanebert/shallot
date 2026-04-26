import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { State, events } from "../src";

describe("events", () => {
    let state: State;

    beforeEach(() => {
        state = new State();
    });

    afterEach(() => {
        state.dispose();
    });

    test("send and read in the same frame round-trips", () => {
        const Damage = events<{ amount: number }>("damage");
        Damage.send(state, { amount: 5 });
        Damage.send(state, { amount: 12 });

        const got = Damage.read(state);
        expect(got.length).toBe(2);
        expect(got[0]).toEqual({ amount: 5 });
        expect(got[1]).toEqual({ amount: 12 });
    });

    test("read returns empty for an unsent channel", () => {
        const Empty = events<number>("empty");
        expect(Empty.read(state).length).toBe(0);
    });

    test("read returns the same empty constant for unsent channels", () => {
        const A = events<number>("a");
        const B = events<number>("b");
        expect(A.read(state)).toBe(B.read(state));
    });

    test("read preserves send order", () => {
        const Tick = events<number>("tick");
        for (let i = 0; i < 100; i++) Tick.send(state, i);
        const got = Tick.read(state);
        for (let i = 0; i < 100; i++) expect(got[i]).toBe(i);
    });

    test("multiple readers see the same events", () => {
        const Pulse = events<string>("pulse");
        Pulse.send(state, "a");
        Pulse.send(state, "b");

        const reader1 = Pulse.read(state);
        const reader2 = Pulse.read(state);
        expect(reader1.length).toBe(2);
        expect(reader2.length).toBe(2);
        expect(reader1).toEqual(reader2);
    });

    test("channels with the same name are independent", () => {
        const A = events<number>("dup");
        const B = events<number>("dup");
        A.send(state, 1);
        B.send(state, 99);
        expect(A.read(state)).toEqual([1]);
        expect(B.read(state)).toEqual([99]);
    });

    test("distinct channels are isolated", () => {
        const Place = events<{ tile: number }>("place");
        const Destroy = events<{ tile: number }>("destroy");
        Place.send(state, { tile: 1 });
        Destroy.send(state, { tile: 99 });

        expect(Place.read(state)).toEqual([{ tile: 1 }]);
        expect(Destroy.read(state)).toEqual([{ tile: 99 }]);
    });

    test("state.step drains events automatically", () => {
        const Beat = events<number>("beat");
        Beat.send(state, 1);
        Beat.send(state, 2);
        state.step(1 / 60);
        expect(Beat.read(state).length).toBe(0);
    });

    test("send after step adds fresh events", () => {
        const Reset = events<number>("reset");
        Reset.send(state, 1);
        state.step(1 / 60);
        Reset.send(state, 2);
        expect(Reset.read(state)).toEqual([2]);
    });

    test("writers and readers within a single step share events", () => {
        const Hit = events<{ id: number }>("hit");
        const seen: number[] = [];

        const writer = {
            group: "simulation" as const,
            update: () => {
                Hit.send(state, { id: 1 });
                Hit.send(state, { id: 2 });
            },
        };
        const reader = {
            group: "simulation" as const,
            after: [writer],
            update: () => {
                for (const ev of Hit.read(state)) seen.push(ev.id);
            },
        };
        state.register(writer);
        state.register(reader);

        state.step(1 / 60);
        expect(seen).toEqual([1, 2]);
        expect(Hit.read(state).length).toBe(0);
    });

    test("read returns a live view; mutations after read are visible", () => {
        const Live = events<number>("live");
        Live.send(state, 1);
        const view = Live.read(state);
        expect(view.length).toBe(1);
        Live.send(state, 2);
        expect(view.length).toBe(2);
    });

    test("dispose drops event storage", () => {
        const Doomed = events<number>("doomed");
        Doomed.send(state, 1);
        state.dispose();
        state = new State();
        expect(Doomed.read(state).length).toBe(0);
    });

    test("handle exposes name", () => {
        const Named = events<number>("named");
        expect(Named.name).toBe("named");
    });

    test("storage is per-State", () => {
        const Channel = events<number>("channel");
        const other = new State();
        try {
            Channel.send(state, 1);
            expect(Channel.read(state)).toEqual([1]);
            expect(Channel.read(other)).toEqual([]);
        } finally {
            other.dispose();
        }
    });
});
