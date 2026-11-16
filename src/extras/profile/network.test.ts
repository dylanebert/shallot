import { expect, test } from "bun:test";
import { createApp, Diagnostic, DiagnosticsStore } from "../../engine";
import { observeNetwork } from "../multiplayer/diagnostics";
import { seededLink } from "../multiplayer/link";
import { NetworkRows } from "./network";

test("network section rows exist only while network diagnostics are registered", () => {
    const store = new DiagnosticsStore();
    const rows = new NetworkRows();
    store.record("fps", 60, 0);
    expect(rows.collect(store, 0)).toEqual([]);
    store.add(new Diagnostic("network/custom", { suffix: "ticks" }));
    expect(rows.collect(store, 0)).toEqual([["custom", "no data"]]);
    store.record("network/custom", 3, 1);
    expect(rows.collect(store, 0)).toEqual([["custom", "3 ticks"]]);
    expect(rows.collect(store, 1001)).toEqual([["custom", "no data"]]);
    store.record("network/custom", 4, 2);
    expect(rows.collect(store, 1002)).toEqual([["custom", "4 ticks"]]);
    store.remove("network/custom");
    expect(rows.collect(store, 1003)).toEqual([]);
});

test("held receive marks the connection stale and its round trip no-data, release restores both", async () => {
    const a = await createApp({ defaults: false, plugins: [] });
    const b = await createApp({ defaults: false, plugins: [] });
    const faults = { minDelay: 10, maxDelay: 10, loss: 0, duplication: 0 };
    const link = seededLink(1, faults);
    const left = observeNetwork(a.world, link.a);
    const right = observeNetwork(b.world, link.b);
    const rows = new NetworkRows();
    const store = a.world.resource(DiagnosticsStore);
    try {
        for (let time = 0; time <= 200; time += 10) {
            link.advance(time);
            left.poll();
            right.poll();
            rows.collect(store, time);
        }
        expect(rows.collect(store, 200).find(([path]) => path === "round_trip")?.[1]).toBe("20 ms");
        // Hold the answering end's receive, leaving local polling and its age gauge running.
        for (let time = 210; time <= 1500; time += 10) {
            link.advance(time);
            left.poll();
            rows.collect(store, time);
        }
        const held = rows.collect(store, 1500);
        expect(held.find(([path]) => path === "round_trip")?.[1]).toBe("no data");
        expect(held.find(([path]) => path === "time_since_receive")?.[1]).toContain("stale");
        for (let time = 1510; time <= 1600; time += 10) {
            link.advance(time);
            right.poll();
            left.poll();
        }
        const released = rows.collect(store, 1600);
        expect(released.find(([path]) => path === "round_trip")?.[1]).not.toBe("no data");
        expect(released.find(([path]) => path === "time_since_receive")?.[1]).not.toContain(
            "stale",
        );
    } finally {
        a.dispose();
        b.dispose();
    }
});
