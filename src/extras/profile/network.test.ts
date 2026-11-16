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

test("counter rows derive rates from timed history, while gauges use smoothed readings", () => {
    const store = new DiagnosticsStore();
    const rows = new NetworkRows();
    for (const [path, suffix, first, last, expected] of [
        ["bytes_sent", "B", 1000, 1400, "200 B/s"],
        ["bytes_received", "B", 2000, 2600, "300 B/s"],
        ["messages_sent", "", 100, 130, "15 messages/s"],
        ["messages_received", "", 200, 250, "25 messages/s"],
        ["replayed_ticks", "ticks", 300, 370, "35 ticks/s"],
    ] as const) {
        store.record(`network/connections/1/${path}`, first, 0, suffix);
        expect(
            rows.collect(store, 0).find(([name]) => name.endsWith(path.replaceAll("_", " ")))?.[1],
        ).toBe("no data");
        store.record(`network/connections/1/${path}`, last, 2, suffix);
        expect(
            rows.collect(store, 1).find(([name]) => name.endsWith(path.replaceAll("_", " ")))?.[1],
        ).toBe(expected);
    }
    store.record("network/round_trip", 10, 0, "ms");
    store.record("network/round_trip", 20, 0.01, "ms");
    expect(rows.collect(store, 2).find(([name]) => name === "round trip")?.[1]).toBe("11.05 ms");
    store.record("network/confirmed_tick_lag", 12, 0, "ticks");
    expect(rows.collect(store, 2).find(([name]) => name === "confirmed tick lag")?.[1]).toBe(
        "12 ticks",
    );
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
        expect(rows.collect(store, 200).find(([path]) => path === "round trip")?.[1]).toBe("20 ms");
        // Hold the answering end's receive, leaving local polling and its age gauge running.
        for (let time = 210; time <= 1500; time += 10) {
            link.advance(time);
            left.poll();
            rows.collect(store, time);
        }
        const held = rows.collect(store, 1500);
        expect(held.find(([path]) => path === "round trip")?.[1]).toBe("no data");
        expect(held.find(([path]) => path === "time since receive")?.[1]).toContain("stale");
        for (let time = 1510; time <= 1600; time += 10) {
            link.advance(time);
            right.poll();
            left.poll();
        }
        const released = rows.collect(store, 1600);
        expect(released.find(([path]) => path === "round trip")?.[1]).not.toBe("no data");
        expect(released.find(([path]) => path === "time since receive")?.[1]).not.toContain(
            "stale",
        );
    } finally {
        a.dispose();
        b.dispose();
    }
});
