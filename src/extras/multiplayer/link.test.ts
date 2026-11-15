import { expect, test } from "bun:test";
import { seededLink } from "./link";

test("seeded byte links copy ownership and reproduce loss, duplication and reordering", () => {
    function run() {
        const link = seededLink(7, { minDelay: 2, maxDelay: 9, loss: 0.2, duplication: 0.4 });
        const bytes = new Uint8Array(1);
        for (let i = 0; i < 60; i++) {
            bytes[0] = i;
            link.a.send(bytes);
        }
        bytes[0] = 255;
        link.advance(1);
        expect(link.b.receive()).toEqual([]);
        link.advance(10);
        const received = link.b.receive().map((b) => b[0]!);
        expect(received).not.toContain(255);
        expect(received).not.toEqual([...received].sort((a, b) => a - b));
        expect(received.length).toBeGreaterThan(new Set(received).size);
        expect(new Set(received).size).toBeLessThan(60);
        expect(link.counts.dropped).toBeGreaterThan(0);
        return received;
    }
    expect(run()).toEqual(run());
});
