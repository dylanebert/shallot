import { expect, test } from "bun:test";
import { f32, World } from "./index";

test("a world-bound scalar reader stays bound when another world resolves the same schema", () => {
    const Value = { amount: f32 };
    const first = new World();
    const second = new World();
    try {
        const a = first.storage(Value).amount;
        a.set(1, 1.25);
        const readA = first.storage(Value).amount.get;
        expect(readA).toBe(a.get);
        const b = second.storage(Value).amount;
        b.set(1, 2.75);
        expect(second.storage(Value).amount.get).toBe(b.get);
        expect(readA(1)).toBe(1.25);
        expect(second.storage(Value).amount.get(1)).toBe(2.75);
    } finally {
        first.dispose();
        second.dispose();
    }
});
