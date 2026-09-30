import { expect, test } from "bun:test";
import { f32, field, State } from "./index";

test("a schema's scalar reader delegates directly to its owning world's reader and stays bound when retained", () => {
    const Value = { amount: field(f32) };
    const first = new State();
    const second = new State();
    try {
        const a = first.of(Value).amount;
        a.set(1, 1.25);
        const readA = Value.amount.get;
        expect(readA).toBe(a.get);
        const b = second.of(Value).amount;
        b.set(1, 2.75);
        expect(Value.amount.get).toBe(b.get);
        expect(readA(1)).toBe(1.25);
        expect(Value.amount.get(1)).toBe(2.75);
    } finally {
        first.dispose();
        second.dispose();
    }
});
