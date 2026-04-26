import { describe, test, expect } from "bun:test";
import { registry } from "../src/engine/utils/registry";

describe("Registry", () => {
    test("add and get", () => {
        const r = registry<string>(10);
        const id = r.add("hello");
        expect(id).toBe(0);
        expect(r.get(id)).toBe("hello");
        expect(r.get(1)).toBeUndefined();
    });

    test("add with name", () => {
        const r = registry<string>(10);
        const id = r.add("hello", "greeting");
        expect(r.getByName("greeting")).toBe(id);
        expect(r.getName(id)).toBe("greeting");
        expect(r.getByName("unknown")).toBeUndefined();
        expect(r.getName(99)).toBeUndefined();
    });

    test("add without name has no name", () => {
        const r = registry<string>(10);
        const id = r.add("anon");
        expect(r.getName(id)).toBeUndefined();
    });

    test("duplicate name updates in place", () => {
        const r = registry<string>(10);
        const id1 = r.add("v1", "key");
        const id2 = r.add("v2", "key");
        expect(id2).toBe(id1);
        expect(r.get(id1)).toBe("v2");
        expect(r.count()).toBe(1);
    });

    test("set replaces data at id", () => {
        const r = registry<string>(10);
        const id = r.add("old");
        r.set(id, "new");
        expect(r.get(id)).toBe("new");
    });

    test("version increments on add, set, clear", () => {
        const r = registry<string>(10);
        expect(r.version).toBe(0);
        r.add("a");
        expect(r.version).toBe(1);
        r.add("b", "named");
        expect(r.version).toBe(2);
        r.add("c", "named");
        expect(r.version).toBe(3);
        r.set(0, "x");
        expect(r.version).toBe(4);
        r.clear();
        expect(r.version).toBe(5);
    });

    test("all returns items", () => {
        const r = registry<number>(10);
        r.add(10);
        r.add(20);
        expect(r.all()).toEqual([10, 20]);
    });

    test("count tracks entries", () => {
        const r = registry<string>(10);
        expect(r.count()).toBe(0);
        r.add("a");
        r.add("b");
        expect(r.count()).toBe(2);
    });

    test("clear resets everything", () => {
        const r = registry<string>(10);
        r.add("a", "x");
        r.add("b", "y");
        r.clear();
        expect(r.count()).toBe(0);
        expect(r.get(0)).toBeUndefined();
        expect(r.getByName("x")).toBeUndefined();
        expect(r.getName(0)).toBeUndefined();
    });

    test("throws at capacity", () => {
        const r = registry<string>(2);
        r.add("a");
        r.add("b");
        expect(() => r.add("c")).toThrow("registry limit reached (2)");
    });

    test("duplicate name does not consume capacity", () => {
        const r = registry<string>(2);
        r.add("a", "key");
        r.add("b");
        r.add("c", "key");
        expect(r.count()).toBe(2);
    });

    test("set throws on out of bounds", () => {
        const r = registry<string>(10);
        r.add("a");
        expect(() => r.set(5, "x")).toThrow("out of bounds");
        expect(() => r.set(-1, "x")).toThrow("out of bounds");
    });
});
