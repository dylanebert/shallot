import { expect, test } from "bun:test";
import { reportStartup } from "./startup";

test("browser startup observations reach current listeners, without replaying retained state", () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, "window");
    const browser = new EventTarget();
    Object.defineProperty(globalThis, "window", { configurable: true, value: browser });
    try {
        const observed: unknown[] = [];
        browser.addEventListener("shallot:startup", (event) =>
            observed.push((event as CustomEvent).detail),
        );
        reportStartup({ kind: "test", value: 1 });
        const later: unknown[] = [];
        browser.addEventListener("shallot:startup", (event) =>
            later.push((event as CustomEvent).detail),
        );
        expect(later).toEqual([]);
        reportStartup({ kind: "test", value: 2 });
        expect(observed).toEqual([
            { kind: "test", value: 1 },
            { kind: "test", value: 2 },
        ]);
        expect(later).toEqual([{ kind: "test", value: 2 }]);
    } finally {
        if (original) Object.defineProperty(globalThis, "window", original);
        else Reflect.deleteProperty(globalThis, "window");
    }
});

test("startup reporting does not require a browser host", () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, "window");
    Reflect.deleteProperty(globalThis, "window");
    try {
        expect(() => reportStartup({ kind: "test" })).not.toThrow();
    } finally {
        if (original) Object.defineProperty(globalThis, "window", original);
    }
});
