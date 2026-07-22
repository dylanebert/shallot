import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mountOverlay, State } from "../..";

// mountOverlay touches the DOM (createElement + appendChild), so a minimal document stub stands in for the
// browser under bun test. The assertion is the State-owned teardown: a state-bound overlay removes itself on
// state.dispose() (the leak class the onDispose primitive closes), and an unbound overlay stays the caller's.
function mockElement() {
    return {
        style: {} as Record<string, string>,
        children: [] as unknown[],
        removed: false,
        appendChild(child: unknown) {
            this.children.push(child);
            return child;
        },
        remove() {
            this.removed = true;
        },
    };
}

describe("mountOverlay", () => {
    let body: ReturnType<typeof mockElement>;
    const priorDocument = (globalThis as { document?: unknown }).document;

    beforeEach(() => {
        body = mockElement();
        (globalThis as { document: unknown }).document = {
            body,
            createElement: () => mockElement(),
        };
    });

    afterEach(() => {
        (globalThis as { document?: unknown }).document = priorDocument;
    });

    test("appends the overlay to document.body headless (no canvas parent)", () => {
        const overlay = mountOverlay(null);
        expect(body.children).toContain(overlay);
    });

    test("a state-bound overlay auto-removes on state dispose", () => {
        const state = new State();
        const overlay = mountOverlay(null, state) as unknown as ReturnType<typeof mockElement>;
        expect(overlay.removed).toBe(false);

        state.dispose();
        expect(overlay.removed).toBe(true);
    });

    test("an unbound overlay is the caller's to remove — dispose leaves it", () => {
        const state = new State();
        const overlay = mountOverlay(null) as unknown as ReturnType<typeof mockElement>;
        state.dispose();
        expect(overlay.removed).toBe(false);
    });
});
