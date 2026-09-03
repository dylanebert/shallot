import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import { InputPlugin, Orbit, OrbitPlugin, OrbitTuningPlugin, State } from "../..";
import { clear, register } from "../../engine/ecs/core";

// biome-ignore lint/complexity/noBannedTypes: DOM listener mock accepts browser callbacks
type Fn = Function;

class MockElement {
    children: MockElement[] = [];
    parentElement: MockElement | null = null;
    style: Record<string, string> = {};
    textContent = "";
    listeners = new Map<string, Fn>();

    append(...children: MockElement[]): void {
        this.children.push(...children);
        for (const child of children) child.parentElement = this;
    }

    appendChild(child: MockElement): void {
        this.append(child);
    }

    addEventListener(type: string, listener: Fn): void {
        this.listeners.set(type, listener);
    }

    remove(): void {
        if (this.parentElement)
            this.parentElement.children = this.parentElement.children.filter(
                (child) => child !== this,
            );
    }

    setPointerCapture(): void {}
    releasePointerCapture(): void {}
    hasPointerCapture(): boolean {
        return false;
    }
    getBoundingClientRect(): DOMRect {
        return { left: 0, top: 0, width: 800, height: 600 } as DOMRect;
    }
}

describe("OrbitTuningPlugin", () => {
    let state: State;
    let windowListeners: Map<string, Fn>;
    let canvasParent: MockElement;
    let savedWindow: typeof globalThis.window;
    let savedDocument: typeof globalThis.document;

    beforeEach(() => {
        clear();
        savedWindow = globalThis.window;
        savedDocument = globalThis.document;
        windowListeners = new Map();
        globalThis.window = {
            addEventListener: (type: string, listener: Fn) => windowListeners.set(type, listener),
            removeEventListener() {},
            focus() {},
        } as unknown as typeof window;
        canvasParent = new MockElement();
        const canvas = new MockElement();
        canvas.parentElement = canvasParent;
        const body = new MockElement();
        globalThis.document = {
            body,
            pointerLockElement: null,
            createElement: () => new MockElement(),
            querySelector: (selector: string) => (selector === "canvas" ? canvas : null),
            querySelectorAll: (selector: string) => (selector === "canvas" ? [canvas] : []),
        } as unknown as typeof document;

        state = new State();
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        for (const [name, component] of Object.entries(InputPlugin.components ?? {}))
            register(name, component, InputPlugin.traits?.[name]);
        attach(state, InputPlugin);
        attach(state, OrbitTuningPlugin);
        const eid = state.create();
        state.add(eid, Orbit);
        state.step();
    });

    afterEach(() => {
        state.dispose();
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    test("a production digit event mutates Orbit and updates the mounted readout", () => {
        const overlay = canvasParent.children[0];
        const readout = overlay.children[0].children[0];
        expect(readout.textContent).toContain("rate 3.0 rad/s");

        windowListeners.get("keydown")!({ code: "Digit2" });
        state.step();

        const eid = [...state.query([Orbit])][0];
        expect(Orbit.keyRate.get(eid)).toBeCloseTo(3.1);
        expect(readout.textContent).toContain("rate 3.1 rad/s");
    });
});
