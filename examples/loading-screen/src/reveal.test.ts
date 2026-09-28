import { expect, test } from "bun:test";
import type { State } from "@dylanebert/shallot";
import { FRAME_BACKGROUND_COLOR, FRAME_ERROR_COLOR, mountHost } from "./host";
import { revealAfterFirstFrame } from "./reveal";

type FakeElement = {
    className: string;
    role: string;
    textContent: string | null;
    parent: FakeElement | null;
    children: FakeElement[];
    style: { zIndex: string; setProperty(name: string, value: string): void };
    innerHTML: string;
    classList: { add(name: string): void };
    setAttribute(name: string, value: string): void;
    appendChild(child: FakeElement): FakeElement;
    querySelector(selector: string): FakeElement | null;
};

function contrast(foreground: string, background: string): number {
    const luminance = (hex: string) => {
        const channels = [1, 3, 5].map(
            (offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255,
        );
        const linear = channels.map((channel) =>
            channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
        );
        return 0.2126 * linear[0]! + 0.7152 * linear[1]! + 0.0722 * linear[2]!;
    };
    const light = Math.max(luminance(foreground), luminance(background));
    const dark = Math.min(luminance(foreground), luminance(background));
    return (light + 0.05) / (dark + 0.05);
}

function fakeElement(): FakeElement {
    const element: FakeElement = {
        className: "",
        role: "",
        textContent: "",
        parent: null,
        children: [],
        style: { zIndex: "", setProperty() {} },
        innerHTML: "",
        classList: { add() {} },
        setAttribute(name, value) {
            if (name === "role") element.role = value;
        },
        appendChild(child) {
            child.parent = element;
            element.children.push(child);
            return child;
        },
        querySelector(selector) {
            return selector === ".frame-error"
                ? (element.children.find((child) => child.className === "frame-error") ?? null)
                : null;
        },
    };
    return element;
}

test("a missing device on the first stepped frame leaves a readable alert inside its scene frame, with at least 4.5:1 contrast", async () => {
    const errorContrast = contrast(FRAME_ERROR_COLOR, FRAME_BACKGROUND_COLOR);
    const app = fakeElement();
    const frame = fakeElement();
    const canvas = fakeElement();
    frame.appendChild(canvas);
    app.querySelector = (selector) =>
        selector === "#frame" ? frame : selector === "#scene" ? canvas : null;
    const fakeDocument = {
        documentElement: { style: { setProperty() {} } },
        querySelector: (selector: string) => (selector === "#app" ? app : null),
        createElement: () => fakeElement(),
    };
    const previousDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
    const previousConsoleError = console.error;
    let errorReported: unknown;
    let loadingError: unknown;
    let errorLine: FakeElement | null = null;

    try {
        Object.defineProperty(globalThis, "document", {
            configurable: true,
            writable: true,
            value: fakeDocument,
        });
        console.error = (error: unknown) => {
            errorReported = error;
        };
        const host = mountHost();
        const plugin = revealAfterFirstFrame(host, {
            error(error) {
                loadingError = error;
            },
        });
        const system = plugin.systems?.[0];
        system?.update?.({} as State);
        await Promise.resolve();
        errorLine = frame.querySelector(".frame-error");
    } finally {
        console.error = previousConsoleError;
        if (previousDocument) Object.defineProperty(globalThis, "document", previousDocument);
        else Reflect.deleteProperty(globalThis, "document");
    }

    const readable = Boolean(
        errorReported instanceof Error &&
            loadingError === errorReported &&
            errorLine?.parent === frame &&
            errorLine.role === "alert" &&
            errorLine.textContent?.includes("first frame had no WebGPU device"),
    );
    expect(readable, "missing-device failure renders an alert line in the scene frame").toBe(true);
    expect(
        errorContrast,
        "frame error ink meets 4.5:1 contrast against the page background",
    ).toBeGreaterThanOrEqual(4.5);
});
