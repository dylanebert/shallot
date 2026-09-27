import { resetCompute, type State } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { mountHost } from "./host";
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

check(
    "a missing device on the first stepped frame leaves a readable error in its scene frame",
    {
        claim: "a missing device on the first stepped frame leaves a readable error inside its scene frame",
        size: "unit",
        subject: ["examples/loading-screen/src/host.ts", "examples/loading-screen/src/reveal.ts"],
    },
    async () => {
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
            resetCompute();
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
            resetCompute();
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
        return {
            ok: readable,
            checks: [
                {
                    name: "missing-device failure renders an alert line in the scene frame",
                    ok: readable,
                },
            ],
        };
    },
);
