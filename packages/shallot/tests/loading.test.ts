import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { shallotDark, minimalDark, canvasLoading } from "../src/standard/loading";

function createMockElement(tag: string) {
    const el: Record<string, any> = {
        tagName: tag.toUpperCase(),
        style: { cssText: "", width: "", position: "" },
        children: [] as any[],
        innerHTML: "",
        appendChild(child: any) {
            el.children.push(child);
            return child;
        },
        remove() {},
    };
    return el;
}

describe("Loading", () => {
    describe("factory exports", () => {
        test("shallotDark returns Loading interface", () => {
            const loading = shallotDark();
            expect(typeof loading.show).toBe("function");
            expect(typeof loading.update).toBe("function");
        });

        test("minimalDark returns Loading interface", () => {
            const loading = minimalDark();
            expect(typeof loading.show).toBe("function");
            expect(typeof loading.update).toBe("function");
        });

        test("canvasLoading aliases shallotDark", () => {
            expect(canvasLoading).toBe(shallotDark);
        });
    });

    describe("headless environment", () => {
        test("show returns void when document is undefined", () => {
            const loading = shallotDark();
            const cleanup = loading.show();
            expect(cleanup).toBeUndefined();
        });

        test("update before show does not throw", () => {
            const loading = shallotDark();
            expect(() => loading.update(0.5)).not.toThrow();
        });

        test("update after headless show does not throw", () => {
            const loading = shallotDark();
            loading.show();
            expect(() => loading.update(0.5)).not.toThrow();
        });
    });

    describe("DOM lifecycle", () => {
        let mockBody: Record<string, any>;
        let createdElements: Record<string, any>[];

        beforeEach(() => {
            createdElements = [];
            mockBody = createMockElement("body");
            mockBody.style.position = "static";

            const mockDoc = {
                createElement(tag: string) {
                    const el = createMockElement(tag);
                    createdElements.push(el);
                    return el;
                },
                querySelector() {
                    return null;
                },
                body: mockBody,
            };
            (globalThis as any).document = mockDoc;
            (globalThis as any).getComputedStyle = () => ({ position: "static" });
        });

        afterEach(() => {
            delete (globalThis as any).document;
            delete (globalThis as any).getComputedStyle;
        });

        test("show creates overlay and returns cleanup", () => {
            const loading = shallotDark();
            const cleanup = loading.show();
            expect(typeof cleanup).toBe("function");
            expect(mockBody.children.length).toBe(1);
        });

        test("cleanup removes overlay", () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            let removed = false;
            mockBody.children[0].remove = () => {
                removed = true;
            };
            cleanup();
            expect(removed).toBe(true);
        });

        test("update sets bar width percentage", () => {
            const loading = shallotDark();
            loading.show();
            loading.update(0.5);
            const bar = createdElements[3];
            expect(bar.style.width).toBe("50%");
        });

        test("update after cleanup does not throw", () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            cleanup();
            expect(() => loading.update(0.75)).not.toThrow();
        });

        test("shallot variant has logo and track", () => {
            const loading = shallotDark();
            loading.show();
            const overlay = mockBody.children[0];
            expect(overlay.children.length).toBe(2);
        });

        test("minimal variant has track only", () => {
            const loading = minimalDark();
            loading.show();
            const overlay = mockBody.children[0];
            expect(overlay.children.length).toBe(1);
        });

        test("minimal update sets bar width", () => {
            const loading = minimalDark();
            loading.show();
            loading.update(1.0);
            const bar = createdElements[2];
            expect(bar.style.width).toBe("100%");
        });

        test("sets parent position to relative when static", () => {
            const loading = shallotDark();
            loading.show();
            expect(mockBody.style.position).toBe("relative");
        });
    });
});
