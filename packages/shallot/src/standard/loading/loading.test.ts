import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { minimalDark, shallotDark } from "./";

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
    });

    describe("headless environment", () => {
        test("show returns void when document is undefined", () => {
            const loading = shallotDark();
            const cleanup = loading.show();
            expect(cleanup).toBeUndefined();
        });
        // the null-bar no-op path `update()` takes here (no overlay, no bar) is positively
        // asserted by "update after cleanup leaves the bar untouched" in the DOM block — same branch
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
            const bar = createdElements[4];
            expect(bar.style.width).toBe("50%");
        });

        test("update after cleanup leaves the bar untouched", () => {
            const loading = shallotDark();
            const cleanup = loading.show()!;
            loading.update(0.5);
            const bar = createdElements[4]; // overlay, panel, logo, track, bar
            expect(bar.style.width).toBe("50%");

            cleanup(); // nulls the bar ref
            loading.update(0.75); // no-op: nothing to write to
            expect(bar.style.width).toBe("50%"); // detached bar unchanged, not "75%"
        });

        test("shallot variant has logo and track", () => {
            const loading = shallotDark();
            loading.show();
            const panel = mockBody.children[0].children[0]; // overlay → centered panel
            expect(panel.children.length).toBe(2);
        });

        test("minimal variant has track only", () => {
            const loading = minimalDark();
            loading.show();
            const panel = mockBody.children[0].children[0]; // overlay → centered panel
            expect(panel.children.length).toBe(1);
        });

        test("minimal update sets bar width", () => {
            const loading = minimalDark();
            loading.show();
            loading.update(1.0);
            const bar = createdElements[3]; // overlay, panel, track, bar
            expect(bar.style.width).toBe("100%");
        });

        test("sets parent position to relative when static", () => {
            const loading = shallotDark();
            loading.show();
            expect(mockBody.style.position).toBe("relative");
        });
    });
});
