import { expect, test } from "bun:test";
import { attribute } from "./allocation-sampler.mjs";
import { browserAllocated } from "./display";

// A page frame: an engine function with a script URL, or a native one (no URL), as the CPU profile reports them.
type PageFrame = { name: string; url: string };
const engine = (name: string): PageFrame => ({ name, url: "http://page/assets/index.js" });
const native = (name: string): PageFrame => ({ name, url: "" });
// The harness's forwarding function at each WebGPU method, injected by init script and so without a URL.
const boundary = native("shallotWebGpuBoundary");
type ProfileNode = Parameters<typeof attribute>[0]["head"];

/** Whether `browserAllocated` admits one sample whose frames below the page's run frame are `frames`. */
function admitted(...frames: PageFrame[]): boolean {
    const node = (id: number, frame: PageFrame, children: ProfileNode[] = []): ProfileNode => ({
        id,
        callFrame: {
            functionName: frame.name,
            url: frame.url,
            scriptId: String(id),
            lineNumber: 0,
            columnNumber: 0,
        },
        children,
    });
    let tail: ProfileNode[] = [];
    for (let i = frames.length - 1; i >= 0; i--) tail = [node(i + 3, frames[i], tail)];
    const profile = {
        head: node(1, native("(root)"), [node(2, engine("frame"), tail)]),
        samples: [{ nodeId: frames.length + 2, size: 16 }],
    };
    const rows = attribute(
        profile,
        (frame: { functionName: string }) => (frame.functionName === "frame" ? "frame" : undefined),
        () => undefined,
    );
    expect(rows).toHaveLength(1);
    return browserAllocated(rows[0]);
}

test("a sample with only native frames below the WebGPU boundary, or made by the boundary itself, is the browser's", () => {
    expect(admitted(engine("submitDrawEncoder"), boundary, native("submit"))).toBe(true);
    // the boundary's own arguments object
    expect(admitted(engine("submitDrawEncoder"), boundary)).toBe(true);
});

test("a sample in a JavaScript frame the browser calls back into below the WebGPU boundary is the engine's", () => {
    expect(admitted(engine("record"), boundary, engine("callback"))).toBe(false);
    expect(admitted(engine("record"), boundary, engine("callback"), native("Array"))).toBe(false);
});

test("an all-native sample with no WebGPU boundary on its stack is the engine's", () => {
    expect(admitted(native("requestAnimationFrame"))).toBe(false);
    expect(admitted(engine("step"), native("submit"))).toBe(false);
});

test("under nested WebGPU boundaries, only the frames below the innermost one decide", () => {
    expect(
        admitted(engine("record"), boundary, engine("callback"), boundary, native("createBindGroup")),
    ).toBe(true);
});

test("a sample whose innermost frame is any native then is the browser's, not only the completion fence's", () => {
    expect(admitted(engine("submitDrawEncoder"), native("then"))).toBe(true);
    expect(admitted(engine("anyEngineFunction"), native("then"))).toBe(true);
});

test("a sample in a JavaScript then, or below a native then, is the engine's", () => {
    expect(admitted(engine("step"), engine("then"))).toBe(false);
    expect(admitted(engine("step"), native("then"), engine("reaction"))).toBe(false);
});

test("a sample row without a stack is the engine's", () => {
    expect(browserAllocated({ site: "frame", bytes: 16, count: 1 })).toBe(false);
});
