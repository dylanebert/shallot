import { expect, test } from "bun:test";
import { CAPTURE_CONTRACT } from "./capture";

test("a frame capture uses the declared final-canvas geometry and encoding", () => {
    expect(CAPTURE_CONTRACT).toEqual({
        width: 1280,
        height: 720,
        deviceScale: 1,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
});
