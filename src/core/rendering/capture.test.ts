import { expect, test } from "bun:test";
import { CAPTURE_CONTRACT } from "./capture";

test("CAPTURE_CONTRACT is the size and scale fixtures set their viewport to", () => {
    expect(CAPTURE_CONTRACT).toEqual({
        width: 1280,
        height: 720,
        deviceScale: 1,
        surface: "final-canvas",
        encoding: "rgba8-tight",
    });
});
