import { expect, test } from "bun:test";
import { UnsupportedError } from "./gpu";
import { assertDawnHost } from "./webgpu";

test("Windows Dawn host guard names the unsupported executable and accepts both working hosts", () => {
    let failure: unknown;
    try {
        assertDawnHost("win32", "C:\\tools\\bun.exe");
    } catch (error) {
        failure = error;
    }

    expect(failure).toBeInstanceOf(UnsupportedError);
    expect((failure as Error).message).toContain("dawn.node imports N-API directly from node.exe");
    expect((failure as Error).message).toContain("use Node, or Bun run as a copy named node.exe");
    expect(() => assertDawnHost("win32", "C:\\tools\\node.exe")).not.toThrow();
    expect(() => assertDawnHost("win32", "C:\\tools\\NODE.EXE")).not.toThrow();
    expect(() => assertDawnHost("darwin", "/tools/bun")).not.toThrow();
});
