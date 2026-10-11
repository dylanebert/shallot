import { expect, test } from "bun:test";
import { loadDawn } from "./dawn-host";
import { UnsupportedError } from "./gpu";

test("the Windows Dawn loader refuses unsupported hosts before import and accepts node.exe", async () => {
    let imports = 0;
    const load = async () => {
        imports++;
        return { dawn: true };
    };

    await expect(loadDawn("win32", "C:\\tools\\bun.exe", load)).rejects.toBeInstanceOf(
        UnsupportedError,
    );
    await expect(loadDawn("win32", "C:\\tools\\bun.exe", load)).rejects.toThrow(
        "dawn.node imports N-API directly from node.exe",
    );
    expect(imports).toBe(0);

    await expect(loadDawn("win32", "C:\\tools\\NODE.EXE", load)).resolves.toEqual({
        dawn: true,
    });
    expect(imports).toBe(1);
});
