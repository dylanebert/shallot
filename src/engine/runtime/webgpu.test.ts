import { expect, test } from "bun:test";
import { loadDawn } from "./dawn-host";
import { UnsupportedError } from "./gpu";

test("a missing optional Dawn peer names its install command and preserves the import cause", async () => {
    const cause = Object.assign(new Error("Cannot find package 'webgpu'"), {
        code: "ERR_MODULE_NOT_FOUND",
    });
    const load = async () => {
        throw cause;
    };

    let refusal: unknown;
    try {
        await loadDawn("darwin", "/usr/local/bin/bun", load);
    } catch (error) {
        refusal = error;
    }

    expect(refusal).toBeInstanceOf(UnsupportedError);
    expect(refusal).toMatchObject({
        message: expect.stringContaining("bun add webgpu"),
        cause,
    });
});

test("other Dawn loader failures keep their original error", async () => {
    const failure = Object.assign(new Error("Dawn native addon failed to load"), {
        code: "ERR_DLOPEN_FAILED",
    });
    await expect(loadDawn("darwin", "/usr/local/bin/bun", async () => Promise.reject(failure))).rejects.toBe(
        failure,
    );
});

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
