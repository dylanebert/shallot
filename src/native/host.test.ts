import { expect, test } from "bun:test";
import { nativeHostDiagnostic } from "./index";

for (const host of ["darwin", "linux", "win32", "freebsd"] as const) {
    for (const [target, requiredHost, requirement] of [
        ["linux", "linux", "Linux"],
        ["mac", "darwin", "macOS"],
    ] as const) {
        if (host === requiredHost) continue;
        test(`${target} refuses ${host} before building or launching`, () => {
            expect(nativeHostDiagnostic(target, host)).toBe(
                `Target ${target} is not supported on host ${host}; a ${target} target needs a ${requirement} host.`,
            );
        });
    }
    test(`windows keeps its native, xwin and WSL paths on ${host}`, () => {
        expect(nativeHostDiagnostic("windows", host)).toBeNull();
    });
}

for (const [target, host] of [
    ["linux", "linux"],
    ["mac", "darwin"],
] as const) {
    test(`${target} supports its ${host} host`, () => {
        expect(nativeHostDiagnostic(target, host)).toBeNull();
    });
}
