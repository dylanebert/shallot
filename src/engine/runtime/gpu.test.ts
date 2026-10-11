import { expect, test } from "bun:test";
import { checkPipelineStorageBindings, countPipelineStorageBindings } from "./gpu";

const storage = (visibility: number, type: "storage" | "read-only-storage" = "storage") =>
    ({ visibility, buffer: { type } }) as GPUBindGroupLayoutEntry;
const uniform = (visibility: number) =>
    ({ visibility, buffer: { type: "uniform" } }) as GPUBindGroupLayoutEntry;

test("pipeline storage counts sum bound groups by visibility and ignore non-storage bindings", () => {
    expect(
        countPipelineStorageBindings([
            [storage(1 | 2), storage(1), uniform(2), storage(4, "read-only-storage")],
            [storage(1), storage(4), storage(2 | 4, "read-only-storage")],
        ]),
    ).toEqual({ vertex: 3, fragment: 2, compute: 3 });
});

test("pipeline storage guard accepts the granted limit and names stage overages", () => {
    expect(() =>
        checkPipelineStorageBindings(
            "StorageOwner",
            "light-compact",
            { vertex: 8, fragment: 0, compute: 8 },
            8,
        ),
    ).not.toThrow();
    expect(() =>
        checkPipelineStorageBindings(
            "StorageOwner",
            "light-compact",
            { vertex: 0, fragment: 0, compute: 10 },
            8,
        ),
    ).toThrow(
        'Plugin "StorageOwner" pipeline "light-compact" uses 10 compute storage-buffer bindings; maxStorageBuffersPerShaderStage is 8',
    );
});
