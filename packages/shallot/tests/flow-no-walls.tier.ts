import { expect, test } from "bun:test";
import { verify } from "../../../scripts/verify";
import { requireDisplay } from "./flows/harness";

// By-path headed tier for the extension escape, moved here from `examples/flows/no-walls` because the
// claim is about this package's device adoption and raw-WGSL seam, not about examples. The fixture app
// under `flows/no-walls` is unchanged: an adopted external GPUDevice creates a raw buffer, TypeGPU wraps
// it with the draw schema, raw WGSL resolved through `tgpu.resolve` writes it, Draws consumes the typed
// handle, and Mirror reads the same allocation back. End-to-end on purpose — isolated API calls would
// not prove the record reached a rendered product.
//
//     bun test ./packages/shallot/tests/flow-no-walls.tier.ts
//
// Headed, on the seat's own display, alone. No display refuses with a named reason.

const CHECKS = [
    "external GPUDevice adopted",
    "raw GPUBuffer wrapped with schema",
    "raw WGSL resolved through TypeGPU",
    "resolved raw dispatch wrote the typed draw record",
    "Draws retained the typed indirect handle",
];

test("no-walls: an adopted device and raw WGSL reach a rendered draw", async () => {
    requireDisplay();
    const result = await verify("packages/shallot/tests/flows/no-walls", ["--timeout", "60000"]);
    expect(result?.pass).toBe(true);
    expect(result?.rendered).toBe(true);
    expect(result?.verdict?.ok).toBe(true);
    for (const name of CHECKS)
        expect(result?.verdict?.checks?.find((check) => check.name === name)?.ok).toBe(true);
}, 180_000);
