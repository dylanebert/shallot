import { expect, test } from "bun:test";
import { createHash } from "node:crypto";

test("LUTs decode to the committed bytes without Uint8Array.fromBase64", async () => {
    const descriptor = Object.getOwnPropertyDescriptor(Uint8Array, "fromBase64");
    Object.defineProperty(Uint8Array, "fromBase64", { configurable: true, value: undefined });
    try {
        const agxPath = "./agx-lut.ts?without-fromBase64";
        const tonyPath = "./tony-lut.ts?without-fromBase64";
        const { agxLut } = await import(agxPath);
        const { tonyLut } = await import(tonyPath);
        expect(agxLut.byteLength).toBe(262144);
        expect(createHash("sha256").update(agxLut).digest("hex")).toBe(
            "30881ed23952e988d5f86324e7c3d90db33c2c7b2512d6220da497383488cd53",
        );
        expect(tonyLut.byteLength).toBe(442368);
        expect(createHash("sha256").update(tonyLut).digest("hex")).toBe(
            "6bb781fa43ac0620b04d5ca3b1656182d11dd0aad6d68be83c58b3441867886f",
        );
    } finally {
        if (descriptor) Object.defineProperty(Uint8Array, "fromBase64", descriptor);
        else Reflect.deleteProperty(Uint8Array, "fromBase64");
    }
});
