// Set BOX3D_NATIVE_EVIDENCE=1 to also report native NEON/SSE2 differences. The returned authority
// remains SSE2; NEON results never participate in kernel comparisons.
import { nativeBinary } from "./native";

export function nativeSseOutput(sourceName: string, input: string, labels?: string[]): string {
    const sse = Bun.spawnSync(["arch", "-x86_64", nativeBinary(sourceName, "x86_64")], {
        stdin: Buffer.from(input),
    });
    if (sse.exitCode !== 0) throw new Error(`SSE2 ${sourceName}: ${sse.stderr.toString()}`);
    const output = sse.stdout.toString();
    if (process.env.BOX3D_NATIVE_EVIDENCE === "1") {
        const neon = Bun.spawnSync(["arch", "-arm64", nativeBinary(sourceName, "arm64")], {
            stdin: Buffer.from(input),
        });
        if (neon.exitCode !== 0) throw new Error(`NEON ${sourceName}: ${neon.stderr.toString()}`);
        const a = output.trim().split("\n");
        const b = neon.stdout.toString().trim().split("\n");
        labels ??= a.map((line) => line.split(" ").slice(0, 2).join(":"));
        if (a.length !== labels.length || b.length !== a.length)
            throw new Error(`${sourceName}: native evidence row count differs`);
        const inputs = input.trim().split("\n");
        let differences = 0;
        for (let i = 0; i < a.length; i++) {
            if (a[i] === b[i]) continue;
            differences++;
            console.info(
                JSON.stringify({
                    nativeArchitectures: sourceName,
                    case: labels[i],
                    input: inputs[i],
                    sse2: a[i],
                    neon: b[i],
                }),
            );
        }
        console.info(
            `native architectures: ${sourceName}; ${a.length} cases; ${differences} SSE2/NEON differences (not kernel reds)`,
        );
    }
    return output;
}
