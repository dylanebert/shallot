import { Glob } from "bun";
import { dirname, relative, resolve, sep } from "path";

// packages/shallot-tui is engine-agnostic by design (shallot-tui spec S2: cells to bytes, no GPU,
// no bun-webgpu, testable on any seat with no adapter). This is the mechanism that carries that
// packaging decision — a check, not a repo location or a convention anybody can quietly cross
// (shallot-tui.md "Packaging"). It is a stricter property than `check-boundary.ts`'s consumer
// boundary: shallot-tui may not import the engine at all, published surface or not, so this is
// its own script rather than a mode of that one.
//
// Two ways `packages/shallot-tui/src/**` could reach the engine:
//   1. a relative import that climbs out of `packages/shallot-tui/` — the same escape
//      `check-boundary.ts` guards for every consumer;
//   2. a bare `@dylanebert/shallot` import, published subpath or not — unlike a consumer, this
//      package must carry zero engine dependency, so even the published surface is a violation.
//
// `--root <dir>` points the check at an alternate tree (fixture-driven proof; check-imports.ts
// and check-boundary.ts carry the same flag for the same reason).

const repoRoot = resolve(import.meta.dir, "..");
const PKG = "@dylanebert/shallot";

const importRe = /(?:from|import)\s+["']([^"']+)["']/g;

type Violation = { file: string; line: number; import: string; reason: string };

async function scan(root: string): Promise<Violation[]> {
    const violations: Violation[] = [];
    const srcDir = resolve(root, "src");
    const glob = new Glob("**/*.ts");
    for await (const path of glob.scan({ cwd: srcDir })) {
        const full = resolve(srcDir, path);
        const lines = (await Bun.file(full).text()).split("\n");
        for (let i = 0; i < lines.length; i++) {
            for (const match of lines[i].matchAll(importRe)) {
                const spec = match[1];
                const at = { file: relative(repoRoot, full), line: i + 1, import: spec };
                if (spec.startsWith(".")) {
                    const resolved = resolve(dirname(full), spec);
                    if (resolved === root || resolved.startsWith(root + sep)) continue;
                    violations.push({
                        ...at,
                        reason: `escapes packages/shallot-tui → ${relative(repoRoot, resolved)}`,
                    });
                } else if (spec === PKG || spec.startsWith(`${PKG}/`)) {
                    violations.push({
                        ...at,
                        reason: "imports the engine (@dylanebert/shallot) — shallot-tui must be engine-agnostic",
                    });
                }
            }
        }
    }
    return violations;
}

const rootArg = process.argv.indexOf("--root");
const root =
    rootArg >= 0 ? resolve(process.argv[rootArg + 1]) : resolve(repoRoot, "packages/shallot-tui");

const violations = await scan(root);

if (violations.length > 0) {
    console.error(`✗ ${violations.length} shallot-tui engine-boundary violation(s):\n`);
    for (const v of violations) {
        console.error(`  ${v.file}:${v.line}`);
        console.error(`    import "${v.import}" ${v.reason}`);
    }
    console.error(
        "\npackages/shallot-tui must import nothing from the engine — no relative escape into\n" +
            "packages/shallot/, and no @dylanebert/shallot import at any subpath, published or not.",
    );
    process.exit(1);
}

console.log("✓ shallot-tui engine boundary clean");
