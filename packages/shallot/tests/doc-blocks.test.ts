/**
 * Source hygiene over the real `src/` tree: a `/**` block that never closes swallows everything up to
 * the next `*​/`, including the JSDoc of the declaration that follows it. That declaration then ships
 * with no doc — and per `exports.md`, no JSDoc means hidden from the reference — while `bun check`,
 * `tsc`, and every unit test stay green, because an over-long comment is still valid TypeScript.
 *
 * Red-proven against `extras/gltf/image.ts`, where the compressed-path section header ran on into
 * `allocCompressed`'s contract.
 */
import { expect, test } from "bun:test";
import { resolve } from "node:path";

const SRC = resolve(import.meta.dir, "..", "src");

/** `/**` openers that appear while already inside a block comment, as `file:line` */
function nested(src: string, file: string): string[] {
    const hits: string[] = [];
    let inBlock = false;
    const lines = src.split("\n");
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i] as string;
        for (let c = 0; c < line.length - 1; c++) {
            const two = line.slice(c, c + 2);
            if (!inBlock && two === "//") break; // a line comment can quote anything
            if (!inBlock && two === "/*") {
                inBlock = true;
                c++;
            } else if (inBlock && two === "/*") {
                // a second opener with no intervening close: the first block never terminated
                hits.push(`${file}:${i + 1}`);
                c++;
            } else if (inBlock && two === "*/") {
                inBlock = false;
                c++;
            }
        }
    }
    return hits;
}

test("no source file opens a block comment inside an unclosed one", async () => {
    const offenders: string[] = [];
    let scanned = 0;
    for await (const rel of new Bun.Glob("**/*.ts").scan({ cwd: SRC })) {
        scanned++;
        offenders.push(...nested(await Bun.file(resolve(SRC, rel)).text(), rel));
    }
    expect(scanned).toBeGreaterThan(100); // the walk reached the real tree, not an empty glob
    expect(offenders).toEqual([]);
});

test("the scan reports a nested opener and tolerates the shapes that only look like one", () => {
    expect(nested("/** a\n/** b */\n", "x.ts")).toEqual(["x.ts:2"]);
    expect(nested("/** a */\n/** b */\n", "x.ts")).toEqual([]);
    // a divide followed by a dereference, and an opener quoted in a line comment, are not blocks
    expect(nested("const n = a / *p;\n// /** not a block\n", "x.ts")).toEqual([]);
});
