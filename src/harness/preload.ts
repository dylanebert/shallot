import { resolve } from "node:path";
import { plugin } from "bun";
import typegpu from "unplugin-typegpu/bun";

// Keep the Bun hook on TypeScript; its default also intercepts JavaScript dependencies.
plugin(typegpu({ include: /\.(?:[cm]?ts|tsx)$/ }));

// Bun test preload: every `.test.ts` and `.oracle.ts` file must declare its checks
// through `check()`. A file that reaches for `bun:test` directly, or that declares nothing at all,
// refuses at load rather than running undeclared.

const CHECK_MODULE = resolve(import.meta.dir, "check.ts");
const SUFFIX = /\.(test|oracle)\.ts$/;
const BUN_TEST_IMPORT = /import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']bun:test["']/g;
const REGISTRARS = new Set(["test", "it", "describe"]);

plugin({
    name: "surface-declaration",
    setup(build) {
        build.onLoad({ filter: SUFFIX }, async (args) => {
            const source = await Bun.file(args.path).text();
            for (const match of source.matchAll(BUN_TEST_IMPORT)) {
                const names = match[1]
                    .split(",")
                    .map((part) =>
                        part
                            .trim()
                            .split(/\s+as\s+/)[0]
                            .trim(),
                    )
                    .filter((name) => REGISTRARS.has(name));
                if (names.length > 0) {
                    throw new Error(
                        `undeclared check file: ${args.path} imports ${names.join(", ")} from bun:test; register through check() from @dylanebert/shallot/harness/check`,
                    );
                }
            }
            const module = JSON.stringify(CHECK_MODULE);
            const path = JSON.stringify(args.path);
            // Keep the first source line on its original line. Bun reports locations from the transformed
            // module, so a newline for either injected statement would move every user statement down.
            const header = `import { beginFile as __beginFile, assertDeclared as __assertDeclared } from ${module}; __beginFile(${path}); `;
            return { contents: `${header}${source}\n__assertDeclared(${path});\n`, loader: "ts" };
        });
    },
});
