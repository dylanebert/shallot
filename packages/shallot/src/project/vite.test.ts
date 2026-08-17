import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Rollup } from "vite";
import { assetSrc, discoverScenes, findPublicDirs, orphanedAssets, projectPlugin } from "./vite";

// a fresh project dir per test, cleaned up after — the shape every fs-reader / hook test needs
function projectDir(): string {
    return mkdtempSync(join(tmpdir(), "shallot-vite-test-"));
}

// minimal bundle builders — the pure prune reads only type / fileName / code|source
const chunk = (fileName: string, code: string) =>
    ({ type: "chunk", fileName, code }) as Rollup.OutputChunk;
const asset = (fileName: string, source: string | Uint8Array) =>
    ({ type: "asset", fileName, source }) as Rollup.OutputAsset;
const bundle = (...files: (Rollup.OutputChunk | Rollup.OutputAsset)[]) =>
    Object.fromEntries(files.map((f) => [f.fileName, f])) as Rollup.OutputBundle;

describe("orphanedAssets", () => {
    test("drops a wasm no chunk references (the new-URL over-emit), keeps the live entry", () => {
        // orbit's shape: codec branch tree-shaken dead, yet vite emitted the wasm at transform time
        const b = bundle(
            asset("index.html", `<script src="./assets/index-AAA.js"></script>`),
            chunk("assets/index-AAA.js", `console.log("imports only Orbit")`),
            asset("assets/draco-BBB.wasm", new Uint8Array([0, 1, 2])),
        );
        expect(orphanedAssets(b)).toEqual(["assets/draco-BBB.wasm"]);
    });

    test("keeps a codec the project actually uses — its hashed name survives in a live chunk", () => {
        const b = bundle(
            chunk("assets/index-AAA.js", `new URL("./assets/draco-BBB.wasm", import.meta.url)`),
            asset("assets/draco-BBB.wasm", new Uint8Array([0, 1, 2])),
        );
        expect(orphanedAssets(b)).toEqual([]);
    });

    test("an asset referenced only by another asset survives via the fixpoint (css → font)", () => {
        // the case a single chunk-only scan gets wrong: the font is reached only through the css
        const b = bundle(
            asset("index.html", `<link rel="stylesheet" href="./assets/style-CCC.css">`),
            asset("assets/style-CCC.css", `@font-face{src:url(./font-DDD.woff2)}`),
            asset("assets/font-DDD.woff2", new Uint8Array([0])),
            asset("assets/orphan-EEE.wasm", new Uint8Array([9])),
        );
        expect(orphanedAssets(b)).toEqual(["assets/orphan-EEE.wasm"]);
    });
});

// the live asset-swap path → cache-key mapping. A changed model file's path maps to its public-relative
// cache src (the watcher's full-reload trigger); everything else falls through to the scene/manifest watch.
describe("assetSrc", () => {
    const pub = "/proj/public";

    test("maps a .glb under the public dir to its public-relative cache src", () => {
        expect(assetSrc("/proj/public/box.glb", [pub])).toBe("box.glb");
    });

    test("maps a nested .gltf to a /-joined src — the scene mesh-name + readBinary use", () => {
        expect(assetSrc("/proj/public/sponza/Sponza.gltf", [pub])).toBe("sponza/Sponza.gltf");
    });

    test("ignores a non-model file — the scene/manifest + sidecar boundary", () => {
        // a .scene / shallot.json ride their own watch; a sidecar (.bin) or separate texture re-decodes
        // through its container, which a re-export rewrites — the deliberate scope boundary for this stage
        expect(assetSrc("/proj/public/sponza/Sponza.bin", [pub])).toBeNull();
        expect(assetSrc("/proj/public/sponza/wall.png", [pub])).toBeNull();
        expect(assetSrc("/proj/scenes/a.scene", [pub])).toBeNull();
    });

    test("ignores a model outside every public dir (not a fetchable cache src)", () => {
        expect(assetSrc("/proj/src/box.glb", [pub])).toBeNull();
        expect(assetSrc("/other/box.glb", [pub])).toBeNull();
    });

    test("resolves against the matching dir when several public dirs are given", () => {
        const dirs = ["/proj/public", "/shared/public"];
        expect(assetSrc("/shared/public/tree.glb", dirs)).toBe("tree.glb");
        expect(assetSrc("/proj/public/box.glb", dirs)).toBe("box.glb");
    });

    test("matches the extension case-insensitively", () => {
        expect(assetSrc("/proj/public/Model.GLB", [pub])).toBe("Model.GLB");
    });
});

describe("discoverScenes", () => {
    test("finds nested .scene files, sorted, relative to dir", () => {
        const dir = projectDir();
        mkdirSync(join(dir, "levels"));
        writeFileSync(join(dir, "b.scene"), "");
        writeFileSync(join(dir, "levels", "a.scene"), "");
        expect(discoverScenes(dir)).toEqual(["b.scene", join("levels", "a.scene")]);
    });

    test("skips node_modules and dist", () => {
        const dir = projectDir();
        mkdirSync(join(dir, "node_modules"));
        writeFileSync(join(dir, "node_modules", "vendored.scene"), "");
        mkdirSync(join(dir, "dist"));
        writeFileSync(join(dir, "dist", "built.scene"), "");
        expect(discoverScenes(dir)).toEqual([]);
    });

    test("returns [] for a dir that doesn't exist", () => {
        expect(discoverScenes(join(projectDir(), "missing"))).toEqual([]);
    });

    // discoverScenes used to wrap the WHOLE recursive walk in one try/catch: an unreadable subtree
    // (a broken symlink here — statSync follows it and throws ENOENT) propagated out of every
    // enclosing `walk()` frame with no per-directory catch, silently truncating every sibling not yet
    // visited at EVERY ancestor level, not just the bad subtree. These three entries are named so this
    // filesystem's `readdirSync` order (deterministic per name-set here, verified empirically — not
    // alphabetical) visits the broken directory FIRST at the top level, and its own broken symlink
    // before its scene sibling — so the pre-fix code returns [] entirely, not a partial list.
    test("an unreadable entry doesn't truncate scanning its siblings or its ancestors' siblings", () => {
        const dir = projectDir();
        writeFileSync(join(dir, "1_before.scene"), "");
        const bad = join(dir, "2_baddir");
        mkdirSync(bad);
        symlinkSync(join(bad, "nonexistent-target"), join(bad, "broken"));
        writeFileSync(join(bad, "3_after.scene"), "");
        writeFileSync(join(dir, "4_after.scene"), "");

        expect(discoverScenes(dir)).toEqual([
            "1_before.scene",
            join("2_baddir", "3_after.scene"),
            "4_after.scene",
        ]);
    });
});

describe("findPublicDirs", () => {
    test("returns [] when neither the project nor its parent has a public dir", () => {
        expect(findPublicDirs(projectDir())).toEqual([]);
    });

    test("includes the project's own public dir", () => {
        const dir = projectDir();
        mkdirSync(join(dir, "public"));
        expect(findPublicDirs(dir)).toEqual([join(dir, "public")]);
    });

    test("includes a shared parent public dir too, own first", () => {
        const parent = projectDir();
        const dir = join(parent, "proj");
        mkdirSync(dir);
        mkdirSync(join(dir, "public"));
        mkdirSync(join(parent, "public"));
        expect(findPublicDirs(dir)).toEqual([join(dir, "public"), join(parent, "public")]);
    });
});

// projectPlugin's hooks — a plain object, each callable against a stub `this` / server (Locked decision:
// "every seam this unit needs already exists as a parameter"). The stubs below are duck-typed to the
// members each hook actually touches, never the full Vite/Rollup types, and every hook property is cast
// at the call site: the plugin object assigns each as a plain function, but Vite's `Plugin` type widens
// the property to `ObjectHook<fn> | { handler: fn }`, which has no call signature to invoke directly.
describe("projectPlugin", () => {
    function stubServer() {
        const middlewareFns: Array<
            (req: { url?: string }, res: ReturnType<typeof stubRes>, next: () => void) => void
        > = [];
        const sent: unknown[] = [];
        const invalidated: unknown[] = [];
        let moduleById: unknown;
        const server = {
            middlewares: { use: (fn: unknown) => middlewareFns.push(fn as never) },
            watcher: { add: () => {}, on: () => {} },
            ws: { send: (msg: unknown) => sent.push(msg) },
            moduleGraph: {
                getModuleById: () => moduleById,
                invalidateModule: (mod: unknown) => invalidated.push(mod),
            },
        };
        return {
            server,
            middlewareFns,
            sent,
            invalidated,
            setModuleById: (m: unknown) => {
                moduleById = m;
            },
        };
    }

    function stubRes() {
        return {
            headers: {} as Record<string, string>,
            ended: undefined as unknown,
            setHeader(name: string, value: string) {
                this.headers[name] = value;
            },
            end(data: unknown) {
                this.ended = data;
            },
        };
    }

    describe("resolveId", () => {
        test("resolves the virtual module id to its internal id", async () => {
            const plugin = projectPlugin();
            const hook = plugin.resolveId as unknown as (
                this: unknown,
                id: string,
                importer?: string,
            ) => Promise<string | undefined>;
            await expect(hook.call({}, "virtual:project")).resolves.toBe("\0virtual:project");
        });

        test("resolves a project-relative import against the project dir via the stub this.resolve", async () => {
            const dir = projectDir();
            const plugin = projectPlugin(dir);
            const hook = plugin.resolveId as unknown as (
                this: {
                    resolve: (
                        id: string,
                        importer: string,
                        opts: { skipSelf: boolean },
                    ) => Promise<{ id: string } | null>;
                },
                id: string,
                importer?: string,
            ) => Promise<{ id: string } | undefined>;
            const calls: unknown[] = [];
            const stubThis = {
                resolve: async (id: string, importer: string, opts: { skipSelf: boolean }) => {
                    calls.push({ id, importer, opts });
                    return { id: `/resolved/${id}` };
                },
            };
            const resolved = await hook.call(stubThis, "orrstead/core/grid", "\0virtual:project");
            expect(resolved).toEqual({ id: "/resolved/orrstead/core/grid" });
            expect(calls).toEqual([
                {
                    id: "orrstead/core/grid",
                    importer: join(dir, "__project__.js"),
                    opts: { skipSelf: true },
                },
            ]);
        });

        test("resolves nothing for an unrelated importer, or without a projectDir", async () => {
            const plugin = projectPlugin(); // no projectDir
            const hook = plugin.resolveId as unknown as (
                this: unknown,
                id: string,
                importer?: string,
            ) => Promise<string | undefined>;
            await expect(hook.call({}, "some/pkg", "\0virtual:project")).resolves.toBeUndefined();
        });
    });

    describe("load", () => {
        function loadHook(plugin: ReturnType<typeof projectPlugin>) {
            return plugin.load as unknown as (this: unknown, id: string) => string | undefined;
        }

        test("generates an empty-manifest module when the plugin carries no projectDir", () => {
            const plugin = projectPlugin();
            const out = loadHook(plugin).call({}, "\0virtual:project");
            expect(out).toContain("const manifest = {};");
            expect(out).toContain("const scenes = [];");
        });

        test("threads readManifest + discoverScenes into the generated module for a real project", () => {
            const dir = projectDir();
            writeFileSync(
                join(dir, "shallot.json"),
                JSON.stringify({ plugins: { Render: false } }),
            );
            writeFileSync(join(dir, "main.scene"), "");
            const plugin = projectPlugin(dir);
            const out = loadHook(plugin).call({}, "\0virtual:project");
            expect(out).toContain('"main.scene"');
            expect(out).toContain('"Render":false');
        });

        test("returns nothing for an id other than the virtual module's", () => {
            const plugin = projectPlugin(projectDir());
            expect(loadHook(plugin).call({}, "some/other/module")).toBeUndefined();
        });
    });

    describe("configureServer's middleware", () => {
        function install(dir?: string) {
            const plugin = projectPlugin(dir);
            const stub = stubServer();
            const configureServer = plugin.configureServer as unknown as (server: unknown) => void;
            configureServer(stub.server);
            return stub;
        }

        test("registers no middleware when the project has no public dir", () => {
            const { middlewareFns } = install(projectDir()); // no public/ subdir
            expect(middlewareFns).toHaveLength(0);
        });

        test("serves a project asset with the right MIME and a no-store cache header", () => {
            const dir = projectDir();
            mkdirSync(join(dir, "public"));
            writeFileSync(join(dir, "public", "icon.svg"), "<svg/>");
            const { middlewareFns } = install(dir);
            expect(middlewareFns).toHaveLength(1);

            const res = stubRes();
            let calledNext = false;
            middlewareFns[0]({ url: "/icon.svg" }, res, () => {
                calledNext = true;
            });

            expect(res.headers["Content-Type"]).toBe("image/svg+xml");
            expect(res.headers["Cache-Control"]).toBe("no-store");
            expect(res.ended?.toString()).toBe("<svg/>");
            expect(calledNext).toBe(false);
        });

        test("falls through to next() for a request with no matching asset", () => {
            const dir = projectDir();
            mkdirSync(join(dir, "public"));
            const { middlewareFns } = install(dir);

            const res = stubRes();
            let calledNext = false;
            middlewareFns[0]({ url: "/missing.svg" }, res, () => {
                calledNext = true;
            });
            expect(calledNext).toBe(true);
            expect(res.ended).toBeUndefined();
        });

        test("falls through to next() when req.url is absent", () => {
            const dir = projectDir();
            mkdirSync(join(dir, "public"));
            writeFileSync(join(dir, "public", "icon.svg"), "<svg/>");
            const { middlewareFns } = install(dir);

            const res = stubRes();
            let calledNext = false;
            middlewareFns[0]({}, res, () => {
                calledNext = true;
            });
            expect(calledNext).toBe(true);
        });
    });

    describe("handleHotUpdate", () => {
        function handleHotUpdateHook(plugin: ReturnType<typeof projectPlugin>) {
            return plugin.handleHotUpdate as unknown as (ctx: {
                file: string;
            }) => unknown[] | undefined;
        }
        function configureServerHook(plugin: ReturnType<typeof projectPlugin>) {
            return plugin.configureServer as unknown as (server: unknown) => void;
        }

        test("invalidates the virtual module and signals a reload on a .scene change, swallowing default HMR", () => {
            const dir = projectDir();
            const plugin = projectPlugin(dir);
            const stub = stubServer();
            configureServerHook(plugin)(stub.server); // sets the closure's live viteServer

            const mod = { id: "\0virtual:project" };
            stub.setModuleById(mod);
            const result = handleHotUpdateHook(plugin)({ file: join(dir, "main.scene") });

            expect(result).toEqual([]);
            expect(stub.invalidated).toEqual([mod]);
            expect(stub.sent).toEqual([{ type: "full-reload" }]);
        });

        test("invalidates on a shallot.json change too", () => {
            const dir = projectDir();
            const plugin = projectPlugin(dir);
            const stub = stubServer();
            configureServerHook(plugin)(stub.server);

            const result = handleHotUpdateHook(plugin)({ file: join(dir, "shallot.json") });
            expect(result).toEqual([]);
            expect(stub.sent).toEqual([{ type: "full-reload" }]);
        });

        test("falls through to default HMR for an unrelated file", () => {
            const dir = projectDir();
            const plugin = projectPlugin(dir);
            const stub = stubServer();
            configureServerHook(plugin)(stub.server);

            const result = handleHotUpdateHook(plugin)({ file: join(dir, "src", "Local.ts") });

            expect(result).toBeUndefined();
            expect(stub.sent).toEqual([]);
            expect(stub.invalidated).toEqual([]);
        });

        test("does nothing without a live server, or without a projectDir", () => {
            // configureServer never ran — the closure's viteServer is still unset
            const notConfigured = projectPlugin(projectDir());
            expect(
                handleHotUpdateHook(notConfigured)({ file: "/whatever/main.scene" }),
            ).toBeUndefined();

            const noProjectDir = projectPlugin();
            const stub = stubServer();
            configureServerHook(noProjectDir)(stub.server);
            expect(
                handleHotUpdateHook(noProjectDir)({ file: "/whatever/main.scene" }),
            ).toBeUndefined();
        });
    });

    describe("generateBundle", () => {
        function generateBundleHook(plugin: ReturnType<typeof projectPlugin>) {
            return plugin.generateBundle as unknown as (
                this: { info: (msg: string) => void },
                options: unknown,
                bundle: Rollup.OutputBundle,
            ) => void;
        }

        test("drops an orphaned asset and reports its pruned byte count", () => {
            const plugin = projectPlugin();
            // deliberately NOT a KB multiple: at 2048 every rounding mode agrees and the report's
            // truncating `| 0` goes unpinned. 3000 B truncates to 2KB and rounds up to 3KB.
            const source = "x".repeat(3000);
            const outputBundle = bundle(
                chunk("assets/index-AAA.js", `console.log("no reference")`),
                asset("assets/draco-BBB.wasm", source),
            );
            const infoMsgs: string[] = [];
            generateBundleHook(plugin).call({ info: (m) => infoMsgs.push(m) }, {}, outputBundle);

            expect(outputBundle["assets/draco-BBB.wasm"]).toBeUndefined();
            expect(infoMsgs).toEqual(["pruned 1 orphaned asset(s), 2KB"]);
        });

        test("leaves a fully-referenced bundle untouched and reports nothing", () => {
            const plugin = projectPlugin();
            const outputBundle = bundle(
                chunk("assets/index-AAA.js", `new URL("./assets/draco-BBB.wasm", import.meta.url)`),
                asset("assets/draco-BBB.wasm", "x"),
            );
            const before = { ...outputBundle };
            const infoMsgs: string[] = [];
            generateBundleHook(plugin).call({ info: (m) => infoMsgs.push(m) }, {}, outputBundle);

            expect(outputBundle).toEqual(before);
            expect(infoMsgs).toEqual([]);
        });
    });
});
