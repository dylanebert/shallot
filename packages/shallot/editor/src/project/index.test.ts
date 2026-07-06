import { afterEach, describe, expect, test } from "bun:test";
import { classifyExternal, matchScene, Persist, pickSaveMode } from "./index";

describe("classifyExternal", () => {
    const Scene = "scenes/world.scene";

    test("the active scene reloads when clean, conflicts when there are unsaved edits", () => {
        const base = { path: Scene, manifest: false, scene: Scene };
        expect(classifyExternal({ ...base, unsaved: false })).toBe("reload");
        expect(classifyExternal({ ...base, unsaved: true })).toBe("conflict");
    });

    test("the manifest follows the same rule regardless of which scene is open", () => {
        const base = { path: "shallot.json", manifest: true, scene: Scene };
        expect(classifyExternal({ ...base, unsaved: false })).toBe("reload");
        expect(classifyExternal({ ...base, unsaved: true })).toBe("conflict");
    });

    test("a different scene is ignored — it can't touch the active content, so never a reload or nuke", () => {
        const other = { path: "scenes/other.scene", manifest: false, scene: Scene };
        expect(classifyExternal({ ...other, unsaved: false })).toBe("ignore");
        expect(classifyExternal({ ...other, unsaved: true })).toBe("ignore");
    });

    test("with no active scene, a scene change is irrelevant but the manifest still matters", () => {
        expect(
            classifyExternal({
                path: "scenes/x.scene",
                manifest: false,
                scene: null,
                unsaved: false,
            }),
        ).toBe("ignore");
        expect(
            classifyExternal({ path: "shallot.json", manifest: true, scene: null, unsaved: false }),
        ).toBe("reload");
    });
});

describe("matchScene", () => {
    test("matches a public-relative config scene against a project-relative discovery", () => {
        // config.scene drops the public dir the discovered path carries — the frame the bug lived in
        expect(matchScene("scenes/cornell.scene", ["public/scenes/cornell.scene"])).toBe(
            "public/scenes/cornell.scene",
        );
        expect(matchScene("scenes/x.scene", ["scenes/x.scene"])).toBe("scenes/x.scene");
    });

    test("picks the first scene of an array config", () => {
        expect(
            matchScene(
                ["scenes/a.scene", "scenes/b.scene"],
                ["public/scenes/a.scene", "public/scenes/b.scene"],
            ),
        ).toBe("public/scenes/a.scene");
    });

    test("returns null when nothing matches or there is nothing to match", () => {
        expect(matchScene("scenes/missing.scene", ["public/scenes/x.scene"])).toBeNull();
        expect(matchScene(undefined, ["public/scenes/x.scene"])).toBeNull();
        expect(matchScene([], ["public/scenes/x.scene"])).toBeNull();
    });

    test("the trailing-segment boundary rejects a false suffix", () => {
        // "scene.scene" must not match "my-scene.scene" — only a full path segment counts
        expect(matchScene("scene.scene", ["public/my-scene.scene"])).toBeNull();
    });
});

describe("pickSaveMode", () => {
    test("only ?save=off opens ephemeral; everything else is file", () => {
        expect(pickSaveMode("?save=off")).toBe("ephemeral");
        expect(pickSaveMode("?foo=1&save=off")).toBe("ephemeral");
        expect(pickSaveMode("")).toBe("file");
        expect(pickSaveMode("?save=on")).toBe("file");
        expect(pickSaveMode("?save")).toBe("file");
        expect(pickSaveMode("?other=off")).toBe("file");
    });
});

describe("Persist.save", () => {
    const realFetch = globalThis.fetch;
    afterEach(() => {
        globalThis.fetch = realFetch;
        Persist.mode = "file";
    });

    test("ephemeral mode never touches the network", async () => {
        let called = false;
        globalThis.fetch = (async () => {
            called = true;
            return new Response("", { status: 200 });
        }) as unknown as typeof fetch;

        Persist.mode = "ephemeral";
        await Persist.save("/proj", "scene.scene", "<scene/>");
        expect(called).toBe(false);
    });

    test("file mode posts to the scene endpoint", async () => {
        let url = "";
        let method = "";
        globalThis.fetch = (async (input: string, init?: RequestInit) => {
            url = String(input);
            method = init?.method ?? "GET";
            return new Response("", { status: 200 });
        }) as unknown as typeof fetch;

        Persist.mode = "file";
        await Persist.save("/proj", "scene.scene", "<scene/>");
        expect(method).toBe("POST");
        expect(url).toContain("/__api/scene");
    });
});

describe("Persist.saveManifest", () => {
    const realFetch = globalThis.fetch;
    afterEach(() => {
        globalThis.fetch = realFetch;
        Persist.mode = "file";
    });

    test("ephemeral mode never writes the manifest — capture/engine-dev sessions stay clean", () => {
        let called = false;
        globalThis.fetch = (async () => {
            called = true;
            return new Response("", { status: 200 });
        }) as unknown as typeof fetch;

        Persist.mode = "ephemeral";
        return Persist.saveManifest("/proj", "{}").then(() => expect(called).toBe(false));
    });

    test("file mode posts to the manifest endpoint", async () => {
        let url = "";
        let method = "";
        globalThis.fetch = (async (input: string, init?: RequestInit) => {
            url = String(input);
            method = init?.method ?? "GET";
            return new Response("", { status: 200 });
        }) as unknown as typeof fetch;

        Persist.mode = "file";
        await Persist.saveManifest("/proj", '{ "plugins": {} }');
        expect(method).toBe("POST");
        expect(url).toContain("/__api/manifest");
    });
});
