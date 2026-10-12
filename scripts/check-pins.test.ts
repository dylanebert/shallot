import { expect, test } from "bun:test";
import { resolve } from "path";

test("documented dependency and Bun pins match their manifests", async () => {
    const root = resolve(import.meta.dir, "..");
    const packageJson = await Bun.file(resolve(root, "package.json")).json();
    expect(packageJson.packageManager).toBe("bun@1.4.3");
    expect(packageJson.engines.bun).toBe(">=1.4.3");
    expect((await Bun.file(resolve(root, ".bun-version")).text()).trim()).toBe("1.4.3");

    // The doc set is what Git tracks, not what the filesystem holds. A checkout may contain generated
    // or third-party Markdown that is not part of the repository's install instructions.
    // One spawn lists both sets: a process start is most of this check's time on Windows.
    const tracked = Bun.spawnSync(
        ["git", "ls-files", "-z", "*.md", "package.json", "**/package.json"],
        { cwd: root },
    );
    if (!tracked.success)
        throw new Error(
            "`git ls-files` failed — the pin scan needs a git checkout to scope its docs and manifests.",
        );
    const trackedFiles = tracked.stdout.toString().split("\0").filter(Boolean);
    const docs = trackedFiles.filter((file) => file.endsWith(".md"));
    if (docs.length === 0)
        throw new Error(
            "`git ls-files '*.md'` matched nothing — the pin scan would be vacuously green.",
        );

    // These are the package ranges that installation commands and workspace manifests must agree with.
    const PinSources: Record<string, { manifest: string; field: string }> = {
        "@types/bun": { manifest: "package.json", field: "devDependencies" },
        typegpu: { manifest: "package.json", field: "peerDependencies" },
        webgpu: { manifest: "package.json", field: "peerDependencies" },
        "unplugin-typegpu": { manifest: "package.json", field: "dependencies" },
        typescript: { manifest: "package.json", field: "devDependencies" },
    };

    const declared: Record<string, string> = {};
    for (const [name, { manifest, field }] of Object.entries(PinSources)) {
        const json = await Bun.file(resolve(root, manifest)).json();
        const range = json[field]?.[name];
        if (typeof range !== "string")
            throw new Error(`${manifest} declares no ${field}.${name} — update PIN_SOURCES.`);
        declared[name] = range;
    }

    // Match only tracked package names in fenced `bun add` commands. A bare package name has no range
    // to disagree with, and a package name embedded in a longer name is not this pin site.
    const PinRe = new RegExp(`(?<![-\\w])(${Object.keys(declared).join("|")})@(\\S+)`, "g");

    type PinDrift = { file: string; line: number; name: string; found: string; want: string };
    const drift: PinDrift[] = [];
    let scanned = 0;

    for (const file of docs) {
        scanned++;
        const lines = (await Bun.file(resolve(root, file)).text()).split("\n");
        let inFence = false;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.trim().startsWith("```")) {
                inFence = !inFence;
                continue;
            }
            if (!inFence || !line.trim().startsWith("bun add")) continue;
            for (const [, name, found] of line.matchAll(PinRe)) {
                if (found !== declared[name]) {
                    drift.push({ file, line: i + 1, name, found, want: declared[name] });
                }
            }
        }
    }

    // The manifests are pin sites too. Enumerate every tracked manifest so an example carrying an old
    // minor cannot silently install a second branded TypeGPU identity.
    const DepFields = [
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ] as const;

    const manifestFiles = trackedFiles.filter(
        (file) => file === "package.json" || file.endsWith("/package.json"),
    );
    if (manifestFiles.length === 0)
        throw new Error(
            "`git ls-files 'package.json' '**/package.json'` matched nothing — manifest scan is vacuous.",
        );

    let manifestPkgCount = 0;
    for (const file of manifestFiles) {
        const json = await Bun.file(resolve(root, file)).json();
        let declaredInThis = false;
        for (const field of DepFields) {
            const deps = json[field];
            if (!deps || typeof deps !== "object") continue;
            for (const [name, range] of Object.entries(deps)) {
                if (!(name in declared)) continue;
                declaredInThis = true;
                if (typeof range !== "string") continue;
                if (range !== declared[name]) {
                    drift.push({ file, line: 0, name, found: range, want: declared[name] });
                }
            }
        }
        if (declaredInThis) manifestPkgCount++;
    }

    if (manifestPkgCount === 0)
        throw new Error(
            "no tracked `package.json` declares a tracked package — manifest scan is vacuous.",
        );

    if (drift.length > 0)
        throw new Error(
            `${drift.length} pin(s) disagree with the manifests:\n${drift
                .map(
                    (pin) =>
                        `  ${pin.line ? `${pin.file}:${pin.line}` : pin.file}\n    ${pin.name}@${pin.found} — the manifest declares ${pin.want}`,
                )
                .join(
                    "\n",
                )}\n\nA documented install or workspace manifest must resolve against the shipped manifests.`,
        );
    expect(scanned).toBeGreaterThan(0);
    expect(manifestPkgCount).toBeGreaterThan(0);
});
