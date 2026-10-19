import { expect, setDefaultTimeout, test } from "bun:test";
import {
    copyFileSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    realpathSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "./bun-child";
import { packedInstall, spawnPackedInstall } from "./packed-install-spawn";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

const BOUND = 2000;

test("Bun children keep their host identity under a copy named node", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-bun-host-"));
    try {
        const host = join(scratch, process.platform === "win32" ? "node.exe" : "node");
        copyFileSync(process.execPath, host);
        const result = Bun.spawnSync({
            cmd: [
                host,
                "test",
                import.meta.filename,
                "-t",
                "async Bun child|sync Bun child|packed registry dependencies",
            ],
            argv0: "bun",
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(result.exitCode, `${result.stdout}${result.stderr}`).toBe(0);
    } finally {
        removeTestDirectory(scratch);
    }
});

test("a sync Bun child runs a Bun command", () => {
    const result = Bun.spawnSync({
        ...bunChild(["pm", "cache"]),
        stdout: "pipe",
        stderr: "pipe",
    });
    expect(result.exitCode, result.stderr.toString()).toBe(0);
    expect(existsSync(result.stdout.toString().trim())).toBe(true);
});

test("an async Bun child runs a Bun command", async () => {
    const child = Bun.spawn({
        ...bunChild(["pm", "cache"]),
        stdout: "pipe",
        stderr: "pipe",
    });
    const output = await new Response(child.stdout).text();
    expect(await child.exited).toBe(0);
    expect(existsSync(output.trim())).toBe(true);
});

test("a stalled install is killed inside its bound and retains the last step", () => {
    const start = performance.now();
    expect(() =>
        spawnPackedInstall(
            bunChild(["-e", 'console.log("Resolving dependencies"); setInterval(() => {}, 1000)']),
            process.cwd(),
            "fake install",
            BOUND,
        ),
    ).toThrow(
        `fake install timed out after ${BOUND} ms (last 8192 bytes per stream):\nstdout:\nResolving dependencies\n`,
    );
    expect(performance.now() - start).toBeLessThan(TIMEOUT.node / 2);
});

test("early stderr cannot displace the last stdout step in a timeout tail", () => {
    let failure = "";
    try {
        spawnPackedInstall(
            bunChild([
                "-e",
                'process.stderr.write("early stderr:" + "x".repeat(17000)); console.log("LAST STEP package-last"); setInterval(() => {}, 1000)',
            ]),
            process.cwd(),
            "fake install",
            BOUND,
        );
    } catch (error) {
        failure = String(error);
    }
    expect(failure).toContain("stdout:\nLAST STEP package-last\n");
    expect(failure).toContain("stderr:\n");
    expect(failure).not.toContain("early stderr:");
});

test("a normal install returns captured stdout and stderr untouched", () => {
    expect(
        spawnPackedInstall(
            bunChild(["-e", 'console.log("installed"); console.error("done")']),
            process.cwd(),
            "fake install",
            BOUND,
        ),
    ).toBe("installed\ndone\n");
});

type FixtureManifest = {
    name: string;
    version: string;
    dependencies?: Record<string, string>;
    peerDependencies?: Record<string, string>;
};

function offlineFixture(
    run: (fixture: { project: string; pack: (manifest: FixtureManifest) => void }) => void,
    omit = "",
): void {
    const scratch = realpathSync.native(mkdtempSync(join(tmpdir(), "shallot-offline-")));
    const previous = { ...process.env };
    try {
        const cache = join(scratch, "cache");
        const source = join(scratch, "source");
        const project = join(scratch, "project");
        mkdirSync(cache);
        mkdirSync(source);
        mkdirSync(project);
        const realCache = spawnPackedInstall(
            bunChild(["pm", "cache"]),
            process.cwd(),
            "locating cache",
        ).trim();
        function linkTarballs(from: string, to: string): void {
            for (const entry of readdirSync(from, { withFileTypes: true })) {
                if (!entry.isDirectory()) continue;
                if (entry.name.includes("@@@") && !entry.name.startsWith(`${omit}@`)) {
                    symlinkSync(join(from, entry.name), join(to, entry.name), "junction");
                } else if (entry.name.startsWith("@") && !entry.name.includes("@@@")) {
                    mkdirSync(join(to, entry.name));
                    linkTarballs(join(from, entry.name), join(to, entry.name));
                }
            }
        }
        linkTarballs(realCache, cache);
        expect(readdirSync(cache).some((name) => name.endsWith(".npm"))).toBe(false);
        process.env.BUN_INSTALL_CACHE_DIR = cache;
        process.env.HTTPS_PROXY = process.env.HTTP_PROXY = "http://127.0.0.1:9";
        delete process.env.NO_PROXY;
        delete process.env.no_proxy;
        writeFileSync(
            join(project, "package.json"),
            JSON.stringify({
                name: "offline-consumer",
                dependencies: {
                    "@dylanebert/shallot": "file:../packed.tgz",
                    typegpu: "~0.12.6",
                },
            }),
        );
        run({
            project,
            pack: (manifest) => {
                writeFileSync(join(source, "package.json"), JSON.stringify(manifest));
                spawnPackedInstall(
                    bunChild([
                        "pm",
                        "pack",
                        "--ignore-scripts",
                        "--filename",
                        join(scratch, "packed.tgz"),
                    ]),
                    source,
                    "packing fixture",
                );
            },
        });
    } finally {
        for (const key of Object.keys(process.env)) if (!(key in previous)) delete process.env[key];
        Object.assign(process.env, previous);
        removeTestDirectory(scratch);
    }
}
const engine = { name: "@dylanebert/shallot", version: "0.0.0" };

test("packed registry dependencies install from a manifest-free cache with an unreachable registry", () => {
    offlineFixture(({ project, pack }) => {
        pack({ ...engine, dependencies: { typegpu: "~0.12.6" } });
        packedInstall(["install"], project, "offline fixture");
        expect(existsSync(join(project, "node_modules/typegpu/package.json"))).toBe(true);
        writeFileSync(
            join(project, "consumer.test.ts"),
            'import { expect, test } from "bun:test";\nimport pkg from "@dylanebert/shallot/package.json";\ntest("installed consumer", () => expect(pkg.version).toBe("0.0.0"));\n',
        );
        expect(
            spawnPackedInstall(
                bunChild(["test", "./consumer.test.ts"]),
                project,
                "installed consumer",
            ),
        ).toContain("1 pass");
    });
});

test("a lock-pinned tarball absent from the cache fails with the offline cache cause", () => {
    offlineFixture(({ project, pack }) => {
        pack({ ...engine, dependencies: { typegpu: "~0.12.6" } });
        let failure = "";
        try {
            packedInstall(["install"], project, "missing cache fixture");
        } catch (error) {
            failure = String(error);
        }
        expect(failure).toContain('--offline: "typegpu" is not in the cache');
        expect(failure).not.toContain("Ignoring lockfile");
        expect(failure).not.toContain("InvalidLockfile");
    }, "typegpu");
});

for (const [claim, edges, refusal] of [
    [
        "out-of-range peer",
        { peerDependencies: { typegpu: "^0.11.0" } },
        "typegpu range ^0.11.0 installed 0.12.6",
    ],
    [
        "unsupported git peer",
        { peerDependencies: { typegpu: "git+https://example.invalid/typegpu.git" } },
        "typegpu range git+https://example.invalid/typegpu.git installed 0.12.6: unsupported spec",
    ],
    [
        "dependency hidden by a peer",
        { dependencies: { typegpu: "^0.11.0" }, peerDependencies: { typegpu: "~0.12.6" } },
        "typegpu range ^0.11.0 installed 0.12.6",
    ],
    ["matching npm alias", { peerDependencies: { typegpu: "npm:typegpu@~0.12.6" } }, ""],
    [
        "out-of-range npm alias",
        { peerDependencies: { typegpu: "npm:typegpu@^0.11.0" } },
        "typegpu range npm:typegpu@^0.11.0 installed 0.12.6",
    ],
    [
        "wrong npm alias target",
        { peerDependencies: { typegpu: "npm:other-package@*" } },
        "invalid npm alias target",
    ],
] as const) {
    test(`offline install checks ${claim}`, () => {
        offlineFixture(({ project, pack }) => {
            pack({ ...engine, ...edges });
            if (refusal)
                expect(() => packedInstall(["install"], project, "range fixture")).toThrow(refusal);
            else expect(() => packedInstall(["install"], project, "range fixture")).not.toThrow();
        });
    });
}

for (const spec of [
    "file:../typegpu",
    "link:../typegpu",
    "https://example.invalid/typegpu.tgz",
    "user/repo",
]) {
    test(`offline install refuses unsupported peer ${spec}`, () => {
        offlineFixture(({ project, pack }) => {
            pack({ ...engine, peerDependencies: { typegpu: spec } });
            expect(() => packedInstall(["install"], project, "spec fixture")).toThrow(
                `typegpu range ${spec} installed 0.12.6: unsupported spec`,
            );
        });
    });
}

for (const spec of ["latest", "^banana"]) {
    test(`offline install refuses invalid range ${spec}`, () => {
        offlineFixture(({ project, pack }) => {
            pack({ ...engine, peerDependencies: { typegpu: spec } });
            expect(() => packedInstall(["install"], project, "invalid range fixture")).toThrow(
                `@dylanebert/shallot peer typegpu range ${spec} installed 0.12.6: unsupported spec`,
            );
        });
    });
}

for (const [version, range] of [
    ["1.5.0", "1.x"],
    ["1.5.0", ">=1 <2"],
    ["1.5.0-beta.1", ">=1.5.0-beta.1 <1.5.0"],
    ["1.5.0", "1.0.0 - 2.0.0"],
]) {
    test(`offline install accepts satisfied npm range ${range}`, () => {
        offlineFixture(({ project, pack }) => {
            pack({ ...engine, version, peerDependencies: { "@dylanebert/shallot": range } });
            expect(() => packedInstall(["install"], project, "valid range fixture")).not.toThrow();
        });
    });
}

test("offline install refuses an engine TypeScript peer range that excludes the project's pin", () => {
    offlineFixture(({ project, pack }) => {
        pack({ ...engine, peerDependencies: { typescript: "^5.0.0 || ^6.0.0" } });
        writeFileSync(
            join(project, "package.json"),
            JSON.stringify({
                name: "peer-range-consumer",
                dependencies: {
                    "@dylanebert/shallot": "file:../packed.tgz",
                    typescript: "^7.0.2",
                },
            }),
        );
        expect(() => packedInstall(["install"], project, "peer range fixture")).toThrow(
            "@dylanebert/shallot peer typescript range ^5.0.0 || ^6.0.0 installed 7.0.2",
        );
    });
});
