// Builds C and C++ harnesses against BOX3D at the pinned revision. Libraries are cached by revision and
// architecture; each harness binary is keyed by its contents. Existing scene oracles use the host architecture.
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

/**
 * The Box3D revision every oracle, corpus lane and gold is held to, and its source tree. A stage moves the
 * pin here and replaces src/standard/physics/oracle/box3d/ with box3d-oracle's lanes generated at it.
 *
 * Authority: Box3D's CMake default build on x86_64, SSE2 at SIMD width 4, the path the kernel's wasm
 * simd128 follows (native-evidence.ts). box3d-oracle generates the lanes scalar (BOX3D_DISABLE_SIMD=ON) and
 * the scene oracles build the host default, NEON on arm64. At 47d7f7cc and every re-pin stage's revision
 * through 3362633, the scalar and default builds, arm64 and -arch x86_64, hash native.c's scenes equal, so
 * each measures the authority there; where builds differ, SSE2 decides. Under Rosetta, runtime dispatch
 * finds no AVX2 and keeps width 4; an AVX2 host would dispatch width 8, which no oracle here measures.
 */
export const BOX3D_PIN = {
    revision: "47d7f7cc7e091142c08d11dc7d2e493c5d34f536",
    tree: "0a6472d03ff61c9af28216443fba341a21cbed96",
} as const;
const here = import.meta.dir;
const corpus = resolve(here, "../../src/standard/physics/oracle/box3d");

/** Run `cmd` from the repository root; its stdout, or a throw naming its stderr. */
export function run(cmd: string[], env: Record<string, string> = {}): string {
    const proc = Bun.spawnSync(cmd, {
        env: { ...process.env, ...env },
        cwd: resolve(here, "../.."),
    });
    if (proc.exitCode !== 0) throw new Error(`${cmd.join(" ")} failed:\n${proc.stderr.toString()}`);
    return proc.stdout.toString();
}

/** The path of the native binary, built on first use. */
export function nativeBinary(sourceName = "native.c", architecture?: "x86_64" | "arm64", benchmarks?: string): string {
    const box3d = pinnedSource();
    const build = join(
        tmpdir(),
        `box3d-parity-${BOX3D_PIN.revision.slice(0, 8)}${architecture ? `-${architecture}` : ""}`,
    );
    const cmake = join(build, "cmake");
    const source = resolve(here, sourceName);
    const native = join(
        build,
        `native-${Bun.hash(Buffer.concat([readFileSync(source), benchmarks ? readFileSync(benchmarks) : Buffer.alloc(0)])).toString(16).padStart(16, "0")}`,
    );
    if (existsSync(native)) return native;
    if (!existsSync(join(cmake, "src/libbox3d.a"))) {
        mkdirSync(build, { recursive: true });
        run([
            "cmake",
            "-S",
            box3d,
            "-B",
            cmake,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBOX3D_BENCHMARKS=ON",
            ...(architecture ? [`-DCMAKE_OSX_ARCHITECTURES=${architecture}`] : []),
            `-DFETCHCONTENT_BASE_DIR=${join(build, "fetch")}`,
        ]);
        run(["cmake", "--build", cmake, "-j", "8", "--target", "box3d", "shared"]);
    }
    const inc = ["include", "src", "shared"].map((d) => `-I${join(box3d, d)}`);
    const object = `${native}-benchmarks.o`;
    if (benchmarks) run(["cc", "-O2", "-DNDEBUG", "-std=c17", "-ffp-contract=off", ...inc, "-c", benchmarks, "-o", object]);
    run([
        source.endsWith(".cpp") ? "c++" : "cc",
        ...(architecture ? ["-arch", architecture] : []),
        "-O2",
        source.endsWith(".cpp") ? "-std=c++17" : "-std=c17",
        "-ffp-contract=off",
        ...inc,
        source,
        ...(benchmarks ? [object] : []),
        join(cmake, "shared/libshared.a"),
        join(cmake, "src/libbox3d.a"),
        "-lm",
        "-o",
        native,
    ]);
    return native;
}

let verified: string | undefined;

/** BOX3D, once its source tree is the pinned tree and every corpus lane was generated at the pin. */
function pinnedSource(): string {
    if (verified) return verified;
    const box3d = process.env.BOX3D;
    const { revision, tree } = BOX3D_PIN;
    const fetch = `git -C <box3d clone> archive ${revision} | tar -x -C <dir>`;
    if (!box3d) throw new Error(`set BOX3D to a Box3D source tree at ${revision}: ${fetch}`);
    const actual = sourceTree(resolve(box3d));
    if (actual !== tree) throw new Error(`BOX3D's source tree is ${actual}, not ${revision}'s ${tree}: ${fetch}`);
    for (const lane of readdirSync(corpus)) {
        const { upstream } = JSON.parse(readFileSync(join(corpus, lane, "manifest.json"), "utf8")) as {
            upstream: { sha: string; tree: string };
        };
        if (upstream.sha !== revision || upstream.tree !== tree)
            throw new Error(`corpus lane ${lane} was generated at ${upstream.sha}, not the pinned ${revision}`);
    }
    verified = box3d;
    return box3d;
}

/** The git tree of `directory`'s files that Box3D's .gitignore keeps, from a git checkout or an archive. */
function sourceTree(directory: string): string {
    const git = mkdtempSync(join(tmpdir(), "box3d-tree-"));
    try {
        run(["git", "init", "--quiet", "--bare", git]);
        const env = { GIT_DIR: git, GIT_WORK_TREE: directory, GIT_INDEX_FILE: join(git, "index") };
        run(["git", "-C", directory, "add", "--all", "."], env);
        return run(["git", "write-tree"], env).trim();
    } finally {
        rmSync(git, { recursive: true, force: true });
    }
}
