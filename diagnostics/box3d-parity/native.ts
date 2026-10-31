// Builds C and C++ harnesses against BOX3D at the revision src/standard/physics/oracle/pin.ts pins. Libraries
// are cached by revision and architecture; each harness binary is keyed by its contents. Existing scene
// oracles use the host architecture.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { BOX3D_PIN } from "../../src/standard/physics/oracle/pin";

const here = import.meta.dir;

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
    if (benchmarks) run(["cc", ...(architecture ? ["-arch", architecture] : []), "-O2", "-DNDEBUG", "-std=c17", "-ffp-contract=off", ...inc, "-c", benchmarks, "-o", object]);
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

/** BOX3D, once its source tree is the pinned tree. */
function pinnedSource(): string {
    if (verified) return verified;
    const box3d = process.env.BOX3D;
    const { revision, tree } = BOX3D_PIN;
    const fetch = `git -C <box3d clone> archive ${revision} | tar -x -C <dir>`;
    if (!box3d) throw new Error(`set BOX3D to a Box3D source tree at ${revision}: ${fetch}`);
    const actual = sourceTree(resolve(box3d));
    if (actual !== tree) throw new Error(`BOX3D's source tree is ${actual}, not ${revision}'s ${tree}: ${fetch}`);
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
