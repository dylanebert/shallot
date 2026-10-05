// Builds native.c against the Box3D checkout named by BOX3D, which must be at 47d7f7cc: Box3D's box3d and
// shared libraries (Release, default SIMD) once into a cache under the system temp directory, then
// native.c into a binary keyed by its contents.
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SHA = "47d7f7cc7e091142c08d11dc7d2e493c5d34f536";
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
export function nativeBinary(): string {
    const box3d = process.env.BOX3D;
    if (!box3d) throw new Error("set BOX3D to a Box3D checkout at 47d7f7cc");
    const head = Bun.spawnSync(["git", "-C", box3d, "rev-parse", "HEAD"]).stdout.toString().trim();
    if (head !== SHA) throw new Error(`BOX3D is at ${head || "no git revision"}, not ${SHA}`);
    const build = join(tmpdir(), `box3d-parity-${SHA.slice(0, 8)}`);
    const cmake = join(build, "cmake");
    const source = join(here, "native.c");
    const native = join(
        build,
        `native-${Bun.hash(readFileSync(source)).toString(16).padStart(16, "0")}`,
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
            `-DFETCHCONTENT_BASE_DIR=${join(build, "fetch")}`,
        ]);
        run(["cmake", "--build", cmake, "-j", "8", "--target", "box3d", "shared"]);
    }
    const inc = ["include", "src", "shared"].map((d) => `-I${join(box3d, d)}`);
    run([
        "cc",
        "-O2",
        "-std=c17",
        "-ffp-contract=off",
        ...inc,
        source,
        join(cmake, "shared/libshared.a"),
        join(cmake, "src/libbox3d.a"),
        "-o",
        native,
    ]);
    return native;
}
