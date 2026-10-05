// Manual oracle for where rain and junkyard first diverge from native Box3D (strategy unit box3d-parity,
// stage 1). Run by path, with a Box3D checkout at 47d7f7cc:
//
//     BOX3D=/path/to/box3d bun test ./diagnostics/box3d-parity/divergence.oracle.ts
//
// Builds Box3D's box3d and shared libraries (Release, default SIMD) and native.c into a cache under the
// system temp directory, then runs native.c and scenes.ts on the smallest diverging scenes at one thread
// and diffs them. Each test asserts the divergence it reproduces, so it fails once the cause is fixed.
import { expect, setDefaultTimeout, test } from "bun:test";
import { existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createCylinder, createRock } from "../../src/standard/physics/api";
import { uploadGeometry } from "../../src/standard/physics/kernel/geocolumns";
import { kernel } from "../../src/standard/physics/kernel/kernel";

setDefaultTimeout(180_000);

const SHA = "47d7f7cc7e091142c08d11dc7d2e493c5d34f536";
const here = import.meta.dir;
const box3d = process.env.BOX3D;
if (!box3d) throw new Error("set BOX3D to a Box3D checkout at 47d7f7cc");
const head = Bun.spawnSync(["git", "-C", box3d, "rev-parse", "HEAD"]).stdout.toString().trim();
if (head !== SHA) throw new Error(`BOX3D is at ${head || "no git revision"}, not ${SHA}`);

const build = join(tmpdir(), `box3d-parity-${SHA.slice(0, 8)}`);
const native = join(build, "native");

function run(cmd: string[], env: Record<string, string> = {}): string {
    const proc = Bun.spawnSync(cmd, {
        env: { ...process.env, ...env },
        cwd: resolve(here, "../.."),
    });
    if (proc.exitCode !== 0) throw new Error(`${cmd.join(" ")} failed:\n${proc.stderr.toString()}`);
    return proc.stdout.toString();
}

if (!existsSync(native)) {
    mkdirSync(build, { recursive: true });
    const cmake = join(build, "cmake");
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
    const inc = ["include", "src", "shared"].map((d) => `-I${join(box3d, d)}`);
    run([
        "cc",
        "-O2",
        "-std=c17",
        "-ffp-contract=off",
        ...inc,
        join(here, "native.c"),
        join(cmake, "shared/libshared.a"),
        join(cmake, "src/libbox3d.a"),
        "-o",
        native,
    ]);
}

type Side = { native: string[]; shallot: string[] };
function both(scene: string, steps: number, env: Record<string, string>): Side {
    const args = [scene, "1", String(steps)];
    return {
        native: run([native, ...args], env)
            .trim()
            .split("\n"),
        shallot: run(["bun", join(here, "scenes.ts"), ...args], env)
            .trim()
            .split("\n"),
    };
}
const hashes = (lines: string[]) => lines.filter((l) => /^\d+ 0x/.test(l));
const firstDifference = (s: Side) => {
    const a = hashes(s.native),
        b = hashes(s.shallot);
    const i = a.findIndex((line, k) => line !== b[k]);
    return i < 0 ? -1 : Number(a[i].split(" ")[0]);
};

test("rain at 9 x 9 tiles of one ragdoll first diverges at step 272, on joints colored over bits a slept ragdoll left in the graph", () => {
    const env = { RAIN_COUNT: "9", RAIN_GROUP: "1", COLORS: "271" };
    const side = both("rain-n", 273, env);
    expect(firstDifference(side)).toBe(272);
    // After step 271 the hashes are equal, but Shallot's graph colors 0-3 still hold the bits of ragdoll 0
    // (bodies 81-94), asleep since its island slept: Shallot's trySleepIsland (world/solverset.ts) moves
    // the island's joints out of the graph without clearing them, as Box3D's b3TrySleepIsland does
    // (solver_set.c:377-382). Step 272 destroys and recreates that ragdoll on the same body ids, and its new
    // joints take other colors. Native holds no bit Shallot lacks.
    const bits = (lines: string[]) => new Set(lines.filter((l) => l.startsWith("C ")));
    const n = bits(side.native),
        s = bits(side.shallot);
    const extra = [...s].filter((l) => !n.has(l));
    expect([...n].filter((l) => !s.has(l))).toEqual([]);
    const bodies = new Set(extra.map((l) => Number(l.split(" ").at(-1))));
    expect([...bodies].sort((a, b) => a - b)).toEqual(Array.from({ length: 14 }, (_, i) => 81 + i));
    expect(new Set(extra.map((l) => l.split(" ")[3]))).toEqual(new Set(["0", "1", "2", "3"]));
    console.log(
        `rain-n 9x1: first divergent step 272; ${extra.length} stale color bits on bodies 81-94 after step 271`,
    );
});

test("junkyard with one rock first diverges at step 12, on an edge-pair normal Shallot's collide_hulls rounds differently from the same inputs", () => {
    const env = { ROCKS: "19,0,11", PROBE: "12", FOCUS: "1" };
    const side = both("junk", 13, env);
    expect(firstDifference(side)).toBe(12);
    // Step 12's manifold for the pusher's cylinder (A) and the rock (B) is one edge-pair point. Fed the same
    // transformBtoA with a fresh cache, Shallot's kernel returns the native point and separation but a
    // normal x 18 ulps away: compute_separating_axis (crates/physics/src/manifold.rs) sums its dot3_w as
    // z + (y + x), where Box3D's b3Dot3W sums x + (y + z) (simd.h:527 NEON, :691 SSE2, :888 scalar).
    const x = side.native
        .find((l) => l.startsWith("X 12"))
        ?.split(" ")
        .slice(-7);
    const fresh = side.native
        .find((l) => l.startsWith("H 12") && l.includes("fresh"))
        ?.split(" ")
        .slice(-3);
    if (!x || !fresh) throw new Error("native printed no hull-hull probe at step 12");
    const f = (h: string) => new Float32Array(new Uint32Array([Number.parseInt(h, 16)]).buffer)[0];
    const h = (v: number) =>
        new Uint32Array(new Float32Array([v]).buffer)[0].toString(16).padStart(8, "0");
    const a = createCylinder(24, 4, 0, 16),
        b = createRock(1.5);
    uploadGeometry(undefined, [a, b]);
    const k = kernel(undefined);
    const t = x.map(f);
    const count = k.collideHullsGeo(
        a.geoIndex,
        b.geoIndex,
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        t[6],
    );
    const out = new Float32Array(k.memory.buffer, k.geoOutPtr(), 4);
    const normal = [out[1], out[2], out[3]].map(h);
    expect(count).toBe(1);
    expect(fresh).toEqual(["3ccc4560", "3f7de980", "3dfffc75"]);
    expect(normal).toEqual(["3ccc454e", "3f7de980", "3dfffc75"]);
    console.log(
        `junk 1 rock: first divergent step 12; edge normal native ${fresh.join(" ")}, Shallot ${normal.join(" ")}`,
    );
});

test("junkyard with two stacked rocks keeps equal hashes but misses Box3D's cached separated edge pair from step 176", () => {
    // The second difference, which full junkyard reaches at step 156 once the first is gone: when a cached
    // edge pair's axis shows the shapes separated, Box3D's b3CollideHulls returns a cache hit with the cache
    // untouched (convex_manifold.c:1927-1932). Shallot's collide_hulls has no such return, falls through to
    // the full separating-axis search and rewrites the cache, so a later contact starts from another axis.
    const side = both("junk", 200, { ROCKS: "20,1,7;20,2,7", CACHE: "2" });
    expect(firstDifference(side)).toBe(-1);
    const at = (lines: string[]) => lines.find((l) => l.startsWith("S 176 "));
    expect(at(side.native)).toBe(
        "S 176 contact 2 bodies 3 2 manifolds 0 cache sep 3db82c82 type 4 indexA 34 indexB 38 hit 1",
    );
    expect(at(side.shallot)).toBe(
        "S 176 contact 2 bodies 3 2 manifolds 0 cache sep 3cc59cc8 type 4 indexA 34 indexB 38 hit 0",
    );
    const first = side.native.find((l, i) => l.startsWith("S ") && l !== side.shallot[i]);
    expect(first).toBe(at(side.native));
});
