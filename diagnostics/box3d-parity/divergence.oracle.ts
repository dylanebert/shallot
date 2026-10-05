// Manual oracle for Shallot following native Box3D on the smallest rain and junkyard scenes that once
// diverged (strategy unit box3d-parity). Run by path, with a Box3D checkout at 47d7f7cc:
//
//     BOX3D=/path/to/box3d bun test ./diagnostics/box3d-parity/divergence.oracle.ts
//
// Builds native.c (native.ts), then runs it and scenes.ts on each scene at one thread and diffs them. Each
// test asserts equality with native at the state where its cause showed.
import { expect, setDefaultTimeout, test } from "bun:test";
import { join } from "node:path";
import { createCylinder, createRock } from "../../src/standard/physics/api";
import { uploadGeometry } from "../../src/standard/physics/kernel/geocolumns";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { nativeBinary, run } from "./native";

setDefaultTimeout(180_000);

const here = import.meta.dir;
const native = nativeBinary();

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

test("rain at 9 x 9 tiles of one ragdoll matches native through step 272, its graph colors free of a slept ragdoll's bits", () => {
    // Box3D's b3TrySleepIsland clears each joint's body bits from its graph color as it moves the joint to
    // the sleeping set (solver_set.c:377-382). Step 272 destroys and recreates the ragdoll whose island
    // slept (bodies 81-94) on the same body ids; a stale bit would give its new joints other colors.
    const side = both("rain-n", 273, { RAIN_COUNT: "9", RAIN_GROUP: "1", COLORS: "271" });
    const bits = (lines: string[]) => lines.filter((l) => l.startsWith("C ")).sort();
    expect(bits(side.shallot)).toEqual(bits(side.native));
    expect(firstDifference(side)).toBe(-1);
});

test("junkyard with one rock matches native through step 12, its edge-pair normal rounded as Box3D's", () => {
    const env = { ROCKS: "19,0,11", PROBE: "12", FOCUS: "1" };
    const side = both("junk", 13, env);
    // Step 12's manifold for the pusher's cylinder (A) and the rock (B) is one edge-pair point, whose normal
    // compute_separating_axis (crates/physics/src/manifold.rs) derives through dot_wide; Box3D's b3Dot3W sums
    // x + (y + z) (simd.h:527 NEON, :691 SSE2, :888 scalar). Fed native's transformBtoA with a fresh cache,
    // Shallot's kernel returns native's normal bit for bit.
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
    expect(count).toBe(1);
    expect([out[1], out[2], out[3]].map(h)).toEqual(fresh);
    expect(firstDifference(side)).toBe(-1);
});

test("junkyard with two stacked rocks matches native's contact caches through step 199, keeping a cached separated edge pair", () => {
    // When a cached edge pair's axis shows the shapes separated, Box3D's b3CollideHulls returns a cache hit
    // with the cache untouched (convex_manifold.c:1927-1932). From step 176 the pusher-rock contact is
    // such a pair; a full search there would rewrite its cache and a later contact would start from
    // another axis.
    const side = both("junk", 200, { ROCKS: "20,1,7;20,2,7", CACHE: "2" });
    expect(firstDifference(side)).toBe(-1);
    const caches = (lines: string[]) => lines.filter((l) => l.startsWith("S "));
    expect(caches(side.native).find((l) => l.startsWith("S 176 "))).toBe(
        "S 176 contact 2 bodies 3 2 manifolds 0 cache sep 3db82c82 type 4 indexA 34 indexB 38 hit 1",
    );
    expect(caches(side.shallot)).toEqual(caches(side.native));
});
