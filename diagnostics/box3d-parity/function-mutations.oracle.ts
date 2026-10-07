// BOX3D=~/kex/reference/box3d bun test ./diagnostics/box3d-parity/function-mutations.oracle.ts
// Each control changes product arithmetic in a temporary source copy, never an assertion or result.
import { expect, setDefaultTimeout, test } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

setDefaultTimeout(180_000);
const root = resolve(import.meta.dir, "../..");
const scratch = mkdtempSync(join(tmpdir(), "shallot-box3d-mutations-"));
process.on("exit", () => rmSync(scratch, { recursive: true, force: true }));
writeFileSync(
    join(scratch, "Cargo.toml"),
    readFileSync(join(root, "Cargo.toml"), "utf8").replace(
        /^members = .*$/m,
        'members = ["crates/physics"]',
    ),
);
cpSync(join(root, "Cargo.lock"), join(scratch, "Cargo.lock"));
cpSync(join(root, "rust-toolchain.toml"), join(scratch, "rust-toolchain.toml"));
cpSync(join(root, "crates/physics"), join(scratch, "crates/physics"), {
    recursive: true,
    filter: (path) => !path.includes("target-shared"),
});
const output = join(scratch, "target/wasm32-unknown-unknown/release/shallot_physics.wasm");
function build(): void {
    const proc = Bun.spawnSync(
        [
            "cargo",
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "shallot-physics",
            "--features",
            "box3d-oracle",
        ],
        {
            cwd: scratch,
            env: {
                ...process.env,
                RUSTFLAGS: "-C target-feature=+simd128 --remap-path-prefix=crates/physics/=",
            },
        },
    );
    if (proc.exitCode !== 0) throw new Error(proc.stderr.toString());
}
function run(subject: string, selector: string) {
    return Bun.spawnSync(
        [
            "bun",
            "test",
            "--preload",
            join(import.meta.dir, "oracle-preload.ts"),
            "--todo",
            join(import.meta.dir, `${subject}.oracle.ts`),
            "-t",
            selector,
        ],
        {
            cwd: root,
            env: { ...process.env, BOX3D_ORACLE_WASM: output },
        },
    );
}
type Mutation = {
    family: string;
    file: string;
    old: string;
    replacement: string;
    subject: string;
    selector: string;
};
const mutations: Mutation[] = [
    {
        family: "normalize",
        file: "math.rs",
        old: "    (a.scale(inv_length), length)",
        replacement: "    (a.scale(inv_length * 0.75), length)",
        subject: "functions",
        selector: "normalize ordinary",
    },
    {
        family: "unwind",
        file: "math.rs",
        old: "    remainderf(radians, TWO_PI)",
        replacement: "    remainderf(radians * 1.125, TWO_PI)",
        subject: "functions",
        selector: "unwind ordinary",
    },
    {
        family: "wide clamp",
        file: "wide.rs",
        old: "    let c = nb.max(a);",
        replacement: "    let c = nb.max(a).add(FloatW::splat(0.125));",
        subject: "functions",
        selector: "symmetric clamp unequal",
    },
    {
        family: "convex manifold",
        file: "manifold.rs",
        old: "    let point = center1\n        .mul_add(sphere_a.radius, normal)\n        .add(center2)\n        .mul_sub(sphere_b.radius, normal)\n        .scale(0.5);",
        replacement:
            "    let point = center1\n        .mul_add(sphere_a.radius, normal)\n        .add(center2)\n        .mul_sub(sphere_b.radius, normal)\n        .scale(0.6);",
        subject: "functions",
        selector: "sphere-sphere generated fresh",
    },
    {
        family: "triangle manifold",
        file: "triangle_manifold.rs",
        old: "            .mul_sub(0.5 * (distance + radius), plane.normal);",
        replacement: "            .mul_sub(0.6 * (distance + radius), plane.normal);",
        subject: "functions",
        selector: "triangle-capsule:",
    },
    {
        family: "distance",
        file: "distance.rs",
        old: "    output.distance = w.0.distance(w.1);",
        replacement: "    output.distance = w.0.distance(w.1) * 1.125;",
        subject: "functions",
        selector: "distance separated",
    },
    {
        family: "shape cast",
        file: "distance.rs",
        old: "                output.fraction = alpha;",
        replacement: "                output.fraction = alpha * 0.75;",
        subject: "functions",
        selector: "shape cast:",
    },
    {
        family: "TOI",
        file: "toi.rs",
        old: "                out.fraction = input.max_fraction;",
        replacement: "                out.fraction = input.max_fraction * 0.75;",
        subject: "functions",
        selector: "TOI:",
    },
    {
        family: "mesh contact",
        file: "mesh_contact.rs",
        old: "        m.normal = matrix.mul_v(cluster.normal);",
        replacement: "        m.normal = matrix.mul_v(cluster.normal).scale(1.125);",
        subject: "public-queries",
        selector: "mesh contacts:",
    },
    {
        family: "height-field contact",
        file: "mesh_contact.rs",
        old: "        m.normal = matrix.mul_v(cluster.normal);",
        replacement: "        m.normal = matrix.mul_v(cluster.normal).scale(1.125);",
        subject: "public-queries",
        selector: "heightfield contacts:",
    },
    {
        family: "compound contact",
        file: "arena.rs",
        old: "            for p in &mut m.points[..m.point_count] {\n                p.anchor_a = p.anchor_a.add(child_offset);\n            }",
        replacement:
            "            for p in &mut m.points[..m.point_count] {\n                p.anchor_a = p.anchor_a.add(child_offset.scale(0.5));\n            }",
        subject: "public-queries",
        selector: "compound contacts:",
    },
    {
        family: "world query",
        file: "world_query.rs",
        old: "        let translation = Vec3::new(r[9], r[10], r[11]);",
        replacement: "        let translation = Vec3::new(r[9], r[10] * 1.25, r[11]);",
        subject: "public-queries",
        selector: "world castRay:",
    },
    {
        family: "body query",
        file: "body_query.rs",
        old: "        let translation = Vec3::new(r[9], r[10], r[11]);",
        replacement: "        let translation = Vec3::new(r[9], r[10] * 1.25, r[11]);",
        subject: "public-queries",
        selector: "body castRay:",
    },
    {
        family: "joint creation",
        file: "joint_creation.rs",
        old: "        (DJ_LENGTH, maxf(length, 0.005)),",
        replacement: "        (DJ_LENGTH, maxf(length * 0.75, 0.005)),",
        subject: "joint-defaults",
        selector: "distance creation defaults",
    },
    {
        family: "joint setters",
        file: "joint_lifecycle.rs",
        old: "        lower = crate::math::clampf(\n            lower,",
        replacement: "        lower = crate::math::clampf(\n            lower * 1.25,",
        subject: "joint-defaults",
        selector: "distance seeded public setters",
    },
    {
        family: "compound tree query",
        file: "compound_query.rs",
        old: "        if !bounds_overlap(lo, hi, lower, upper) {",
        replacement:
            "        if !bounds_overlap(lo, hi, lower.add(Vec3::new(200.0, 0.0, 0.0)), upper) {",
        subject: "compound-depth",
        selector: "compound retained images below",
    },
];
const baseline = join(scratch, "baseline.wasm");
build();
cpSync(output, baseline);
for (const mutation of mutations) {
    test(`${mutation.family}: changing kernel arithmetic reddens its previously green native case`, () => {
        cpSync(baseline, output);
        const control = run(mutation.subject, mutation.selector);
        expect(control.exitCode, control.stderr.toString()).toBe(0);
        const path = join(scratch, "crates/physics/src", mutation.file);
        const source = readFileSync(path, "utf8");
        expect(source.split(mutation.old).length, `${mutation.file} mutation must be unique`).toBe(
            2,
        );
        try {
            writeFileSync(path, source.replace(mutation.old, mutation.replacement));
            build();
            const mutant = run(mutation.subject, mutation.selector);
            const evidence = mutant.stderr.toString();
            expect(mutant.exitCode).not.toBe(0);
            expect(evidence).toContain("(fail)");
            console.info(
                `${mutation.family}: ${mutation.file}; ${mutation.selector}; control green, arithmetic mutant red`,
            );
        } finally {
            writeFileSync(path, source);
        }
    });
}
