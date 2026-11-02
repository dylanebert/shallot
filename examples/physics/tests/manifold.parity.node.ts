import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ManifoldPlugin, ManifoldSample, manifoldNames } from "../src/manifold";
import { manifoldDiagnosticScenes } from "../src/manifold-diagnostics";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";
import { manifoldDiagnosticGesture } from "./manifold-diagnostic-gesture";
import { manifoldDiagnosticRecords } from "./manifold-diagnostic-records";
import { manifoldInput } from "./manifold-input";
import { manifoldPoseRecord, manifoldPoses } from "./manifold-poses";
import { manifoldRecords } from "./manifold-records";

setDefaultTimeout(TIMEOUT.node);
for (const scene of manifoldNames)
    test(`${scene}: 600 native manifolds, caches and geometry from recorded world poses`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const stem = scene + (controlled ? ".controls" : "");
            const gold = async (suffix: string) =>
                (await Bun.file(new URL(`./${stem}.${suffix}`, import.meta.url)).text())
                    .trimEnd()
                    .split("\n");
            const hashes = await gold("gold.txt"),
                records = await gold("manifold.gold.txt");
            const poses = controlled
                ? manifoldPoses(
                      await Bun.file(new URL(`./${scene}.poses.gold.txt`, import.meta.url)).text(),
                  )
                : [];
            const changes = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((l) => {
                          const [step, name, value] = l.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [ManifoldPlugin],
                setup: (w) => {
                    createScene(w, scene);
                },
            });
            const values = controlValues(scene),
                ray = { origin: { x: 0, y: 0, z: 0 }, held: false };
            try {
                expect(hashes).toHaveLength(600);
                expect(records).toHaveLength(600);
                if (controlled) expect(poses).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const c of changes)
                        if (c.step === step && !c.name.startsWith("drag"))
                            manifoldInput(app.world, scene, values, ray, c.name, c.value);
                    if (controlled) app.world.resource(ManifoldSample).b = poses[step]!;
                    app.world.tick();
                    expect(
                        hashPhysics(app.world).toString(16).padStart(16, "0"),
                        `${stem} step ${step} hash`,
                    ).toBe(hashes[step]);
                    expect(
                        drawingDigest(manifoldRecords(app.world)),
                        `${stem} step ${step} manifold`,
                    ).toBe(records[step]);
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });

for (const scene of Object.keys(
    manifoldDiagnosticScenes,
) as (keyof typeof manifoldDiagnosticScenes)[])
    for (const mode of ["default", "controls", "gizmo"] as const)
        test(`${scene}: native geometry, contacts and diagnostics / ${mode}`, async () => {
            const start = performance.now(),
                stem = scene + (mode === "default" ? "" : `.${mode}`),
                count = mode === "gizmo" ? 24 : 600;
            const read = async (suffix: string) =>
                (await Bun.file(new URL(`./${stem}.${suffix}`, import.meta.url)).text())
                    .trimEnd()
                    .split("\n");
            const hashes = sampleGold((await read("gold.txt")).join("\n")),
                manifolds = await read("manifold.gold.txt"),
                diagnostics = await read("diagnostic.gold.txt"),
                poses = mode === "gizmo" ? await read("poses.gold.txt") : [];
            const changes =
                mode === "controls"
                    ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                          .trimEnd()
                          .split("\n")
                          .map((line) => {
                              const [step, name, value] = line.split("\t");
                              return { step: Number(step), name: name!, value: Number(value) };
                          })
                    : [];
            const app = await createApp({
                defaults: false,
                plugins: [ManifoldPlugin],
                setup: (w) => {
                    createScene(w, scene);
                },
            });
            try {
                expect(hashes).toHaveLength(count);
                expect(manifolds).toHaveLength(count);
                expect(diagnostics).toHaveLength(count);
                const values = controlValues(scene),
                    controls = manifoldDiagnosticScenes[scene].controls,
                    used = new Set<string>();
                for (let step = 0; step < count; step++) {
                    for (const c of changes)
                        if (c.step === step) {
                            const feature = ["auto", "faceA", "faceB", "edgePair"].indexOf(c.name),
                                label = feature >= 0 ? "Feature" : c.name,
                                key = Object.entries(controls).find(
                                    ([, control]) => control.label === label,
                                )![0];
                            changeControl(
                                app.world,
                                scene,
                                [],
                                values,
                                key,
                                feature >= 0 ? feature : c.value,
                            );
                            used.add(key);
                        }
                    if (mode === "gizmo") {
                        manifoldDiagnosticGesture(app.world, step);
                        if (step >= 4 && step <= 12)
                            expect(app.world.resource(ManifoldSample).tracking).toBe(true);
                        if (step === 13)
                            expect(app.world.resource(ManifoldSample).tracking).toBe(false);
                    }
                    app.world.tick();
                    if (mode === "gizmo")
                        expect(manifoldPoseRecord(step, app.world.resource(ManifoldSample).b)).toBe(
                            poses[step]!,
                        );
                    expect(hashPhysics(app.world), `${stem} step ${step} world`).toBe(
                        hashes[step]!.hash,
                    );
                    expect(
                        drawingDigest(manifoldRecords(app.world)),
                        `${stem} step ${step} contacts`,
                    ).toBe(manifolds[step]!);
                    expect(
                        drawingDigest(manifoldDiagnosticRecords(app.world)),
                        `${stem} step ${step} diagnostic`,
                    ).toBe(diagnostics[step]!);
                }
                if (mode === "controls")
                    expect([...used].sort()).toEqual(
                        Object.keys(controls)
                            .filter((key) => !controls[key]!.read)
                            .sort(),
                    );
            } finally {
                app.dispose();
            }
            console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
        });
test("Manifold diagnostics clear on leaving and return to their native default", async () => {
    const app = await createApp({ defaults: false, plugins: [ManifoldPlugin] });
    try {
        const select = sceneSwitcher(app.world);
        for (const scene of ["hullCulling", "edgeAxisWedge"] as const) {
            select(scene);
            app.world.tick();
            const first = (
                await Bun.file(new URL(`./${scene}.diagnostic.gold.txt`, import.meta.url)).text()
            ).split("\n")[0]!;
            expect(drawingDigest(manifoldDiagnosticRecords(app.world))).toBe(first);
            for (let step = 0; step < 24; step++) {
                manifoldDiagnosticGesture(app.world, step);
                app.world.tick();
            }
            expect(drawingDigest(manifoldDiagnosticRecords(app.world))).not.toBe(first);
            select("boxStack");
            expect(app.world.resource(ManifoldSample)).toEqual(ManifoldSample.create());
            app.world.tick();
            select(scene);
            app.world.tick();
            expect(drawingDigest(manifoldDiagnosticRecords(app.world))).toBe(first);
            select("boxStack");
            app.world.tick();
        }
    } finally {
        app.dispose();
    }
});
