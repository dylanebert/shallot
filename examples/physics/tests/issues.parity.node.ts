import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, type World } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Geometry, GeometryPlugin } from "../src/geometry";
import { hullCrashPoints, IssueState, IssuesPlugin, issueScenes } from "../src/issues";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { hullRecord } from "./geometry-records";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
function readouts(world: World, scene: "ghostCollisions" | "restitutionOvershoot") {
    const s = world.resource(IssueState);
    const row =
        scene === "ghostCollisions"
            ? [
                  issueScenes.ghostCollisions.controls.Launches.read(world),
                  issueScenes.ghostCollisions.controls.Worst.read(world),
                  s.markers,
              ]
            : [
                  issueScenes.restitutionOvershoot.controls.Drop.read(),
                  issueScenes.restitutionOvershoot.controls.Maximum.read(world),
                  Number(issueScenes.restitutionOvershoot.controls.Result.read(world) === "No"),
              ];
    return JSON.parse(
        JSON.stringify(row, (_key, value) =>
            typeof value === "number" ? Number(value.toPrecision(9)) : value,
        ),
    );
}
for (const scene of Object.keys(issueScenes) as (keyof typeof issueScenes)[]) {
    test(`${scene} equals every native step`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const geometry =
            scene === "hullCrash"
                ? (await Bun.file(new URL("./hullCrash.geometry.gold.txt", import.meta.url)).text())
                      .trim()
                      .split("\n")
                : [];
        const rows =
            scene === "ghostCollisions" || scene === "restitutionOvershoot"
                ? (await Bun.file(new URL(`./${scene}.readouts.gold.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((line) => JSON.parse(line))
                : [];
        const start = performance.now();
        const app = await createApp({ defaults: false, plugins: [GeometryPlugin, IssuesPlugin] });
        createScene(app.world, scene);
        try {
            for (const [step, expected] of gold.entries()) {
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step}`).toBe(expected.hash);
                if (rows.length) {
                    if (scene === "ghostCollisions" || scene === "restitutionOvershoot")
                        expect(readouts(app.world, scene), `readouts ${step}`).toEqual(rows[step]);
                }
                if (scene === "hullCrash")
                    expect(
                        drawingDigest(
                            app.world.resource(Geometry).hulls[0]
                                ? [
                                      [
                                          ...hullRecord(app.world.resource(Geometry).hulls[0]!, 0),
                                          ...app.world
                                              .resource(Geometry)
                                              .hulls[0]!.faces.map((face) => face.edge),
                                      ],
                                  ]
                                : hullCrashPoints().map((p) => [0, p.x, p.y, p.z, 5]),
                        ),
                        `geometry ${step}`,
                    ).toBe(geometry[step]!);
            }
        } finally {
            app.dispose();
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
test("off-center shape's initial angular velocity recovers and applies only once", async () => {
    const gold = sampleGold(
        await Bun.file(new URL("./slideTwistOffCenterShape.gold.txt", import.meta.url)).text(),
    );
    const app = await createApp({ defaults: false, plugins: [GeometryPlugin, IssuesPlugin] });
    try {
        createScene(app.world, "slideTwistOffCenterShape");
        const pending = app.world.snapshot();
        for (let replay = 0; replay < 2; replay++) {
            if (replay) app.world.restore(pending);
            for (const expected of gold.slice(0, 3)) {
                app.world.tick();
                expect(hashPhysics(app.world)).toBe(expected.hash);
            }
        }
    } finally {
        app.dispose();
    }
});

for (const scene of ["crash", "ghostCollisions"] as const)
    test(`${scene} controls follow the native DrawControls sequence`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.controls.gold.txt`, import.meta.url)).text(),
        );
        const sequence = (
            await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text()
        )
            .trim()
            .split("\n")
            .map((line) => {
                const [step, name, value] = line.split("\t");
                return { step: Number(step), name: name!, value: Number(value) };
            });
        const app = await createApp({ defaults: false, plugins: [GeometryPlugin, IssuesPlugin] });
        try {
            const rows =
                scene === "ghostCollisions"
                    ? (
                          await Bun.file(
                              new URL(
                                  "./ghostCollisions.controls.readouts.gold.txt",
                                  import.meta.url,
                              ),
                          ).text()
                      )
                          .trim()
                          .split("\n")
                          .map((line) => JSON.parse(line))
                    : [];
            const bodies = createScene(app.world, scene),
                values = controlValues(scene);
            for (const [step, expected] of gold.entries()) {
                for (const change of sequence)
                    if (change.step === step)
                        changeControl(app.world, scene, bodies, values, change.name, change.value);
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step}`).toBe(expected.hash);
                if (rows.length) {
                    expect(readouts(app.world, "ghostCollisions"), `readouts ${step}`).toEqual(
                        rows[step],
                    );
                }
            }
        } finally {
            app.dispose();
        }
    });
