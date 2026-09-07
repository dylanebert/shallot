import { isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { expect, test } from "@playwright/test";

test("real handle pick, constrained edit, empty clear and reload", async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("console", (message) => {
        if (message.type() === "error" || isDegradedBootMessage(message.text()))
            errors.push(message.text());
    });
    await page.goto("/");
    await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
    const before = await page.evaluate(async () => {
        const path = "/src/harness.ts";
        const h = await import(path);
        h.poseCamera({ distance: 300, pitch: 0.7, yaw: 0, smoothness: 1 });
        await h.frames(3);
        const edit = await import("/src/" + "edit.ts");
        return {
            camera: h.cameraSnapshot(),
            edit: edit.editSnapshot(),
            positions: edit.handlePositions(),
        };
    });
    await info.attach("before", { body: JSON.stringify(before), contentType: "application/json" });
    const display = before.camera.cameras.find((c: any) => c.canvas);
    expect(before.edit.camera, "pick must use the canvas-presenting camera").toBe(display.eid);
    const screen = await page.evaluate(
        (positions) => window.__roadsProbe!(positions),
        before.positions,
    );
    const canvas = page.locator("canvas").first();
    const box = (await canvas.boundingBox())!;
    const end = screen.findIndex((p) => p.x > 0.1 && p.x < 0.9 && p.y > 0.1 && p.y < 0.9);
    expect(end, "at least one handle in frame").toBeGreaterThanOrEqual(0);
    const at = screen[end];
    await page.mouse.move(box.x + at.x * box.width, box.y + at.y * box.height);
    await expect
        .poll(() =>
            page.evaluate(async () => (await import("/src/" + "edit.ts")).editSnapshot().hovered),
        )
        .toBe(end);
    await page.mouse.down();
    await expect
        .poll(() =>
            page.evaluate(
                async () => (await import("/src/" + "edit.ts")).editSnapshot().grab.dragging,
            ),
        )
        .toBe(true);
    for (let step = 1; step <= 12; step++) {
        await page.mouse.move(
            box.x + at.x * box.width + (70 * step) / 12,
            box.y + at.y * box.height + (30 * step) / 12,
        );
        await page.evaluate(async () => (await import("/src/" + "harness.ts")).frames(2));
    }
    const held = await page.evaluate(async () => ({
        edit: (await import("/src/" + "edit.ts")).editSnapshot(),
        camera: (await import("/src/" + "harness.ts")).cameraSnapshot(),
        doc: (await import("/src/terrain/" + "terrain.ts")).getDocument(),
    }));
    console.log("EDIT_HELD", JSON.stringify({ before, screen, end, held }));
    expect(held.edit.grab.dragEnd).toBe(end);
    expect(held.edit.grab.dragging).toBe(true);
    const heldCamera = held.camera.cameras.find((c: any) => c.canvas);
    expect([heldCamera.pos, heldCamera.rot]).toEqual([display.pos, display.rot]);
    expect(held.edit.ray.origin.concat(held.edit.ray.dir).every(Number.isFinite)).toBe(true);
    await info.attach("held", { body: JSON.stringify(held), contentType: "application/json" });
    await page.mouse.up();
    await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
    const after = await page.evaluate(async () => {
        const terrain = await import("/src/terrain/" + "terrain.ts");
        const h = await import("/src/" + "harness.ts");
        await h.frames(3);
        return {
            doc: terrain.getDocument(),
            vertices: Array.from(await terrain.readVertices()),
            camera: h.cameraSnapshot(),
        };
    });
    expect(after.doc.polylines[0].points[end]).not.toEqual([
        before.positions[end][0],
        before.positions[end][2],
    ]);
    expect(after.doc.polylines[0].points.flat().every(Number.isFinite)).toBe(true);
    expect(after.vertices.length).toBeGreaterThan(0);
    expect(await page.evaluate(() => window.__roadsFlatnessViolations!())).toEqual({
        longitudinal: 0,
        crossSection: 0,
    });
    const refused = await page.evaluate(async () => {
        const terrain = await import("/src/terrain/" + "terrain.ts");
        const old = terrain.getDocument();
        try {
            await terrain.editDocument({
                polylines: [
                    {
                        points: [
                            [0, 0],
                            [NaN, 0],
                        ],
                        halfWidth: 4,
                    },
                ],
            });
            return false;
        } catch (error) {
            return String(error).includes("finite") && terrain.getDocument() === old;
        }
    });
    expect(refused, "invalid coordinates cannot replace the live document").toBe(true);
    for (const length of [0, 1]) {
        const repaired = await page.evaluate(async (length) => {
            const terrain = await import("/src/terrain/" + "terrain.ts");
            await terrain.editDocument({
                polylines: [
                    {
                        points: [
                            [0, 0],
                            [length, 0],
                        ],
                        halfWidth: 4,
                    },
                ],
            });
            await window.__roadsEdit!(1, 0, 0);
            return terrain.getDocument();
        }, length);
        const [a, b] = repaired.polylines[0].points;
        expect([a, b].flat().every(Number.isFinite)).toBe(true);
        expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBe(80);
        await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
    }
    await info.attach("after", {
        body: JSON.stringify({ doc: after.doc, camera: after.camera }),
        contentType: "application/json",
    });
    for (const polylines of [
        [],
        [{ points: [], halfWidth: 4 }],
        [{ points: [[0, 0]], halfWidth: 4 }],
    ]) {
        const cleared = await page.evaluate(async (polylines) => {
            const terrain = await import("/src/terrain/" + "terrain.ts");
            const atlas = await import("/src/overlay/" + "atlas.ts");
            const h = await import("/src/" + "harness.ts");
            await terrain.editDocument({ polylines });
            await h.frames(3);
            const bindings = atlas.bindings();
            return {
                idle: terrain.overlayIdle(),
                tiles: Array.from(await bindings.indirection.read()),
                doc: terrain.getDocument(),
            };
        }, polylines);
        expect(cleared.idle).toBe(true);
        expect(cleared.tiles.every((tile) => tile === -1)).toBe(true);
        await page.evaluate(() => window.__roadsRegenerate!(1337));
        await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
    }
    await page.reload();
    await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
    expect(errors, errors.join("\n")).toEqual([]);
});
