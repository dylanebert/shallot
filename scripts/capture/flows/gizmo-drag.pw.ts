import { defineFlow } from "../runner";
import { S } from "../selectors";

// The viewport-gizmo + multi-select wiring gate (testing.md "Editor tiers": "a gizmo drag moves the
// entity"). The drag math, the gesture/undo substrate, the selection rule, and the multi-section merge are
// unit-tested (gizmo / pick / sections / pivot .test.ts); this drives the live editor and asserts what the
// unit tier can't reach end to end: the entity moves under the cursor, a multi-entity drag commits as ONE
// undoable gesture (one undo restores every selected entity), Rotate (rings, incl. the screen-facing
// trackball) and Scale (uniform) ride the same handle core, shift-click toggles the outliner selection, an
// inspector field edit fans out to every selected entity as one undo, and the pivot hotkey cycles.
//
// The fixture's `box` sits at the world origin, so the gizmo (selection centroid) projects to ≈ the canvas
// centre. The centre handle there is the screen / uniform-centre handle; the rotation rings project around
// it (their on-screen radius scales with the view, so the rotate leg sweeps a few radii to find the rim).
// Gizmo legs drive selection through the shallot:request control API (the seam an agent uses, multi-id);
// the multi-select leg drives a real modifier-click through the outliner.
const PORT = process.env.CAPTURE_PORT || "3004";
const API = `http://localhost:${PORT}/__api`;

// the entity's `transform` attr value out of the serialized scene; "" if absent. transform values hold no
// double-quotes (just `;` + spaces), so the field is a clean `[^"]*` slice.
function transformOf(scene: string, id: string): string {
    const el = scene.match(new RegExp(`<a id="${id}"[\\s\\S]*?/>`));
    const t = el?.[0].match(/transform="([^"]*)"/);
    return t ? t[1] : "";
}

defineFlow({ name: "gizmo-drag", scene: "demo" }, async ({ page, step, act, assert }) => {
    const canvas = page.locator(S.canvas);
    const box = (await canvas.boundingBox())!;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    const scene = async (): Promise<string> =>
        (await (await page.request.get(`${API}/state`)).json()).scene;
    // the control-API `select` command is additive (compose with clearSelection — it mirrors the doc.select
    // primitive), so set the selection to exactly `ids` by clearing first; otherwise an earlier multi-select
    // bleeds through and the gizmo pivot (selection median) drifts off the lone entity.
    const select = async (...ids: string[]): Promise<void> => {
        await page.request.post(`${API}/command`, { data: { method: "clearSelection", args: {} } });
        await page.request.post(`${API}/command`, { data: { method: "select", args: { ids } } });
        await act.wait(150);
    };
    const grabDrag = async (dx: number, dy: number): Promise<void> => {
        await page.mouse.move(cx, cy);
        await page.mouse.down();
        await page.mouse.move(cx + dx, cy + dy, { steps: 8 });
        await page.mouse.up();
        await act.wait(200);
    };

    // ---- Move: a multi-entity drag commits as one undoable gesture ----
    await select("ground", "box");
    await act.key("2"); // Move tool (number-row scheme; lib/tool.ts)
    await act.wait(150);

    const groundBefore = transformOf(await scene(), "ground");
    const boxBefore = transformOf(await scene(), "box");
    await grabDrag(90, -40);
    await step("moved", { highlight: [S.viewport, S.outliner] });

    assert("the gizmo drag moved the box", transformOf(await scene(), "box") !== boxBefore);
    assert(
        "the same drag moved the other selected entity",
        transformOf(await scene(), "ground") !== groundBefore,
    );

    // one undo restores BOTH — the gesture coalesced the multi-entity move into a single history entry
    await act.key("Control+z");
    await act.wait(200);
    assert("one undo restores the box", transformOf(await scene(), "box") === boxBefore);
    assert(
        "the same undo restores the other entity (one gesture, one entry)",
        transformOf(await scene(), "ground") === groundBefore,
    );

    // ---- Rotate: the ring manipulator (quaternion drag) ----
    await select("box");
    await act.key("3"); // Rotate tool
    await act.wait(150);

    // The horizontal Y ring (XZ-plane circle) reads as a wide ellipse, so its rim crosses the line straight
    // right of centre. The on-screen radius isn't fixed (the gizmo scales with the view), so try a few radii
    // along that line and take the first that grabs the rim — robust to the gizmo's screen size. A miss
    // leaves the transform untouched, so only the grabbing attempt changes it (one undo restores it).
    const rotBefore = transformOf(await scene(), "box");
    let rotated = false;
    // dense radii (step < the 8px pick threshold) so one lands on a ring rim whatever its on-screen size;
    // at cy the rings all cross the horizontal line, so some radius hits. The outer radii cross the
    // screen-facing trackball ring (SCREEN_RING_SCALE, outside the axis rings), so this leg also exercises
    // the trackball handle through the same generic pick/drag driver.
    for (let r = 55; r <= 125 && !rotated; r += 7) {
        await page.mouse.move(cx + r, cy);
        await page.mouse.down();
        await page.mouse.move(cx + r, cy + 55, { steps: 8 }); // tangential sweep around the rim
        await page.mouse.up();
        await act.wait(120);
        if (transformOf(await scene(), "box") !== rotBefore) rotated = true;
    }
    await step("rotated", { highlight: [S.viewport] });
    assert("a rotation ring changed the box transform", rotated);

    await act.key("Control+z");
    await act.wait(200);
    assert("one undo restores the rotation", transformOf(await scene(), "box") === rotBefore);

    // ---- Scale: the uniform-centre handle ----
    await select("box");
    await act.key("4"); // Scale tool
    await act.wait(150);

    // the box starts at scale 1.8 (fixture). A sideways drag of the centre cube reads as a uniform scale in
    // any direction, so it must GROW the box by a real margin — a bare "transform changed" check passes on an
    // imperceptible delta, so assert the magnitude
    const scaleAxisOf = (t: string): number => Number(t.match(/scale:\s*([\d.]+)/)?.[1] ?? "1");
    const scaleBefore = transformOf(await scene(), "box");
    await grabDrag(100, 0);
    await step("scaled", { highlight: [S.viewport] });
    const scaledBox = transformOf(await scene(), "box");
    assert(
        "the sideways scale drag grows the box uniformly by a visible margin (not ≈1.0)",
        scaleAxisOf(scaledBox) > scaleAxisOf(scaleBefore) * 1.2,
    );
    // uniform: all three axes scale together (an axis grab would move only one)
    const scaleTriple = scaledBox.match(/scale:\s*([\d.]+) ([\d.]+) ([\d.]+)/);
    assert(
        "the scale stays uniform — x and z grew with y",
        !!scaleTriple &&
            Math.abs(Number(scaleTriple[1]) - Number(scaleTriple[2])) < 1e-3 &&
            Math.abs(Number(scaleTriple[1]) - Number(scaleTriple[3])) < 1e-3,
    );

    await act.key("Control+z");
    await act.wait(200);
    assert("one undo restores the scale", transformOf(await scene(), "box") === scaleBefore);

    // ---- Outliner multi-select: shift-click toggles a row in/out of the set (lib/pick nextSelection) ----
    const rows = page.locator(S.row);
    await rows.nth(0).click();
    await act.wait(120);
    await rows.nth(1).click({ modifiers: ["Shift"] });
    await act.wait(120);
    assert(
        "shift-click adds a second row to the selection",
        (await page.locator(S.rowSelected).count()) === 2,
    );
    await step("multi-select", { highlight: [S.outliner] });
    // wait past the outliner's 400ms double-click-to-rename window before re-clicking the same row, so the
    // second shift-click reads as a toggle (a fast re-click of one row is a rename gesture, not a select)
    await act.wait(450);
    await rows.nth(1).click({ modifiers: ["Shift"] });
    await act.wait(120);
    assert(
        "shift-click again toggles that row back out (one remains)",
        (await page.locator(S.rowSelected).count()) === 1,
    );

    // ---- Inspector multi-edit: one field edit fans out to every selected entity, one undo ----
    // select ground then shift-add box (both carry transform), so the shared `transform` section shows
    await act.click(page.locator(S.row, { hasText: "ground" }));
    await act.wait(150);
    await page.locator(S.row, { hasText: "box" }).click({ modifiers: ["Shift"] });
    await act.wait(150);
    assert(
        "both entities are selected for the multi-edit",
        (await page.locator(S.rowSelected).count()) === 2,
    );
    const gEditBefore = transformOf(await scene(), "ground");
    const bEditBefore = transformOf(await scene(), "box");
    // the transform `pos` field's first lane (the same field edit-transform.pw.ts drives, single-select)
    const posX = page
        .locator(`${S.fieldRow}:has(${S.fieldLabel}:text('pos'))`)
        .locator(S.fieldInput)
        .first();
    await act.fill(posX, "7");
    await posX.press("Enter"); // the field commits on the change event (blur / Enter), not on input
    await act.wait(200);
    await step("multi-edit", { highlight: [S.inspector] });
    assert(
        "the inspector edit reaches the first selected entity",
        transformOf(await scene(), "ground") !== gEditBefore,
    );
    assert(
        "the same edit fans out to the other selected entity",
        transformOf(await scene(), "box") !== bEditBefore,
    );
    await act.key("Control+z");
    await act.wait(200);
    assert(
        "one undo restores both edited entities (one gesture)",
        transformOf(await scene(), "box") === bEditBefore &&
            transformOf(await scene(), "ground") === gEditBefore,
    );

    // ---- Pivot dropdown: the hotkey cycles Median ↔ Active (lib/pivot) ----
    const pivotBtn = page.locator('.viewport-btn[title^="Pivot:"]');
    assert(
        "the pivot control starts on Median",
        /Median/.test((await pivotBtn.getAttribute("title")) ?? ""),
    );
    await page.locator(S.canvas).click({ position: { x: 4, y: 4 } }); // move focus off the inspector input
    await act.key(".");
    await act.wait(120);
    assert(
        "the pivot hotkey switches to Active",
        /Active/.test((await pivotBtn.getAttribute("title")) ?? ""),
    );
});
