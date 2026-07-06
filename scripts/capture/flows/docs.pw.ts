import { defineFlow } from "../runner";
import { S } from "../selectors";

// The in-editor docs reader gate (testing.md "Editor tiers": the flow asserts behavior, the screenshot is
// the side artifact). The reader is a pure view over the generated docs/dist artifact (lib/docs.ts, already
// unit-covered); this drives the wiring the unit tier can't reach — summon as an overlay (no layout reflow,
// inspector intact behind it), search, breadcrumb navigation, dismiss, and the context-sensitive seam
// (selecting a component opens its reference). Orbit is the proven specimen: its page generates from the
// literate specimen + module source, so its slug and ref-Orbit anchor exist — and as a leaf module with
// no #doc:dev blocks it renders single-audience, no tab chrome (hardening.md "The Internals tab is opt-in").
defineFlow({ name: "docs", scene: "demo" }, async ({ page, step, assert, act }) => {
    // summon from the master menu's Docs entry (editor-ui.md: acted-on-occasionally → summoned, and an
    // occasional global action lives in the menu, not the status/view cluster)
    await act.click(S.menuBtn);
    await act.click(S.menuDocs);
    const reader = page.locator(S.docsReader);
    assert("the menu Docs entry summons the docs reader", await reader.isVisible());
    // the drawer is an overlay, not a swap — the inspector stays mounted behind it (no layout reflow)
    assert(
        "the inspector stays present behind the overlay",
        await page.locator(S.inspector).isVisible(),
    );

    // search resolves orbit's page and its reference symbols
    await act.fill(S.docsSearch, "orbit");
    const pageHit = page.locator(
        `${S.docsResult}:has(${S.docsResultTitle}:not(.symbol):text-is('Orbit'))`,
    );
    const symbolHit = page.locator(`${S.docsResultTitle}.symbol`, { hasText: "OrbitPlugin" });
    assert("search returns the Orbit page", (await pageHit.count()) === 1);
    assert("search returns reference symbols (OrbitPlugin)", (await symbolHit.count()) >= 1);
    await step("docs-search", { highlight: [S.docsResult], clip: S.docsOverlay });

    // navigate to the page: title renders, and a leaf module shows no tab chrome (single-audience)
    await act.click(pageHit);
    assert(
        "the Orbit page header reads its title",
        (await page.locator(`${S.docsPageHead} h1`).innerText()).trim() === "Orbit",
    );
    assert("a leaf module renders no tab chrome", (await page.locator(S.docsTab).count()) === 0);
    await step("docs-page", { clip: S.docsOverlay });

    // the breadcrumb is the sole within-docs nav: `Docs` returns to the browse index
    await act.click(page.locator(S.docsCrumb, { hasText: "Docs" }));
    assert(
        "the breadcrumb returns to the index",
        (await page.locator(S.docsPageHead).count()) === 0,
    );
    assert("the index lists doc groups", (await page.locator(S.docsGroup).count()) >= 1);

    // dismiss (the header ✕) leaves docs entirely — the inspector is exactly where it was
    await act.click(page.locator(`${S.docsReader} button[aria-label='Close docs']`));
    assert("closing dismisses the overlay", (await reader.count()) === 0);
    assert("the inspector is untouched underneath", await page.locator(S.inspector).isVisible());

    // context-sensitive help: selecting the orbit component surfaces its reference via docFor
    await act.click(page.locator(S.row, { hasText: "camera" }));
    const orbitDocs = page.locator(
        `${S.inspector} .section:has(${S.sectionLabel}:text-is('orbit')) ${S.sectionDocs}`,
    );
    assert("the orbit section exposes a docs affordance", (await orbitDocs.count()) === 1);

    await act.click(orbitDocs);
    assert("the component's docs link summons the reader", await reader.isVisible());
    // it opens the orbit page (the rendered ref-Orbit anchor proves the reference is inline), landing at
    // the top rather than deep-scrolled to the bottom reference entry
    const refAnchor = page.locator(`${S.docsReader} #ref-Orbit`);
    await refAnchor.waitFor({ timeout: 4000 });
    assert("the Orbit reference is on the page", await refAnchor.isVisible());
    await step("docs-reference", {
        highlight: [`${S.docsReader} #ref-Orbit`],
        clip: S.docsOverlay,
    });
});
