import { mkdirSync, writeFileSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";

declare global {
    interface Window {
        hall: {
            adapter: { class: string; identity: string };
            image(): Promise<string>;
            evidence(regions: number[][]): Promise<number[]>;
            contributions(regions: number[][]): Promise<number[]>;
            sunColumns(region: number[]): Promise<number[]>;
            walk(count: number): Promise<{ before: number; after: number }>;
        };
    }
}

// Both claims observe the same built hall; the seam is read before the walk moves its player.
test.describe.configure({ mode: "serial" });
let page: Page;
const errors: string[] = [];
test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    page.on("pageerror", (error) => errors.push(error.message));
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => window.hall);
});
test.afterAll(async () => {
    await page?.close();
});

test("the far wall's top seam takes no sunlight from the air above the hall", async () => {
    // Ceiling columns at the seam end metres short of the far wall their neighbours see, and the cells between
    // lie in the sunlit air above the hall. Lit there, they streaked the seam's haze column by column: the
    // brightest pixel column took 1.6 of sunlight at b8718d34.
    const seam = await page.evaluate(() => window.hall.sunColumns([150, 208, 60, 68]));
    console.log("far seam sunlight, brightest column", Math.max(...seam));
    expect(Math.max(...seam)).toBeLessThan(0.5);
    expect(errors).toEqual([]);
});

test("the window shaft, spot cone and point glow brighten haze beside their shadows in the walked hall", async () => {
    console.log("hall adapter", await page.evaluate(() => window.hall.adapter));
    // Paired mid-air patches across the side-on shaft, below the spot and below the far pendant.
    // Fog-on minus fog-off isolates air light from the lit wall behind each patch.
    const evidence = await page.evaluate(() =>
        window.hall.evidence([
            [144, 149, 81, 86],
            [156, 161, 81, 86],
            [184, 190, 52, 70],
            [205, 211, 52, 70],
            [128, 136, 54, 61],
            [110, 118, 54, 61],
        ]),
    );
    console.log("shaft / shade, cone / shade, glow / shade", evidence);
    for (const [i, name] of ["shaft", "cone", "glow"].entries()) {
        expect(evidence[i * 2], name).toBeGreaterThan(evidence[i * 2 + 1]! + 5);
    }
    const contributions = await page.evaluate(() =>
        window.hall.contributions([
            [144, 149, 81, 86],
            [184, 190, 52, 70],
            [128, 136, 54, 61],
        ]),
    );
    console.log("isolated shaft, cone, glow contributions", contributions);
    for (const value of contributions) expect(value).toBeGreaterThan(2);
    const walk = await page.evaluate(() => window.hall.walk(60));
    expect(walk.after).toBeLessThan(walk.before - 3);
    // Captures for a person's judgment show the game's full-size view, outside the claims' frame.
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (directory) {
        const view = await page.context().newPage();
        await view.goto("/tests/stepped.html?captures");
        await view.waitForFunction(() => window.hall);
        mkdirSync(directory, { recursive: true });
        const save = async (name: string) => {
            const image = await view.evaluate(() => window.hall.image());
            writeFileSync(`${directory}/${name}.png`, Buffer.from(image.split(",")[1]!, "base64"));
        };
        await save("entrance-shaft-cone-glows");
        await view.evaluate(() => window.hall.walk(60));
        await save("walked-shaft-cone");
        await view.close();
    }
    expect(errors).toEqual([]);
});
