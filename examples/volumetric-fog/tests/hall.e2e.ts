import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        hall: {
            adapter: { class: string; identity: string };
            image(): Promise<string>;
            evidence(regions: number[][]): Promise<number[]>;
            contributions(regions: number[][]): Promise<number[]>;
            walk(count: number): { before: number; after: number };
        };
    }
}

test("the window shaft, spot cone and point glow brighten haze beside their shadows in the walked hall", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
    await page.goto("/tests/stepped.html");
    await page.waitForFunction(() => window.hall);
    console.log("hall adapter", await page.evaluate(() => window.hall.adapter));
    // Paired mid-air patches across the side-on shaft, below the spot and below the far pendant.
    // Fog-on minus fog-off isolates air light from the lit wall behind each patch.
    const evidence = await page.evaluate(() =>
        window.hall.evidence([
            [575, 595, 325, 345],
            [625, 645, 325, 345],
            [735, 760, 210, 280],
            [820, 845, 210, 280],
            [510, 545, 215, 245],
            [440, 470, 215, 245],
        ]),
    );
    console.log("shaft / shade, cone / shade, glow / shade", evidence);
    for (const [i, name] of ["shaft", "cone", "glow"].entries()) {
        expect(evidence[i * 2], name).toBeGreaterThan(evidence[i * 2 + 1]! + 5);
    }
    const contributions = await page.evaluate(() =>
        window.hall.contributions([
            [575, 595, 325, 345],
            [735, 760, 210, 280],
            [510, 545, 215, 245],
        ]),
    );
    console.log("isolated shaft, cone, glow contributions", contributions);
    for (const value of contributions) expect(value).toBeGreaterThan(2);
    async function capture(name: string) {
        const image = await page.evaluate(() => window.hall.image());
        const directory = process.env.SHALLOT_CAPTURE_DIR;
        if (directory) {
            mkdirSync(directory, { recursive: true });
            writeFileSync(`${directory}/${name}.png`, Buffer.from(image.split(",")[1]!, "base64"));
        }
    }
    await capture("entrance-shaft-cone-glows");
    const walk = await page.evaluate(() => window.hall.walk(60));
    expect(walk.after).toBeLessThan(walk.before - 3);
    await capture("walked-shaft-cone");
    expect(errors).toEqual([]);
});
