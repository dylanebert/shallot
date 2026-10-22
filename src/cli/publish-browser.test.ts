import { expect, spyOn, test } from "bun:test";
import type { Browser, LaunchOptions } from "playwright";
import { launchPublishChromium } from "./publish-verify";

test("production verification uses Chromium's sandbox when it starts", async () => {
    const browser = {} as Browser;
    const options: LaunchOptions[] = [];
    const output = spyOn(process.stderr, "write").mockImplementation(() => true);
    try {
        expect(
            await launchPublishChromium({
                async launch(option) {
                    options.push(option!);
                    return browser;
                },
            }),
        ).toBe(browser);
        expect(options.map((option) => option.chromiumSandbox)).toEqual([true]);
        expect(output).not.toHaveBeenCalled();
    } finally {
        output.mockRestore();
    }
});

test("only Chromium's No usable sandbox failure relaunches without its sandbox and reports why", async () => {
    const browser = {} as Browser;
    const options: LaunchOptions[] = [];
    const output = spyOn(process.stderr, "write").mockImplementation(() => true);
    try {
        expect(
            await launchPublishChromium({
                async launch(option) {
                    options.push(option!);
                    if (options.length === 1)
                        throw new Error(
                            "browserType.launch: [err] No usable sandbox! Ubuntu AppArmor restrictions",
                        );
                    return browser;
                },
            }),
        ).toBe(browser);
        expect(options.map((option) => option.chromiumSandbox)).toEqual([true, false]);
        expect(output).toHaveBeenCalledTimes(1);
        expect(output.mock.calls[0]![0]).toBe(
            "Verification ran without Chromium's sandbox: Chromium reported No usable sandbox!\n",
        );
    } finally {
        output.mockRestore();
    }
});

for (const message of ["Executable doesn't exist", "Chromium crashed", "Permission denied"]) {
    test(`an unrelated Chromium launch failure still fails: ${message}`, async () => {
        let calls = 0;
        const error = new Error(message);
        await expect(
            launchPublishChromium({
                async launch() {
                    calls++;
                    throw error;
                },
            }),
        ).rejects.toBe(error);
        expect(calls).toBe(1);
    });
}

test("a failed unsandboxed launch still fails", async () => {
    let calls = 0;
    const error = new Error("Chromium crashed");
    await expect(
        launchPublishChromium({
            async launch() {
                if (++calls === 1) throw new Error("No usable sandbox!");
                throw error;
            },
        }),
    ).rejects.toBe(error);
    expect(calls).toBe(2);
});
