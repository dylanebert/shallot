import { expect, test } from "bun:test";
import { cleanupOracleWorkers } from "./publish-oracle-cleanup";

for (const passed of [true, false]) {
    test(`a ${passed ? "successful" : "failed"} trial ${passed ? "deletes" : "retains"} its Workers`, () => {
        const deleted: string[] = [];
        const reports: string[] = [];
        cleanupOracleWorkers(
            ["trial", "collision", "trial"],
            passed,
            (name) => {
                deleted.push(name);
                return { status: 0, stderr: "" };
            },
            (message) => reports.push(message),
        );
        expect(deleted).toEqual(passed ? ["trial", "collision"] : []);
        expect(reports).toEqual(
            passed
                ? ["Deleted Worker trial.", "Deleted Worker collision."]
                : [
                      "Retained Worker for inspection: wrangler delete --name trial",
                      "Retained Worker for inspection: wrangler delete --name collision",
                  ],
        );
    });
}

test("a failed deletion reports the command, attempts remaining names and fails the trial", () => {
    const deleted: string[] = [];
    const reports: string[] = [];
    expect(() =>
        cleanupOracleWorkers(
            ["trial", "collision"],
            true,
            (name) => {
                deleted.push(name);
                return name === "trial"
                    ? { status: 1, stderr: "permission denied" }
                    : { status: 0, stderr: "" };
            },
            (message) => reports.push(message),
        ),
    ).toThrow("trial: permission denied");
    expect(deleted).toEqual(["trial", "collision"]);
    expect(reports).toEqual([
        "Could not delete Worker: wrangler delete --name trial",
        "Deleted Worker collision.",
    ]);
});
