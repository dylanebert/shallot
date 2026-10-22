import { expect, test } from "bun:test";
import { cleanupOracleWorkers } from "./publish-oracle-cleanup";

for (const passed of [true, false]) {
    test(`a ${passed ? "successful" : "failed"} trial ${passed ? "deletes" : "retains"} its Workers`, () => {
        const deleted: string[] = [];
        const reports: string[] = [];
        cleanupOracleWorkers(
            ["trial", "collision", "trial"],
            passed,
            "account-a",
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
                      "Retained Worker for inspection: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name trial",
                      "Retained Worker for inspection: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name collision",
                  ],
        );
    });
}

test("an unresolved account never prints or executes an unbound delete", () => {
    const reports: string[] = [];
    cleanupOracleWorkers(
        ["trial"],
        false,
        undefined,
        () => {
            throw new Error("must not delete");
        },
        (message) => reports.push(message),
    );
    expect(reports).toEqual([
        "Retained Worker trial: account unresolved; resolve its account before deleting.",
    ]);
});

test("a failed deletion reports the command, attempts remaining names and fails the trial", () => {
    const deleted: string[] = [];
    const reports: string[] = [];
    expect(() =>
        cleanupOracleWorkers(
            ["trial", "collision"],
            true,
            "account-a",
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
        "Could not delete Worker: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name trial",
        "Deleted Worker collision.",
    ]);
});
