import { expect, test } from "bun:test";
import { CLOUDFLARE_LOGIN_ARGS, CLOUDFLARE_LOGIN_DISCLOSURE } from "./publish-cloudflare";
import { PUBLISH_LAUNCH_OPTIONS } from "./publish-verify";

test("inline login requests only publishing scopes and discloses shared Wrangler login", () => {
    expect(CLOUDFLARE_LOGIN_ARGS).toEqual([
        "login", "--scopes", "account:read", "user:read", "workers_scripts:write",
    ]);
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("offline_access");
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("shared with other Wrangler use");
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("wrangler login again");
});

test("production Chromium enables its sandbox and never passes --no-sandbox", () => {
    expect(PUBLISH_LAUNCH_OPTIONS.chromiumSandbox).toBe(true);
    expect(PUBLISH_LAUNCH_OPTIONS.args).not.toContain("--no-sandbox");
});
