import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CLOUDFLARE_LOGIN_ARGS, CLOUDFLARE_LOGIN_DISCLOSURE } from "./publish-cloudflare";

setDefaultTimeout(TIMEOUT.node);

test("inline login requests only publishing scopes and discloses shared Wrangler login", () => {
    expect(CLOUDFLARE_LOGIN_ARGS).toEqual([
        "login",
        "--scopes",
        "account:read",
        "user:read",
        "workers_scripts:write",
    ]);
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("offline_access");
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("shared with other Wrangler use");
    expect(CLOUDFLARE_LOGIN_DISCLOSURE).toContain("wrangler login again");
});
