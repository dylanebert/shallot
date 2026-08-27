// Datadog RUM browser-app credentials — the one place a rotation edits. A RUM `clientToken` is
// public by design (it ships in every RUM customer's HTML), so committing it here is normal.
// The application lives in the Dogfood org (ID 1975629), app name shallot-site.
export const RUM_CONFIG = {
    applicationId: "a5ffefd9-8f32-49dd-b8ae-ff31988e6bfe",
    clientToken: "pube2bd4a743b9fd4d05457e14b56fa53c5",
    site: "datadoghq.com",
    service: "shallot-site",
    sessionSampleRate: 100,
    trackLongTasks: true,
    sessionReplaySampleRate: 0,
};

// Marker comment `scripts/build-site.ts` injects at the top of the RUM snippet and
// `scripts/check-site.ts`'s injection-presence clause matches on — cheaper and more specific
// than matching the CDN URL or a config field, and shared so the two never drift apart.
export const RUM_INJECTION_MARKER = "<!-- shallot: datadog rum slow-frame vitals -->";

// Plain, dependency-free inline JS — runs in the visitor's browser before `DD_RUM.init`, so it
// cannot import anything. Derives the Datadog `env` facet from the served hostname: "prod" for
// dylanebert.com and any subdomain (the site is served at dylanebert.com/shallot/, README.md /
// AGENTS.md), "local" otherwise (a `bun run verify`/dev preview at localhost, or a raw
// `file://`/IP checkout) — so a localhost preview is tagged "local" and never pollutes prod's
// slow-frame vitals. One shared string: `scripts/build-site.ts` injects
// it verbatim and `scripts/check-site.ts`'s env-presence clause matches on the same literal, so
// the two never drift apart (mirrors `RUM_INJECTION_MARKER` above).
export const RUM_ENV_SNIPPET =
    "var ddEnv=/(^|\\.)dylanebert\\.com$/.test(location.hostname)?'prod':'local';";

// The `env` wiring into `DD_RUM.init` — the derivation line above (`RUM_ENV_SNIPPET`) computes
// `ddEnv` but does nothing until it's spread into the init call. `scripts/build-site.ts` opens
// its `Object.assign(...)` call with this exact literal; `scripts/check-site.ts`'s clause 5
// matches on it separately from `RUM_ENV_SNIPPET`, so a mutation that drops the derivation line
// and one that drops the wiring (leaving `ddEnv` computed but never read) each red on their own
// clause instead of one silently covering for the other.
export const RUM_ENV_USAGE = "Object.assign({env:ddEnv},";
