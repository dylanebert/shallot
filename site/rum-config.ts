// Datadog RUM browser-app credentials — the one place a rotation edits. A RUM `clientToken` is
// public by design (it ships in every RUM customer's HTML), so committing it here is normal;
// `applicationId`/`clientToken`/`site` are placeholders until the Dogfood-org RUM application
// exists (spec: shallot-rum-slow-frame-vitals, S1 credentials ask).
export const RUM_CONFIG = {
    applicationId: "PLACEHOLDER_APPLICATION_ID",
    clientToken: "PLACEHOLDER_CLIENT_TOKEN",
    site: "PLACEHOLDER_SITE",
    service: "shallot-site",
    sessionSampleRate: 100,
    trackLongTasks: true,
    sessionReplaySampleRate: 0,
};

// Marker comment `scripts/build-site.ts` injects at the top of the RUM snippet and
// `scripts/check-site.ts`'s injection-presence clause matches on — cheaper and more specific
// than matching the CDN URL or a config field, and shared so the two never drift apart.
export const RUM_INJECTION_MARKER = "<!-- shallot: datadog rum slow-frame vitals -->";
