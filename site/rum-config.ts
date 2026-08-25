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
