// Extracted from visualization.playwright.ts so the S3 arm can exercise the DEMOS derivation
// behaviorally (without a browser). The spec's `page.locator("iframe").evaluateAll` produces
// an array of iframe src strings; this pure function turns those into demo names. An empty
// iframe list (missing index, broken parse) produces an empty array — the spec's empty-guard
// `expect(demos.length).toBeGreaterThan(0)` reds on that, never vacuous-green.

/** Derive demo names from iframe src URLs: matches `/demos/<name>.html` and returns the names.
 *  An empty or non-matching input returns `[]` — the condition the empty-guard reds on. */
export function deriveDemosFromIframeSrcs(srcs: string[]): string[] {
    return srcs
        .map((s) => {
            const m = s.match(/\/demos\/([^/]+)\.html$/);
            return m ? m[1] : null;
        })
        .filter((s): s is string => s !== null);
}
