/** The test-tier suffix roster is authoritative here; consumers derive their populations from it.
 *  Consumers of this constant include `cli-coverage.ts`'s `TEST_TIER_SUFFIXES` (excludes test-tier
 *  files from the CLI coverage population so a `*.tier.ts` isn't demanded a coverage row as
 *  production code), `standards.ts`'s shared `sourceModules()` (excludes them from both TGSL
 *  standards verdicts), and `check-exports.ts`'s `isTestFile()` (excludes them from the dead-export walk).
 *  `scripts/check-docs.ts` scans every tracked file for a literal tier-suffix roster and
 *  asserts none exists outside this constant — so the roster stops being restated. A fix that
 *  leaves two hand-written lists in agreement fails that criterion. */

/** Shared suffix names for test-tier discovery and exclusion. */
export const TEST_TIER_SUFFIX_NAMES = [
    "test",
    "oracle",
    "probes",
    "tier",
    "lab",
    "playwright",
] as const;

/** a RegExp matching any `.ts` file whose suffix is one of {@link TEST_TIER_SUFFIX_NAMES}. */
export const TEST_TIER_SUFFIXES = new RegExp(`\\.(${TEST_TIER_SUFFIX_NAMES.join("|")})\\.ts$`);
