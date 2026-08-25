/** The test-tier suffix roster — one exported constant, the single place the tier list lives.
 *  Derived from `testing.md`'s tier-section bullet ledes (`.test.ts`, `.oracle.ts`, `.probes.ts`,
 *  `.tier.ts`, `.lab.ts`, `.playwright.ts`), which is the enumeration `testing.md` itself makes — not
 *  the section heading (which once under-named `.probes.ts`; the heading now agrees with its own body).
 *
 *  Consumers of this constant include `cli-coverage.ts`'s `TEST_TIER_SUFFIXES` (excludes test-tier
 *  files from the CLI coverage population so a `*.tier.ts` isn't demanded a coverage row as
 *  production code), `standards.test.ts`'s `sourceModules()` (excludes them from the TGSL kernel
 *  walk), and `check-exports.ts`'s `isTestFile()` (excludes them from the dead-export walk).
 *  `scripts/check-docs.ts`'s arm (c) scans every tracked file for a literal tier-suffix roster and
 *  asserts none exists outside this constant — so the roster stops being restated. A fix that
 *  leaves two hand-written lists in agreement fails that criterion. */

/** the six test-tier suffix names `testing.md`'s tier-section bullet ledes name, in the order the
 *  bullets list them. The source of truth is the bullet ledes, not the section heading. */
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
