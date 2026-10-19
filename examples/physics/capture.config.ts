import config from "./playwright.config";

// Run with SHALLOT_CAPTURE_DIR=<directory> bunx playwright test -c examples/physics/capture.config.ts.
export default { ...config, testMatch: "**/*.capture.ts" };
