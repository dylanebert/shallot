import config from "./playwright.config";

// Run with SHALLOT_CAPTURE_DIR=<directory> bunx playwright test -c scripts/phone/capture.config.ts.
export default {
    ...config,
    metadata: { captureDirectory: process.env.SHALLOT_CAPTURE_DIR ?? ".artifacts/phone" },
};
