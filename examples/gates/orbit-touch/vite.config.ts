import typegpu from "unplugin-typegpu/vite";

// an ejected app owns its bundler, so it declares the TGSL build transform itself — same shape as
// examples/flows/blank/vite.config.ts.
export default { plugins: [typegpu()] };
