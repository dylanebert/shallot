import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { shallot } from "@dylanebert/shallot/vite";

export default defineConfig({
    root: fileURLToPath(new URL("../../examples/loading-screen", import.meta.url)),
    plugins: [shallot()],
    server: { host: "0.0.0.0", port: 4174, strictPort: true },
});
