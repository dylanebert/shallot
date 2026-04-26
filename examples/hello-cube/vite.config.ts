import { defineConfig } from "vite";

export default defineConfig({
    server: {
        port: 3000,
    },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
    },
});
