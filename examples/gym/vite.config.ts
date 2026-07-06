import { defineConfig } from "vite";

export default defineConfig({
    base: "./",
    server: {
        port: 3000,
    },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
    },
});
