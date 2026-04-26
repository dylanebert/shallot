import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
    server: {
        port: 3000,
    },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: resolve(__dirname, "index.html"),
                hierarchy: resolve(__dirname, "demos/hierarchy.html"),
                lines: resolve(__dirname, "demos/lines.html"),
                text: resolve(__dirname, "demos/text.html"),
                particles: resolve(__dirname, "demos/particles.html"),
                tween: resolve(__dirname, "demos/tween.html"),
            },
        },
    },
});
