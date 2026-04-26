#!/usr/bin/env bun

import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join, resolve } from "path";

const name = process.argv[2];
if (!name) {
    console.error("Usage: bun create shallot <project-name>");
    process.exit(1);
}

const dir = resolve(name);
if (existsSync(dir)) {
    console.error(`Directory "${name}" already exists`);
    process.exit(1);
}

mkdirSync(dir, { recursive: true });
mkdirSync(join(dir, "src"), { recursive: true });
mkdirSync(join(dir, "public/scenes"), { recursive: true });

writeFileSync(
    join(dir, "public/icon.svg"),
    `<svg id="Shallot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
  <title>shallot icon</title>
  <defs>
    <radialGradient id="baseGradient" cx="35%" cy="30%" r="70%" fx="25%" fy="20%">
      <stop offset="0%" stop-color="#F5D4B8"/>
      <stop offset="45%" stop-color="#E8A86B"/>
      <stop offset="100%" stop-color="#B87654"/>
    </radialGradient>
  </defs>
  <g transform="rotate(35 40.0 40.0)">
    <path d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="#E8A86B"/>
    <path d="M40,6 C37,14 22,28 20,44 C20,52 28,62 36,70 C34,58 26,46 26,38 C26,26 38,12 40,6 Z" fill="#D49560"/>
    <path d="M40,6 C43,14 58,28 60,44 C60,52 52,62 44,70 C46,58 54,46 54,38 C54,26 42,12 40,6 Z" fill="#D49560"/>
    <path d="M40,8 C40,20 40,50 40,72" stroke="#6B4230" stroke-width="1" stroke-opacity="0.4" fill="none" stroke-linecap="round"/>
    <path d="M40,78 C48,70 66,60 66,46 C61,58 44,70 40,73 Z" fill="#D49560"/>
    <path d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="none" stroke="#6B4230" stroke-width="2"/>
  </g>
</svg>
`,
);

writeFileSync(
    join(dir, ".gitignore"),
    `node_modules/
dist/
build/
`,
);

writeFileSync(
    join(dir, "package.json"),
    JSON.stringify(
        {
            name,
            version: "0.0.0",
            private: true,
            type: "module",
            scripts: {
                dev: "vite",
                build: "vite build",
                preview: "vite preview",
            },
            dependencies: { "@dylanebert/shallot": "latest" },
            devDependencies: { typescript: "^5.9.3", vite: "^8.0.0" },
        },
        null,
        2,
    ) + "\n",
);

writeFileSync(
    join(dir, "tsconfig.json"),
    JSON.stringify(
        {
            compilerOptions: {
                target: "ESNext",
                module: "ESNext",
                moduleResolution: "bundler",
                strict: true,
                noEmit: true,
                skipLibCheck: true,
            },
            include: ["src"],
        },
        null,
        2,
    ) + "\n",
);

writeFileSync(
    join(dir, "index.html"),
    `<!doctype html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" type="image/svg+xml" href="/icon.svg" />
        <title>${name}</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: #0a0a0a; overflow: hidden; }
            canvas { display: block; width: 100vw; height: 100vh; }
        </style>
    </head>
    <body>
        <canvas id="canvas"></canvas>
        <script type="module" src="./src/main.ts"></script>
    </body>
</html>
`,
);

writeFileSync(
    join(dir, "vite.config.ts"),
    `import { defineConfig } from "vite";

export default defineConfig({
    server: { port: 3000 },
    build: {
        target: "esnext",
        outDir: "dist",
        emptyOutDir: true,
    },
});
`,
);

writeFileSync(
    join(dir, "src/lib.ts"),
    `import { type Config } from "@dylanebert/shallot";
import { OrbitPlugin } from "@dylanebert/shallot/extras";

export const config: Config = {
    plugins: [OrbitPlugin],
    scene: "/scenes/demo.scene",
};
`,
);

writeFileSync(
    join(dir, "src/main.ts"),
    `import { run } from "@dylanebert/shallot";
import { config } from "./lib";

const state = await run(config);

if (import.meta.hot) {
    import.meta.hot.dispose(() => state.dispose());
}
`,
);

writeFileSync(
    join(dir, "public/scenes/demo.scene"),
    `<scene>
    <a id="camera" camera fxaa orbit tonemap transform />
    <a ambient-light />
    <a directional-light />
    <a id="cube" part="size: 1; shape: box; color: 0xd49560" transform="pos: 0" />
</scene>
`,
);

console.log(`Created ${name}/`);
console.log();
console.log("Next steps:");
console.log(`  cd ${name}`);
console.log("  bun install");
console.log("  bun dev");
