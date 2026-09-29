import { relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";

const root = resolve(import.meta.dir, "..");
const sourceFiles = [...new Glob("src/**/*.ts").scanSync(root)].filter(
    (file) => !file.endsWith(".d.ts") && !/\.(?:test|fixture|oracle|probes)\.ts$/.test(file),
);
const resourceType =
    /\b(?:GPU(?:Buffer|Texture|TextureView|Sampler|Device|BindGroup|CommandBuffer|ComputePipeline|RenderPipeline|PipelineLayout|BindGroupLayout)|Tgpu(?:Buffer|BindGroup|ComputePipeline|RenderPipeline|Root)|StorageFlag|UniformFlag)\b/;
const engineInfrastructure = new Set([
    "_instrumentedDevices",
    "_observedDevices",
    "_lostDevices",
    "_rawDevices",
]);
const resourceName =
    /(?:pipeline|\bpipe\b|bind(?:group)?|buffer|texture|sampler|atlas|cache|compiled|groups?)/i;
const resourceFactory =
    /\b(?:createBuffer|createTexture|createTextureView|createSampler|createBindGroup|createBindGroupLayout|createPipelineLayout|createShaderModule|createCommandEncoder|createQuerySet|createComputePipeline|createRenderPipeline)\s*\(/;
const resourceMapName =
    /(?:pipeline|\bpipe\b|bind(?:group)?|buffer|texture|sampler|atlas|cache|compiled|groups?|offscreen|targets?)/i;
const violations: string[] = [];

for (const path of sourceFiles) {
    const text = await Bun.file(resolve(root, path)).text();
    let ast: ReturnType<typeof parse>;
    try {
        ast = parse(text, {
            sourceType: "module",
            plugins: ["typescript", "decorators-legacy"],
        });
    } catch (error) {
        console.error(`${path}: ${error}`);
        throw error;
    }
    for (const statement of ast.program.body) {
        if (statement.type !== "VariableDeclaration") continue;
        for (const declaration of statement.declarations) {
            if (declaration.id.type !== "Identifier") continue;
            const name = declaration.id.name;
            // The ECS query cache contains CPU opcodes. The GPU runtime's weak collections only track
            // device diagnostics/unwrap identity; typed roots now live in State resources.
            if (path === "src/engine/ecs/query.ts" && name === "_opCache") continue;
            if (path === "src/engine/runtime/gpu.ts" && engineInfrastructure.has(name)) continue;
            if (
                declaration.init?.type === "ArrowFunctionExpression" ||
                declaration.init?.type === "FunctionExpression"
            )
                continue;
            const typeText = declaration.id.typeAnnotation
                ? text.slice(
                      declaration.id.typeAnnotation.start ?? 0,
                      declaration.id.typeAnnotation.end ?? undefined,
                  )
                : "";
            const initializerText = declaration.init
                ? text.slice(declaration.init.start ?? 0, declaration.init.end ?? undefined)
                : "";
            const directResource =
                resourceType.test(typeText) || resourceType.test(initializerText);
            const createsResource = resourceFactory.test(initializerText);
            const isMap = /new (?:Map|WeakMap|Set)\s*</.test(initializerText);
            const mapResource = isMap && resourceMapName.test(name);
            // State accessors are façades; TGSL layouts and uppercase contracts are recipes, not resources.
            if (/new Proxy\s*\(/.test(initializerText) && !createsResource) continue;
            if (/^tgpu\.bindGroupLayout\s*\(/.test(initializerText)) continue;
            if (name === name.toUpperCase() && !directResource && !createsResource && !mapResource)
                continue;
            if (
                !directResource &&
                !createsResource &&
                !mapResource &&
                !(name.startsWith("_") && resourceName.test(name))
            )
                continue;
            const line = declaration.loc?.start.line ?? statement.loc?.start.line ?? 0;
            violations.push(`${relative(root, path)}:${line}: module-level GPU state "${name}"`);
        }
    }
}

if (violations.length > 0) {
    console.error(violations.join("\n"));
    process.exit(1);
}
console.log("No module-level plugin GPU state in src.");
