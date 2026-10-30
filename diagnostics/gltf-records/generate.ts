// Regenerate with THREE=/path/to/three.js bun diagnostics/gltf-records/generate.ts;
// THREE must be a checkout at 1e7e3cff304a05416a54343e7ce273c709bccefe.
// Audit values against the record's named glTF clauses before accepting a changed record.
// For a clause-versus-loader divergence, add the field, clause, loader value and specified
// replacement to that fixture's divergences, and implement the replacement here.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { fixtures, recordArray } from "./format";

const revision = "1e7e3cff304a05416a54343e7ce273c709bccefe";
const checkout = process.env.THREE;
if (!checkout) throw new Error(`set THREE to a three.js checkout at ${revision}`);
const head = Bun.spawnSync(["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.toString().trim();
if (head !== revision) throw new Error(`THREE is at ${head || "no git revision"}, not ${revision}`);
const { GLTFLoader } = await import(pathToFileURL(resolve(checkout, "examples/jsm/loaders/GLTFLoader.js")).href);
const { LoadingManager, Mesh, MeshBasicMaterial, DoubleSide } = await import(pathToFileURL(resolve(checkout, "build/three.module.js")).href);

// Pixels are not recorded. The loader still resolves texture identity and sampler state.
globalThis.self = globalThis as typeof self;
globalThis.ProgressEvent = class extends Event {} as unknown as typeof ProgressEvent;
globalThis.createImageBitmap = (async () => ({ width: 1, height: 1, close() {} })) as typeof createImageBitmap;
const wrap = (value: number) => ({ 1000: 1, 1001: 0, 1002: 2 })[value];
const filter = (value: number) => [1003, 1004, 1005].includes(value) ? 0 : 1;
const records = [];
for (const fixture of fixtures) {
    const path = resolve(import.meta.dir, "../../src/extras/gltf/fixtures", fixture);
    const manager = new LoadingManager();
    manager.setURLModifier((url: string) => url.startsWith("data:") || url.startsWith("blob:") ? url :
        `data:application/octet-stream;base64,${readFileSync(resolve(dirname(path), url)).toString("base64")}`);
    const bytes = readFileSync(path);
    const reference = await new GLTFLoader(manager).parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
    reference.scene.updateMatrixWorld(true);
    const instances: any[] = [];
    reference.scene.traverse((object: any) => { if (object instanceof Mesh) instances.push(object); });
    const textures = [];
    for (let i = 0; i < (reference.parser.json.textures?.length ?? 0); i++) {
        const texture = await reference.parser.getDependency("texture", i);
        textures.push({ image: reference.parser.json.textures[i].source, sampler: {
            addressModeU: wrap(texture.wrapS), addressModeV: wrap(texture.wrapT),
            magFilter: filter(texture.magFilter), minFilter: filter(texture.minFilter),
            mipmapFilter: [1003, 1004, 1006, 1007].includes(texture.minFilter) ? 0 : 1,
        } });
    }
    records.push({ fixture, textures, instances: instances.map((object) => {
        let geometry = object.geometry;
        const implicitNormals = !geometry.getAttribute("normal");
        if (implicitNormals) {
            // meshes-usage: one normal per triangle. GLTFLoader leaves these implicit
            // (flatShading); compute on split corners, never average shared vertices.
            geometry = geometry.index ? geometry.toNonIndexed() : geometry.clone();
            geometry.computeVertexNormals();
        }
        const position = geometry.getAttribute("position");
        const normal = geometry.getAttribute("normal");
        const uv = geometry.getAttribute("uv");
        const positions = [], normals = [], uvs = [];
        for (let i = 0; i < position.count; i++) {
            positions.push(position.getX(i), position.getY(i), position.getZ(i));
            normals.push(normal.getX(i), normal.getY(i), normal.getZ(i));
            uvs.push(uv ? uv.getX(i) : 0, uv ? uv.getY(i) : 0);
        }
        const ref = object.material;
        const unlit = ref instanceof MeshBasicMaterial;
        const materialIndex = reference.parser.associations.get(ref)?.materials;
        const material = materialIndex === undefined ? null : {
            color: [...ref.color.toArray(), ref.opacity],
            ...(!unlit ? { metallic: ref.metalness, roughness: ref.roughness, emissive: ref.emissive.toArray() } : {}),
            alphaMode: reference.parser.json.materials[materialIndex].alphaMode ?? "OPAQUE",
            ...(reference.parser.json.materials[materialIndex].alphaMode === "MASK" ? { cutoff: ref.alphaTest } : {}),
            textureImage: ref.map ? reference.parser.json.textures[reference.parser.associations.get(ref.map).textures].source : null,
            unlit, doubleSided: ref.side === DoubleSide,
        };
        return { matrix: object.matrixWorld.elements, implicitNormals,
            positions: recordArray(positions, 3), normals: recordArray(normals, 3), uvs: recordArray(uvs, 2),
            indices: recordArray(geometry.index?.array ?? Array.from({ length: position.count }, (_, i) => i), 1, true),
            material,
        };
    }), divergences: [] });
}
writeFileSync(resolve(import.meta.dir, "../../src/extras/gltf/fixtures/records.json"), `${JSON.stringify({
    specification: "a55d8458c1e82720413acd3b420fd09e73b8e40b", generator: revision,
    arrayFormat: "At most 128 components: exact JSON values. Otherwise: element count, per-component min/max, SHA-256 of little-endian float32 attributes or uint32 indices, without rounding.",
    defaults: "material.schema.json; material.pbrMetallicRoughness.schema.json; sampler.schema.json; texture.schema.json; node.schema.json at specification revision",
    samplerEnums: "address modes: clamp=0, repeat=1, mirror=2; filters: nearest=0, linear=1",
    scope: "Mapped static fields only; image pixels excluded. Null material means no authored material (not asserted); lighting factors omitted for unlit. UV zero is a storage placeholder, not a specification default.",
    clauses: {
        instances: "scenes-overview; nodes-and-hierarchy; geometry-overview; meshes-overview (one record per instantiated primitive; depth-first order is a comparison convention)",
        material: "meshes-overview; default-material (null denotes no authored material)",
        implicitNormals: "meshes-usage (NORMAL absent)",
        positions: "accessors-overview; accessor-data-types; meshes-attributes (POSITION)",
        normals: "meshes-attributes (NORMAL); meshes-usage (implicit flat normals)",
        uvs: "accessors-overview; accessor-data-types; meshes-attributes (TEXCOORD_0); absent UVs use Shallot's zero placeholder",
        indices: "meshes-topology (indexed order; sequential vertices if indices absent); meshes-usage (split corners for flat normals)",
        matrix: "transformations (TRS; parent global times local)",
        color: "metallic-roughness-material (linear baseColorFactor); alpha-coverage (fourth component)",
        metallic: "metallic-roughness-material (metallicFactor)", roughness: "metallic-roughness-material (roughnessFactor)",
        emissive: "emissive (linear emissiveFactor)", alphaMode: "alpha-coverage", cutoff: "alpha-coverage (MASK only)",
        textureImage: "materials-overview (texture index); textures (source image)",
        unlit: "KHR_materials_unlit: Extending Materials (at specification revision)", doubleSided: "double-sided",
        textures: "textures (source); samplers (Filtering; Wrapping); absent filters choose linear / linear-mipmap-linear, permitted by Samplers Overview",
    }, records,
}, null, 2)}\n`);
console.log(`Recorded ${records.length} fixtures from ${head}`);
