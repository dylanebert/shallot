// Regenerate with THREE=/path/to/three.js bun diagnostics/gltf-records/generate.ts;
// THREE must be a checkout at 1e7e3cff304a05416a54343e7ce273c709bccefe.
// With --example, sync examples/gltf/assets.json and write its models.gold.json instead.
// Audit values against the record's named glTF clauses before accepting a changed record.
// For a clause-versus-loader divergence, add the field, clause, loader value and specified
// replacement to that fixture's divergences, and implement the replacement here.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { fixtures, recordArray } from "./format";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";

const example = process.argv.includes("--example");
const subjects = example ? manifest.models.map(model => `Models/${model.name}/glTF/${model.name}.gltf`) : fixtures;
const array = example ? (values: ArrayLike<number>, _width: number, _integer = false) => Array.from(values) : recordArray;

const revision = "1e7e3cff304a05416a54343e7ce273c709bccefe";
const checkout = process.env.THREE;
if (!checkout) throw new Error(`set THREE to a three.js checkout at ${revision}`);
const head = Bun.spawnSync(["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.toString().trim();
if (head !== revision) throw new Error(`THREE is at ${head || "no git revision"}, not ${revision}`);
if (example) await syncFiles(manifest.models.flatMap(model => model.files), new URL(`${manifest.commit}/`, manifest.source), resolve(import.meta.dir, "../../examples/gltf/public/assets"));
const { GLTFLoader } = await import(pathToFileURL(resolve(checkout, "examples/jsm/loaders/GLTFLoader.js")).href);
const { LoadingManager, Mesh, MeshBasicMaterial, DoubleSide } = await import(pathToFileURL(resolve(checkout, "build/three.module.js")).href);

// Pixels are not recorded. The loader still resolves texture identity and sampler state.
globalThis.self = globalThis as typeof self;
globalThis.ProgressEvent = class extends Event {} as unknown as typeof ProgressEvent;
globalThis.createImageBitmap = (async () => ({ width: 1, height: 1, close() {} })) as typeof createImageBitmap;
const wrap = (value: number) => ({ 1000: 1, 1001: 0, 1002: 2 })[value];
const filter = (value: number) => [1003, 1004, 1005].includes(value) ? 0 : 1;
const records = [];
for (const fixture of subjects) {
    const path = resolve(import.meta.dir, example ? "../../examples/gltf/public/assets" : "../../src/extras/gltf/fixtures", fixture);
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
    const divergences: any[] = [];
    const decodedInstances = await Promise.all(instances.map(async (object) => {
        let geometry = object.geometry;
        const association = reference.parser.associations.get(object);
        const primitive = reference.parser.json.meshes[association.meshes].primitives[association.primitives];
        if (primitive.mode === 5 || primitive.mode === 6) {
            const source = primitive.indices === undefined
                ? Array.from({ length: geometry.getAttribute("position").count }, (_, i) => i)
                : Array.from((await reference.parser.getDependency("accessor", primitive.indices)).array) as number[];
            const specified: number[] = [];
            for (let i = 0; i < source.length - 2; i++) {
                const triangle = primitive.mode === 5
                    ? [i, i + 1 + i % 2, i + 2 - i % 2]
                    : [i + 1, i + 2, 0];
                specified.push(...triangle.map(index => source[index]));
            }
            divergences.push({
                field: `instances[${instances.indexOf(object)}].indices`,
                clause: "meshes-topology: Triangle Strips / Triangle Fans (ordered vertex equations)",
                reason: "GLTFLoader cyclically rotates odd strip triangles and every fan triangle. Winding agrees, but the specification's vertex order is retained in this record.",
                loader: Array.from(geometry.index.array), specified,
            });
            geometry = geometry.clone();
            geometry.setIndex(specified);
        }
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
        const stream = (name: string, width: number) => {
            const attribute = geometry.getAttribute(name);
            if (!attribute) return null;
            const values: number[] = [];
            for (let i = 0; i < position.count; i++)
                for (let c = 0; c < width; c++) values.push(c < attribute.itemSize ? attribute.getComponent(i, c) : 1);
            return array(values, width);
        };
        let tangent = stream("tangent", 4);
        if (implicitNormals && tangent) {
            divergences.push({
                field: `instances[${instances.indexOf(object)}].tangent`,
                clause: "meshes-usage: provided tangents are ignored when normals are not specified",
                reason: "GLTFLoader retains the tangent attribute without NORMAL; the record omits it.",
                loader: tangent, specified: null,
            });
            tangent = null;
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
            ...(example ? { textureIndex: reference.parser.json.materials[materialIndex].pbrMetallicRoughness?.baseColorTexture?.index ?? null } : {}),
            unlit, doubleSided: ref.side === DoubleSide,
        };
        const nodePath = [];
        if (example) {
            for (let node = object; node; node = node.parent) {
                const index = reference.parser.associations.get(node)?.nodes;
                if (index !== undefined) {
                    const { matrix, translation, rotation, scale } = reference.parser.json.nodes[index];
                    nodePath.unshift({ ...(matrix ? { matrix } : {}), ...(translation ? { translation } : {}), ...(rotation ? { rotation } : {}), ...(scale ? { scale } : {}) });
                }
            }
        }
        return { frontFace: object.matrixWorld.determinant() < 0 ? "cw" : "ccw", matrix: object.matrixWorld.elements, ...(example ? { nodePath } : {}), implicitNormals,
            positions: array(positions, 3), normals: array(normals, 3), uvs: array(uvs, 2),
            uv1: stream("uv1", 2), color: stream("color", 4), tangent,
            indices: array(geometry.index?.array ?? Array.from({ length: position.count }, (_, i) => i), 1, true),
            material,
        };
    }));
    records.push({ fixture, textures, ...(example ? { images: (reference.parser.json.images ?? []).map((image: any) => image.uri) } : {}), instances: decodedInstances, divergences });
}
writeFileSync(resolve(import.meta.dir, example ? "../../examples/gltf/tests/models.gold.json" : "../../src/extras/gltf/fixtures/records.json"), `${JSON.stringify({
    specification: "a55d8458c1e82720413acd3b420fd09e73b8e40b", generator: revision,
    arrayFormat: example ? "Exact JSON arrays, before mesh registration and quantization." : "At most 128 components: exact JSON values. Otherwise: element count, per-component min/max, SHA-256 of little-endian float32 attributes or uint32 indices, without rounding.",
    defaults: "material.schema.json; material.pbrMetallicRoughness.schema.json; sampler.schema.json; texture.schema.json; node.schema.json at specification revision",
    samplerEnums: "address modes: clamp=0, repeat=1, mirror=2; filters: nearest=0, linear=1",
    scope: example ? "Mapped static fields only; image pixels excluded. Null material denotes the specification's default material; lighting factors omitted for unlit. UV zero is a storage placeholder, not a specification default." : "Mapped static fields only; image pixels excluded. Null material means no authored material (not asserted); lighting factors omitted for unlit. UV zero is a storage placeholder, not a specification default.",
    clauses: {
        instances: "scenes-overview; nodes-and-hierarchy; geometry-overview; meshes-overview (one record per instantiated primitive; depth-first order is a comparison convention)",
        material: "meshes-overview; default-material (null denotes no authored material)",
        implicitNormals: "meshes-usage (NORMAL absent)",
        positions: "accessors-overview (bufferView undefined: elements initialized with zeros; sparse MAY override); accessor-data-types; sparse-accessors (replacements over base values); meshes-attributes (POSITION)",
        normals: "meshes-attributes (NORMAL); meshes-usage (implicit flat normals)",
        uvs: "accessors-overview; accessor-data-types; meshes-attributes (TEXCOORD_0); absent UVs use Shallot's zero placeholder",
        uv1: "accessors-overview; accessor-data-types (normalized unsigned integers); meshes-attributes (TEXCOORD_1); null means absent",
        vertexColor: "meshes-attributes (COLOR_0, linear RGB/RGBA); meshes-usage (VEC3 alpha is 1); accessor-data-types (normalized unsigned integers); null means absent",
        tangent: "meshes-attributes (authored TANGENT xyzw, w unchanged); meshes-usage (ignored without NORMAL); null means absent, generated tangents are checked against the separate MikkTSpace reference",
        indices: "meshes-topology (indexed order; sequential vertices if indices absent); meshes-usage (split corners for flat normals)",
        matrix: "transformations (TRS; parent global times local)",
        frontFace: "instantiation (negative global determinant makes front faces clockwise)",
        ...(example ? { nodePath: "transformations; nodes-and-hierarchy (authored local matrices or TRS, root to instance, for evaluation at the client's float32 storage boundaries)" } : {}),
        color: "metallic-roughness-material (linear baseColorFactor); alpha-coverage (fourth component)",
        metallic: "metallic-roughness-material (metallicFactor)", roughness: "metallic-roughness-material (roughnessFactor)",
        emissive: "emissive (linear emissiveFactor)", alphaMode: "alpha-coverage", cutoff: "alpha-coverage (MASK only)",
        textureImage: "materials-overview (texture index); textures (source image)",
        ...(example ? { textureIndex: "material.pbrMetallicRoughness.schema.json (baseColorTexture.index); read from JSON because GLTFLoader's shared texture associations may name the last equivalent texture index", images: "images (uri relative to the glTF asset)" } : {}),
        unlit: "KHR_materials_unlit: Extending Materials (at specification revision)", doubleSided: "double-sided",
        textures: "textures (source); samplers (Filtering; Wrapping); absent filters choose linear / linear-mipmap-linear, permitted by Samplers Overview",
    }, records,
}, null, 2)}\n`);
console.log(`Recorded ${records.length} fixtures from ${head}`);
