// glTF extension surface for tooling + custom async pipelines: the deviceless `decode` and the
// content-keyed cache (`ensureDecoded` / `register`), off-thread decode (`decodeInWorker`), the
// union-staging progress (`unionPending`) + cache management (`invalidate` / `clearGltfCache` /
// `gltfCacheStats`), the PBR baseColor size-bucket names, and the raw parsed-glTF types. The author happy
// path (`loadGltf` / `placeScene` / `GltfPlugin`) rides the barrel.

export type { DecodedGltf } from "./assets";
export {
    clearGltfCache,
    decode,
    ensureDecoded,
    gltfCacheStats,
    invalidate,
    register,
    unionPending,
} from "./assets";
export type { GltfImage, GltfInstance, GltfJson, GltfMaterial, GltfMesh, GltfScene } from "./gltf";
export { ALBEDO_BUCKETS, ALBEDO_NAMES } from "./image";
export { decodeInWorker } from "./pool";
