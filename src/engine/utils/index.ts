// The ./utils subpath: the GPU storage codecs — TGSL functions callable on the CPU and resolvable into a
// shader, except the GPU-only smallest-3 pair — the lazily resolved WGSL chunks a raw-WGSL producer or
// surface splices, the escape vocabulary for the WGSL constructs TGSL has no binding for, and the author
// math, color helpers and Registry, most of which the main barrel also exports.
export {
    linearToOklabWgsl,
    linearToSrgb,
    oklabToLinearWgsl,
    packColor,
    srgbToLinear,
    unpackColor,
    unpackColorInto,
} from "./color";
export {
    hdrColorPackWgsl,
    hdrColorUnpackWgsl,
    ldrColorPackWgsl,
    ldrColorUnpackWgsl,
    linearToSrgb1,
    octDecodeNormal,
    octEncode,
    octEncodeNormal,
    octEncodeWgsl,
    packColor4,
    packHdrColor,
    packLdrColor,
    packQuatSmallest3,
    packQuatSnorm16x4,
    packUnorm2,
    quatSnorm16x4Wgsl,
    smallest3Wgsl,
    srgbToLinear1,
    unpackHdrColor,
    unpackLdrColor,
    unpackQuatSmallest3,
    unpackQuatSnorm16x4,
} from "./encode";
export {
    clamp,
    composeMat4,
    decomposeMat4,
    eulerToQuat,
    invertMat4,
    lerp,
    lookAt,
    lookAtRotation,
    multiplyMat4,
    orthographic,
    perspective,
    qRotate,
    quatToEuler,
    type Ray,
    rotateQuatByEuler,
    slerp,
} from "./math";
export { Registry } from "./registry";
export {
    bitcastF32toU32,
    chunk,
    idiv,
    packSnorm2x16,
    packUnorm2x16,
    packUnorm4x8,
    spliceNs,
    uniformLoad,
    unpackSnorm2x16,
    unpackUnorm2x16,
} from "./tgsl";
