// engine/utils extension surface: the WGSL codec chunks a custom producer or surface splices, and their
// CPU-side pack/unpack twins (bit-identical to the intrinsics, so a value round-trips CPU↔GPU). The
// author math + color + trait-authoring helpers ride the main barrel; this is what an extender building a
// pipeline reaches for.
export { LINEAR_TO_OKLAB_WGSL, OKLAB_TO_LINEAR_WGSL, packColor } from "./color";
export {
    LDR_COLOR_UNPACK_WGSL,
    OCT_ENCODE_WGSL,
    octDecodeNormal,
    octEncodeNormal,
    POS_QUANT_PACK_WGSL,
    POS_QUANT_WGSL,
    pack2x16unorm,
    packLdrColor,
    unpack2x16unorm,
    XFORM_WGSL,
} from "./encode";
