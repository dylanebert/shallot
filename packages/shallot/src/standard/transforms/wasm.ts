import { capacity } from "../../engine";
import wasmInit, {
    get_pos_x_ptr,
    get_pos_y_ptr,
    get_pos_z_ptr,
    get_quat_x_ptr,
    get_quat_y_ptr,
    get_quat_z_ptr,
    get_quat_w_ptr,
    get_scale_x_ptr,
    get_scale_y_ptr,
    get_scale_z_ptr,
    get_matrices_ptr,
    get_indices_ptr,
    get_parents_ptr,
    get_capacity,
    get_no_parent,
    init_data,
    ensure_capacity as wasmEnsureCapacity,
    compute_transforms,
} from "../../../rust/transforms/pkg/shallot_transforms.js";

export let posX: Float32Array;
export let posY: Float32Array;
export let posZ: Float32Array;
export let quatX: Float32Array;
export let quatY: Float32Array;
export let quatZ: Float32Array;
export let quatW: Float32Array;
export let scaleX: Float32Array;
export let scaleY: Float32Array;
export let scaleZ: Float32Array;
export let matrices: Float32Array;
export let indices: Uint32Array;
export let parents: Uint32Array;
export let NoParent: number;

let wasm: Awaited<ReturnType<typeof wasmInit>>;
let wasmCap = 0;

function rebindViews() {
    const buffer = wasm.memory.buffer;
    wasmCap = get_capacity();
    posX = new Float32Array(buffer, get_pos_x_ptr(), wasmCap);
    posY = new Float32Array(buffer, get_pos_y_ptr(), wasmCap);
    posZ = new Float32Array(buffer, get_pos_z_ptr(), wasmCap);
    quatX = new Float32Array(buffer, get_quat_x_ptr(), wasmCap);
    quatY = new Float32Array(buffer, get_quat_y_ptr(), wasmCap);
    quatZ = new Float32Array(buffer, get_quat_z_ptr(), wasmCap);
    quatW = new Float32Array(buffer, get_quat_w_ptr(), wasmCap);
    scaleX = new Float32Array(buffer, get_scale_x_ptr(), wasmCap);
    scaleY = new Float32Array(buffer, get_scale_y_ptr(), wasmCap);
    scaleZ = new Float32Array(buffer, get_scale_z_ptr(), wasmCap);
    matrices = new Float32Array(buffer, get_matrices_ptr(), wasmCap * 16);
    indices = new Uint32Array(buffer, get_indices_ptr(), wasmCap);
    parents = new Uint32Array(buffer, get_parents_ptr(), wasmCap);
}

export function sync(): void {
    if (!wasm) return;
    const cap = capacity();
    if (cap === wasmCap) return;
    wasmEnsureCapacity(cap);
    rebindViews();
}

export async function init(): Promise<void> {
    if (posX) return;
    wasm = await wasmInit();
    init_data();
    NoParent = get_no_parent();
    rebindViews();
}

export function compute(count: number): void {
    compute_transforms(count);
}
