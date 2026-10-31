//! glTF pose evaluation. Arithmetic stays f64 until an authored f32 storage boundary.
use std::slice;

#[cfg(test)]
mod tests;

struct Node {
    index: usize,
    parent: i32,
    pose: Vec<f64>,
}
struct Skin {
    joints: Vec<usize>,
    binds: Vec<f32>,
}
struct Primitive {
    pose_offset: usize,
    skin: i32,
    boxes: Vec<f32>,
    reach: Vec<f32>,
    bounds: Vec<f32>,
}
struct Asset {
    nodes: Vec<Node>,
    skins: Vec<Skin>,
    primitives: Vec<Primitive>,
}
struct Sampler {
    input: Vec<f64>,
    output: Vec<f64>,
    mode: usize,
}
struct Channel {
    sampler: usize,
    node: usize,
    path: usize,
}
struct Animation {
    samplers: Vec<Sampler>,
    channels: Vec<Channel>,
}
struct Instance {
    poses: Vec<f64>,
    globals: Vec<f32>,
    input: Vec<u32>,
    output: Vec<f32>,
}
struct Reader<'a> {
    data: &'a [f64],
    cursor: usize,
}
impl Reader<'_> {
    fn number(&mut self) -> f64 {
        let v = self.data[self.cursor];
        self.cursor += 1;
        v
    }
    fn integer(&mut self) -> usize {
        self.number() as usize
    }
    fn array(&mut self) -> Vec<f64> {
        let n = self.integer();
        let v = self.data[self.cursor..self.cursor + n].to_vec();
        self.cursor += n;
        v
    }
    fn floats(&mut self) -> Vec<f32> {
        self.array().into_iter().map(|v| v as f32).collect()
    }
}

#[no_mangle]
pub extern "C" fn alloc(count: usize) -> *mut f64 {
    Box::into_raw(vec![0.; count].into_boxed_slice()) as *mut f64
}
/// # Safety
/// `ptr` must be a live allocation from `alloc(count)`.
#[export_name = "freeLanes"]
pub unsafe extern "C" fn free_lanes(ptr: *mut f64, count: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        ptr, count,
    )));
}
/// # Safety
/// Input is a live serialized asset of `count` f64 lanes.
#[export_name = "assetNew"]
pub unsafe extern "C" fn asset_new(ptr: *const f64, count: usize) -> *mut std::ffi::c_void {
    let mut r = Reader {
        data: slice::from_raw_parts(ptr, count),
        cursor: 0,
    };
    let nodes: Vec<Node> = (0..r.integer())
        .map(|_| Node {
            index: r.integer(),
            parent: r.number() as i32,
            pose: r.array(),
        })
        .collect();
    let skins = (0..r.integer())
        .map(|_| Skin {
            joints: r.array().into_iter().map(|v| v as usize).collect(),
            binds: r.floats(),
        })
        .collect();
    let primitives = (0..r.integer())
        .map(|_| {
            let node = r.integer();
            Primitive {
                pose_offset: nodes[..node].iter().map(|n| n.pose.len()).sum(),
                skin: r.number() as i32,
                boxes: r.floats(),
                reach: r.floats(),
                bounds: r.floats(),
            }
        })
        .collect();
    Box::into_raw(Box::new(Asset {
        nodes,
        skins,
        primitives,
    }))
    .cast()
}
/// # Safety
/// Input is a live serialized animation of `count` f64 lanes.
#[export_name = "animationNew"]
pub unsafe extern "C" fn animation_new(ptr: *const f64, count: usize) -> *mut std::ffi::c_void {
    let mut r = Reader {
        data: slice::from_raw_parts(ptr, count),
        cursor: 0,
    };
    let samplers = (0..r.integer())
        .map(|_| Sampler {
            mode: r.integer(),
            input: r.array(),
            output: r.array(),
        })
        .collect();
    let channels = (0..r.integer())
        .map(|_| Channel {
            sampler: r.integer(),
            node: r.integer(),
            path: r.integer(),
        })
        .collect();
    Box::into_raw(Box::new(Animation { samplers, channels })).cast()
}
fn output_len(a: &Asset) -> usize {
    a.primitives
        .iter()
        .map(|p| {
            4 + p.reach.len()
                + if p.skin < 0 {
                    0
                } else {
                    a.skins[p.skin as usize].joints.len() * 16
                }
        })
        .sum()
}
/// # Safety
/// `asset` is a live asset handle.
#[export_name = "instanceNew"]
pub unsafe extern "C" fn instance_new(asset: *const std::ffi::c_void) -> *mut std::ffi::c_void {
    let a = &*asset.cast::<Asset>();
    Box::into_raw(Box::new(Instance {
        poses: vec![0.; a.nodes.iter().map(|n| n.pose.len()).sum()],
        globals: vec![0.; a.nodes.len() * 16],
        input: vec![0; a.primitives.len() * 2],
        output: vec![0.; output_len(a)],
    }))
    .cast()
}
/// # Safety
/// `handle` is a live instance handle; field is 0 (poses), 1 (globals), 2 (inputs), or 3 (outputs).
#[no_mangle]
pub unsafe extern "C" fn view(
    handle: *mut std::ffi::c_void,
    field: usize,
) -> *mut std::ffi::c_void {
    let i = &mut *handle.cast::<Instance>();
    match field {
        0 => i.poses.as_mut_ptr().cast(),
        1 => i.globals.as_mut_ptr().cast(),
        2 => i.input.as_mut_ptr().cast(),
        _ => i.output.as_mut_ptr().cast(),
    }
}
/// # Safety
/// Handle is live and matches kind: 0 asset, 1 animation, 2 instance. It must not be reused.
#[no_mangle]
pub unsafe extern "C" fn destroy(handle: *mut std::ffi::c_void, kind: usize) {
    match kind {
        0 => drop(Box::from_raw(handle.cast::<Asset>())),
        1 => drop(Box::from_raw(handle.cast::<Animation>())),
        _ => drop(Box::from_raw(handle.cast::<Instance>())),
    }
}

fn evaluate(nodes: &[Node], animation: Option<&Animation>, time: f64, poses: &mut [f64]) {
    let mut start = 0;
    for node in nodes {
        poses[start..start + node.pose.len()].copy_from_slice(&node.pose);
        start += node.pose.len();
    }
    let Some(animation) = animation else {
        return;
    };
    for channel in &animation.channels {
        let mut index = 0;
        let mut start = 0;
        while index < nodes.len() && nodes[index].index != channel.node {
            start += nodes[index].pose.len();
            index += 1;
        }
        if index == nodes.len() {
            continue;
        }
        start += [0, 3, 7, 10][channel.path];
        let Sampler {
            input,
            output,
            mode,
        } = &animation.samplers[channel.sampler];
        let cubic = *mode == 2;
        let stride = if cubic { 3 } else { 1 };
        let width = output.len() / input.len() / stride;
        let mut low = 0;
        let mut high = input.len() - 1;
        while low < high {
            let middle = (low + high).div_ceil(2);
            if input[middle] <= time {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let key = low;
        let offset = (key * stride + usize::from(cubic)) * width;
        let value = &mut poses[start..start + width];
        value.copy_from_slice(&output[offset..offset + width]);
        if time > input[0] && time < input[input.len() - 1] && time != input[key] && *mode != 0 {
            let duration = input[key + 1] - input[key];
            let t = (time - input[key]) / duration;
            let next = offset + width * stride;
            if cubic {
                let t2 = t * t;
                let t3 = t2 * t;
                for c in 0..width {
                    value[c] = (2. * t3 - 3. * t2 + 1.) * value[c]
                        + duration * (t3 - 2. * t2 + t) * output[offset + width + c]
                        + (-2. * t3 + 3. * t2) * output[next + c]
                        + duration * (t3 - t2) * output[next - width + c];
                }
                if channel.path == 1 {
                    let max = value.iter().fold(0_f64, |m, v| m.max(v.abs()));
                    let mut sum = 0.;
                    for v in value.iter() {
                        let v = v / max;
                        sum += v * v;
                    }
                    let length = max * libm::sqrt(sum);
                    for v in value {
                        *v /= length;
                    }
                }
            } else if channel.path == 1 {
                let mut dot = 0.;
                for c in 0..width {
                    dot += value[c] * output[next + c];
                }
                let sign = if dot < 0. { -1. } else { 1. };
                let angle = libm::acos(js_min(dot.abs(), 1.));
                let a = if angle < 1e-6 {
                    1. - t
                } else {
                    libm::sin((1. - t) * angle) / libm::sin(angle)
                };
                let b = if angle < 1e-6 {
                    t
                } else {
                    libm::sin(t * angle) / libm::sin(angle)
                };
                for c in 0..width {
                    value[c] = a * value[c] + sign * b * output[next + c];
                }
            } else {
                for c in 0..width {
                    value[c] = (1. - t) * value[c] + t * output[next + c];
                }
            }
        }
    }
}
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0. && b == 0. && (a.is_sign_negative() || b.is_sign_negative()) {
        -0.
    } else {
        a.min(b)
    }
}

/// # Safety
/// Handles are live, belong to this asset, and do not alias.
#[export_name = "evaluate"]
pub unsafe extern "C" fn evaluate_only(
    asset: *const std::ffi::c_void,
    instance: *mut std::ffi::c_void,
    animation: *const std::ffi::c_void,
    time: f64,
) {
    let a = &*asset.cast::<Asset>();
    let i = &mut *instance.cast::<Instance>();
    evaluate(
        &a.nodes,
        animation.cast::<Animation>().as_ref(),
        time,
        &mut i.poses,
    );
}

fn compose(p: &[f64]) -> [f32; 16] {
    let (qx, qy, qz, qw) = (p[3], p[4], p[5], p[6]);
    let (sx, sy, sz) = (p[7], p[8], p[9]);
    let (x2, y2, z2) = (qx + qx, qy + qy, qz + qz);
    let (xx, xy, xz, yy, yz, zz, wx, wy, wz) = (
        qx * x2,
        qx * y2,
        qx * z2,
        qy * y2,
        qy * z2,
        qz * z2,
        qw * x2,
        qw * y2,
        qw * z2,
    );
    [
        ((1. - yy - zz) * sx) as f32,
        ((xy + wz) * sx) as f32,
        ((xz - wy) * sx) as f32,
        0.,
        ((xy - wz) * sy) as f32,
        ((1. - xx - zz) * sy) as f32,
        ((yz + wx) * sy) as f32,
        0.,
        ((xz + wy) * sz) as f32,
        ((yz - wx) * sz) as f32,
        ((1. - xx - yy) * sz) as f32,
        0.,
        p[0] as f32,
        p[1] as f32,
        p[2] as f32,
        1.,
    ]
}
fn multiply(a: &[f32], b: &[f32], out: &mut [f32]) {
    for r in 0..4 {
        for c in 0..4 {
            out[c * 4 + r] = (f64::from(a[r]) * f64::from(b[c * 4])
                + f64::from(a[r + 4]) * f64::from(b[c * 4 + 1])
                + f64::from(a[r + 8]) * f64::from(b[c * 4 + 2])
                + f64::from(a[r + 12]) * f64::from(b[c * 4 + 3]))
                as f32;
        }
    }
}
fn compose_globals(a: &Asset, i: &mut Instance) {
    let mut start = 0;
    for (n, node) in a.nodes.iter().enumerate() {
        let local = compose(&i.poses[start..]);
        start += node.pose.len();
        if node.parent == -1 {
            i.globals[n * 16..n * 16 + 16].copy_from_slice(&local);
        } else {
            let (previous, current) = i.globals.split_at_mut(n * 16);
            multiply(
                &previous[node.parent as usize * 16..],
                &local,
                &mut current[..16],
            );
        }
    }
}
fn write_morph_pose(p: &Primitive, pose: &[f64], count: usize, out: &mut [f32]) -> f64 {
    let mut reach = 0.;
    for m in 0..count.min(p.reach.len()) {
        out[m] = pose[10 + m] as f32;
        reach += f64::from(out[m]).abs() * f64::from(p.reach[m]);
    }
    reach
}
fn write_skin_pose(
    skin: &Skin,
    boxes: &[f32],
    globals: &[f32],
    reach: f64,
    palette: &mut [f32],
    bounds: &mut [f32],
) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for (j, node) in skin.joints.iter().enumerate() {
        let o = j * 16;
        multiply(
            &globals[node * 16..],
            &skin.binds[o..],
            &mut palette[o..o + 16],
        );
        let q = j * 6;
        if boxes[q] == f32::INFINITY {
            continue;
        }
        let mut center = [0.; 3];
        let mut half = [0.; 3];
        for a in 0..3 {
            center[a] = (f64::from(boxes[q + a]) + f64::from(boxes[q + 3 + a])) / 2.;
            half[a] = (f64::from(boxes[q + 3 + a]) - f64::from(boxes[q + a])) / 2. + reach;
        }
        for a in 0..3 {
            let tx = f64::from(palette[o + a]) * center[0]
                + f64::from(palette[o + 4 + a]) * center[1]
                + f64::from(palette[o + 8 + a]) * center[2]
                + f64::from(palette[o + 12 + a]);
            let sx = f64::from(palette[o + a]).abs() * half[0]
                + f64::from(palette[o + 4 + a]).abs() * half[1]
                + f64::from(palette[o + 8 + a]).abs() * half[2];
            min[a] = js_min(min[a], tx - sx);
            max[a] = -js_min(-max[a], -(tx + sx));
        }
    }
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    for a in 0..3 {
        bounds[a] = ((min[a] + max[a]) / 2.) as f32;
    }
    bounds[3] = (libm::sqrt(dx * dx + dy * dy + dz * dz) / 2.) as f32;
}
/// # Safety
/// Handles are live, belong to this asset, and do not alias. Null animation selects rest pose.
#[no_mangle]
pub unsafe extern "C" fn pose(
    asset: *const std::ffi::c_void,
    instance: *mut std::ffi::c_void,
    animation: *const std::ffi::c_void,
    time: f64,
) {
    let a = &*asset.cast::<Asset>();
    let i = &mut *instance.cast::<Instance>();
    evaluate(
        &a.nodes,
        animation.cast::<Animation>().as_ref(),
        time,
        &mut i.poses,
    );
    compose_globals(a, i);
    let mut offset = 0;
    for (n, p) in a.primitives.iter().enumerate() {
        let (before, rest) = i.output[offset..].split_at_mut(4);
        let (weights, palette) = rest.split_at_mut(p.reach.len());
        let reach = write_morph_pose(
            p,
            &i.poses[p.pose_offset..],
            i.input[n * 2 + 1].saturating_sub(1) as usize,
            weights,
        );
        if p.skin >= 0 {
            let skin = &a.skins[p.skin as usize];
            if i.input[n * 2] != 0 {
                write_skin_pose(
                    skin,
                    &p.boxes,
                    &i.globals,
                    reach,
                    &mut palette[..skin.joints.len() * 16],
                    before,
                );
            }
            offset += skin.joints.len() * 16;
        } else if i.input[n * 2 + 1] != 0 {
            before.copy_from_slice(&p.bounds);
            before[3] = (f64::from(p.bounds[3]) + reach) as f32;
        }
        offset += 4 + p.reach.len();
    }
}
