export function compileVertexBody(vertex?: string): string {
    return vertex
        ? `var pos = localPos;
    var uv = meshUv;
    ${vertex}
    return VertexTransformResult(pos, uv);`
        : "return VertexTransformResult(localPos, meshUv);";
}

export function injectInstPreamble(vertexBody: string): string {
    return vertexBody.replace(
        "var pos = localPos;",
        "var pos = localPos;\n    let inst = instanceData[eid];",
    );
}

const STARS_WGSL = /* wgsl */ `
fn hashStar(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2Star(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.x, p.y, p.x) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn sampleStars(dir: vec3<f32>) -> vec3<f32> {
    if (sky.starParams.z <= 0.0 || dir.y < 0.0) {
        return vec3(0.0);
    }

    let theta = atan2(dir.z, dir.x);
    let phi = asin(clamp(dir.y, -1.0, 1.0));

    let gridSize = mix(20.0, 100.0, sky.starParams.y);
    let cell = vec2(theta * gridSize / 3.14159, phi * gridSize / 1.5708);
    let cellId = floor(cell);
    let cellFract = fract(cell);

    var starColor = vec3(0.0);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = cellId + vec2(f32(dx), f32(dy));
            let starHash = hashStar(neighbor);

            if (starHash > sky.starParams.y * 0.7) {
                continue;
            }

            let starPos = hash2Star(neighbor);
            let starCenter = neighbor + starPos;
            let dist = length(cell - starCenter);

            let brightness = hashStar(neighbor + vec2(100.0, 100.0));
            let radius = 0.02 + brightness * 0.03;

            if (dist < radius) {
                let twinkle = 0.8 + 0.2 * sin(brightness * 100.0);
                let intensity = sky.starParams.x * brightness * twinkle;
                let falloff = 1.0 - smoothstep(0.0, radius, dist);

                let temp = hashStar(neighbor + vec2(200.0, 200.0));
                let tint = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), temp);

                starColor = max(starColor, tint * intensity * falloff);
            }
        }
    }

    return starColor;
}
`;

export const NOISE_WGSL = /* wgsl */ `
fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn value2d(p: vec2f, seed: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(fract(sin(dot(i, seed)) * 43758.5) * 2.0 - 1.0,
            fract(sin(dot(i + vec2(1.0, 0.0), seed)) * 43758.5) * 2.0 - 1.0, u.x),
        mix(fract(sin(dot(i + vec2(0.0, 1.0), seed)) * 43758.5) * 2.0 - 1.0,
            fract(sin(dot(i + vec2(1.0, 1.0), seed)) * 43758.5) * 2.0 - 1.0, u.x), u.y);
}

fn simplex2(p: vec2<f32>) -> f32 {
    let K1 = 0.366025404;
    let K2 = 0.211324865;

    let i = floor(p + (p.x + p.y) * K1);
    let a = p - i + (i.x + i.y) * K2;

    let o = select(vec2(0.0, 1.0), vec2(1.0, 0.0), a.x > a.y);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0 * K2;

    let h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), vec3(0.0));
    let h4 = h * h * h * h;

    let n = vec3(
        dot(a, vec2(hash2(i) * 2.0 - 1.0, hash2(i + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(b, vec2(hash2(i + o) * 2.0 - 1.0, hash2(i + o + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(c, vec2(hash2(i + 1.0) * 2.0 - 1.0, hash2(i + vec2(1.0, 2.0)) * 2.0 - 1.0))
    );

    return dot(h4, n) * 70.0;
}

const FBM2_OCTAVES = 5;

fn fbm2(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < FBM2_OCTAVES; i++) {
        value += amplitude * simplex2(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}
`;

const MOON_WGSL = /* wgsl */ `
fn sampleMoon(dir: vec3<f32>) -> vec3<f32> {
    if (sky.moonParams.z <= 0.0) {
        return vec3(0.0);
    }

    let moonDir = sky.moonDirection.xyz;
    let moonDot = dot(dir, moonDir);

    let moonSize = 0.9995;
    let moonColor = vec3(0.9, 0.9, 0.85);
    let edgeWidth = 0.0003;
    let opacity = sky.moonParams.y;

    if (moonDot <= moonSize - edgeWidth) {
        return vec3(0.0);
    }

    let toCenter = dir - moonDir * moonDot;
    let diskRight = normalize(cross(moonDir, vec3(0.0, 1.0, 0.0)));
    let diskUp = cross(diskRight, moonDir);

    let diskRadius = sqrt(1.0 - moonSize * moonSize);
    let u = dot(toCenter, diskRight) / diskRadius;
    let v = dot(toCenter, diskUp) / diskRadius;

    let r2 = u * u + v * v;
    let z = sqrt(max(0.0, 1.0 - r2));

    let diskEdge = smoothstep(1.0 + edgeWidth / diskRadius, 1.0 - edgeWidth / diskRadius, sqrt(r2));

    let limb = pow(z, 0.6);

    let cellU = u * 8.0;
    let cellV = v * 8.0;
    let craterNoise = hashStar(floor(vec2(cellU, cellV)) + vec2(50.0, 50.0));
    let surfaceVariation = 0.85 + 0.15 * craterNoise;

    let phase = sky.moonParams.x;
    let sunAngle = phase * 6.28318;
    let sunLocalX = sin(sunAngle);
    let sunLocalZ = -cos(sunAngle);

    let illumination = u * sunLocalX + z * sunLocalZ;
    let lit = smoothstep(-0.05, 0.05, illumination);

    let earthshine = vec3(0.06, 0.07, 0.1);
    let dayColor = moonColor * surfaceVariation * limb;
    let surfaceColor = mix(earthshine * limb, dayColor, lit);

    return surfaceColor * diskEdge * opacity;
}
`;

const CLOUDS_WGSL = /* wgsl */ `
fn sampleClouds(dir: vec3<f32>) -> vec4<f32> {
    if (sky.cloudParams.w <= 0.0 || dir.y < 0.01) {
        return vec4(0.0);
    }

    let t = sky.cloudParams.z / max(dir.y, 0.001);
    let uv = dir.xz * t;

    var n = fbm2(uv);

    let coverage = sky.cloudParams.x;
    let density = sky.cloudParams.y;
    n = smoothstep(1.0 - coverage, 1.0, n * 0.5 + 0.5) * density;

    n *= smoothstep(0.0, 0.15, dir.y);

    return vec4(sky.cloudColor.rgb, n);
}
`;

export const SKY_DIR_WGSL = /* wgsl */ `
const DEG_TO_RAD: f32 = 0.017453292;

fn computeSkyDir(screenX: f32, screenY: f32) -> vec3<f32> {
    let width = scene.viewport.x;
    let height = scene.viewport.y;

    let ndcX = screenX * 2.0 - 1.0;
    let ndcY = 1.0 - screenY * 2.0;

    let aspect = width / height;

    let cameraWorld = scene.cameraWorld;
    let r00 = cameraWorld[0][0]; let r10 = cameraWorld[0][1]; let r20 = cameraWorld[0][2];
    let r01 = cameraWorld[1][0]; let r11 = cameraWorld[1][1]; let r21 = cameraWorld[1][2];
    let r02 = cameraWorld[2][0]; let r12 = cameraWorld[2][1]; let r22 = cameraWorld[2][2];

    let skyFov = select(scene.fov, 60.0, scene.cameraMode > 0.5);
    let tanHalfFov = tan((skyFov * DEG_TO_RAD) / 2.0);
    let camDirX = ndcX * aspect * tanHalfFov;
    let camDirY = ndcY * tanHalfFov;
    let camDirZ = -1.0;
    var dirX = r00 * camDirX + r01 * camDirY + r02 * camDirZ;
    var dirY = r10 * camDirX + r11 * camDirY + r12 * camDirZ;
    var dirZ = r20 * camDirX + r21 * camDirY + r22 * camDirZ;
    let len = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
    dirX /= len; dirY /= len; dirZ /= len;
    return vec3(dirX, dirY, dirZ);
}
`;

export const SKY_WGSL = /* wgsl */ `
${STARS_WGSL}
${MOON_WGSL}
${CLOUDS_WGSL}

fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    if (sky.skyZenith.a <= 0.0) {
        return scene.clearColor.rgb;
    }

    let t = pow(clamp(dir.y, 0.0, 1.0), 0.25);
    var color = mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, t);

    if (sky.horizonBand > 0.0) {
        let horizonBlend = pow(1.0 - abs(dir.y), 32.0) * sky.horizonBand;
        let bandColor = sky.skyHorizon.rgb * 1.5;
        color = mix(color, bandColor, horizonBlend);
    }

    color += sampleStars(dir);

    let clouds = sampleClouds(dir);
    color = mix(color, clouds.rgb, clouds.a);

    let moonContrib = sampleMoon(dir);
    color += moonContrib * (1.0 - clouds.a * 0.7);

    if (sky.sunParams.y > 0.0) {
        let sunDir = sky.sunDirection.xyz;
        let sunDot = dot(dir, sunDir);

        let sunVisualColor = select(scene.sunColor.rgb, sky.sunVisualColor.rgb, sky.sunParams.z > 0.5);

        let glowStrength = sky.sunParams.w;
        if (glowStrength > 0.0) {
            let g = 0.76;
            let gg = g * g;
            let mie = (1.0 - gg) / pow(1.0 + gg - 2.0 * g * sunDot, 1.5);
            color += sunVisualColor * mie * glowStrength * 0.025;

            let angle = max(0.0, sunDot);
            let corona = pow(angle, 512.0) * 0.4 + pow(angle, 128.0) * 0.06;
            let warmTint = vec3f(1.0, 0.9, 0.7);
            color += warmTint * sunVisualColor * corona * glowStrength;
        }

        let baseSunSize = 0.9995;
        let sunSizeParam = sky.sunParams.x;
        let sunThreshold = 1.0 - (1.0 - baseSunSize) * sunSizeParam;
        let sunEdgeWidth = (1.0 - sunThreshold) * 0.15;

        let diskBlend = smoothstep(sunThreshold - sunEdgeWidth, sunThreshold + sunEdgeWidth, sunDot);
        if (diskBlend > 0.0) {
            let radial = saturate((sunDot - sunThreshold) / (1.0 - sunThreshold));
            let r = 1.0 - radial;
            let mu = sqrt(1.0 - r * r);
            let limbDarken = 1.0 - 0.6 * (1.0 - mu);
            color += sunVisualColor * limbDarken * diskBlend;

            let edgeDist = 1.0 - smoothstep(0.0, 1.0, radial);
            let fringe = vec3f(
                smoothstep(0.3, 0.7, edgeDist),
                smoothstep(0.5, 0.9, edgeDist),
                smoothstep(0.7, 1.0, edgeDist)
            );
            color += fringe * sunVisualColor * 0.15 * diskBlend * (1.0 - radial);
        }
    }

    if (sky.hazeDensity > 0.0) {
        let horizonFactor = 1.0 - clamp(dir.y, 0.0, 1.0);
        let hazeAmount = pow(horizonFactor, 2.0) * saturate(sky.hazeDensity * 5.0);
        color = mix(color, sky.hazeColor.rgb, hazeAmount);
    }

    return color;
}
`;

export const HAZE_WGSL = /* wgsl */ `
fn applyHaze(color: vec3<f32>, dist: f32) -> vec3<f32> {
    if (sky.hazeDensity <= 0.0) {
        return color;
    }
    let haze = 1.0 - exp(-sky.hazeDensity * dist);
    return mix(color, sky.hazeColor.rgb, haze);
}
`;

export const SHADOW_SAMPLE_WGSL = /* wgsl */ `
const CASCADE_BLEND_RANGE: f32 = 0.1;
const PCF_SAMPLE_COUNT: i32 = 5;
const VOGEL_GOLDEN_ANGLE: f32 = 2.399963;

fn selectCascade(viewZ: f32) -> u32 {
    if (viewZ < shadow.cascadeSplits.x) { return 0u; }
    if (viewZ < shadow.cascadeSplits.y) { return 1u; }
    if (viewZ < shadow.cascadeSplits.z) { return 2u; }
    return 3u;
}

fn getCascadeViewProj(cascade: u32) -> mat4x4<f32> {
    switch cascade {
        case 0u: { return shadow.cascade0ViewProj; }
        case 1u: { return shadow.cascade1ViewProj; }
        case 2u: { return shadow.cascade2ViewProj; }
        default: { return shadow.cascade3ViewProj; }
    }
}

fn getCascadeSplit(cascade: u32) -> f32 {
    switch cascade {
        case 0u: { return shadow.cascadeSplits.x; }
        case 1u: { return shadow.cascadeSplits.y; }
        case 2u: { return shadow.cascadeSplits.z; }
        default: { return shadow.cascadeSplits.w; }
    }
}

fn sampleShadowAtCascade(worldPos: vec3<f32>, cascade: u32, fragCoord: vec2<f32>) -> f32 {
    let lightPos = getCascadeViewProj(cascade) * vec4(worldPos, 1.0);
    let ndc = lightPos.xyz / lightPos.w;

    let inBounds = abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;
    if (!inBounds) { return 1.0; }

    var uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;

    let offset = vec2(f32(cascade % 2u) * 0.5, f32(cascade / 2u) * 0.5);
    uv = uv * 0.5 + offset;

    let atlasSize = vec2<f32>(textureDimensions(shadowMap));
    let pcfRadius = scene.shadowSoftness / atlasSize.x;

    let ign = fract(52.9829189 * fract(0.06711056 * fragCoord.x + 0.00583715 * fragCoord.y));
    let angle = ign * 6.28318;

    var total = 0.0;
    for (var i = 0; i < PCF_SAMPLE_COUNT; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(PCF_SAMPLE_COUNT)) * pcfRadius;
        let a = f32(i) * VOGEL_GOLDEN_ANGLE + angle;
        let tapOffset = vec2(cos(a), sin(a)) * r;
        total += textureSampleCompareLevel(shadowMap, shadowSampler, uv + tapOffset, ndc.z);
    }
    return total / f32(PCF_SAMPLE_COUNT);
}

fn computeCascadeBlend(viewZ: f32, cascade: u32) -> f32 {
    if (cascade >= 3u) { return 0.0; }

    let splitEnd = getCascadeSplit(cascade);
    let blendStart = splitEnd * (1.0 - CASCADE_BLEND_RANGE);

    if (viewZ < blendStart) { return 0.0; }
    return saturate((viewZ - blendStart) / (splitEnd - blendStart));
}

fn distanceFade(viewZ: f32, maxDist: f32) -> f32 {
    let fadeStart = maxDist * 0.9;
    let fade = saturate((maxDist - viewZ) / (maxDist - fadeStart));
    return select(fade, 1.0, viewZ <= fadeStart);
}

fn sampleShadow(worldPos: vec3<f32>, viewZ: f32, fragCoord: vec2<f32>) -> f32 {
    let cascade = selectCascade(viewZ);
    let shadowCurrent = sampleShadowAtCascade(worldPos, cascade, fragCoord);

    let nextCascade = min(cascade + 1u, 3u);
    let shadowNext = sampleShadowAtCascade(worldPos, nextCascade, fragCoord);

    let blendFactor = computeCascadeBlend(viewZ, cascade) * f32(cascade < 3u);
    let cascadeShadow = mix(shadowCurrent, shadowNext, blendFactor);

    let fade = distanceFade(viewZ, shadow.cascadeSplits.w);
    return mix(1.0, cascadeShadow, fade);
}
`;

export const SPECULAR_WGSL = /* wgsl */ `
fn blinnPhongSpecular(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, roughness: f32) -> f32 {
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);
    let shininess = pow(2.0, (1.0 - roughness) * 10.0);
    let intensity = (1.0 - roughness) * (1.0 - roughness);
    return pow(NdotH, shininess) * intensity;
}
`;

export const POINT_SHADOW_SAMPLE_WGSL = /* wgsl */ `
fn selectCubeFace(dir: vec3<f32>) -> u32 {
    let absDir = abs(dir);
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        return select(1u, 0u, dir.x > 0.0);
    }
    if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        return select(3u, 2u, dir.y > 0.0);
    }
    return select(5u, 4u, dir.z > 0.0);
}

fn samplePointShadow(worldPos: vec3<f32>, normal: vec3<f32>, shadowIdx: u32, lightPos: vec3<f32>, lightRadius: f32) -> f32 {
    let toFrag = worldPos - lightPos;
    let dist = length(toFrag);
    if (dist < 1e-4) { return 1.0; }
    let dir = toFrag / dist;

    let texelSize = dist * 2.0 / 512.0;
    let NdotL = abs(dot(normal, -dir));
    let offsetScale = texelSize * (1.0 + 2.0 * saturate(1.0 - NdotL));
    let offsetPos = worldPos + normal * offsetScale;

    let face = selectCubeFace(dir);
    let vpIdx = shadowIdx * 6u + face;
    let lightClip = pointShadow.viewProj[vpIdx] * vec4(offsetPos, 1.0);
    let ndc = lightClip.xyz / lightClip.w;

    var uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;

    let border = 1.0 / 512.0;
    uv = clamp(uv, vec2(border), vec2(1.0 - border));

    let atlasU = (f32(face) + uv.x) / 6.0;
    let atlasV = (f32(shadowIdx) + uv.y) / 4.0;

    return textureSampleCompareLevel(pointShadowMap, shadowSampler, vec2(atlasU, atlasV), ndc.z);
}
`;

export const POINT_LIGHT_EVAL_WGSL = /* wgsl */ `
fn evaluatePointLight(
    surface: SurfaceData,
    lightColor: vec3<f32>,
    L: vec3<f32>,
    V: vec3<f32>,
    NdotL: f32,
    attenuation: f32,
    shadow: f32,
) -> vec3<f32> {
    let diffuse = surface.baseColor * lightColor * NdotL * attenuation * shadow;
    let spec = blinnPhongSpecular(surface.worldNormal, L, V, surface.roughness);
    let specular = lightColor * spec * NdotL * attenuation * shadow * surface.reflectivity;
    return diffuse + specular;
}
`;

export const SURFACE_HELPERS_WGSL = `${NOISE_WGSL}\n${SKY_WGSL}\n${HAZE_WGSL}\n${SPECULAR_WGSL}\n${POINT_LIGHT_EVAL_WGSL}`;

export const WGSL_LIGHTING_CALC = /* wgsl */ `
let V = normalize(scene.cameraWorld[3].xyz - surface.worldPos);
let L = -scene.sunDirection.xyz;
let NdotL = max(dot(surface.worldNormal, L), 0.0);
let ambient = scene.ambientColor.rgb * scene.ambientColor.a;
let sunDiffuse = scene.sunColor.rgb * NdotL * shadowFactor;
let diffuseColor = surface.baseColor * (ambient + sunDiffuse) + surface.emission;
let specTerm = blinnPhongSpecular(surface.worldNormal, L, V, surface.roughness);
let specular = scene.sunColor.rgb * specTerm * NdotL * shadowFactor * surface.reflectivity;
let litColor = diffuseColor + specular;
`;

export const REFLECTION_WGSL = /* wgsl */ `
fn sampleReflection(dir: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(sampleSky(dir), 0.0);
}

fn reflectionColor(surface: SurfaceData, V: vec3<f32>) -> vec3<f32> {
    if (scene.reflectionEnabled == 0u || surface.reflectivity <= 0.001) {
        return vec3<f32>(0.0);
    }
    let R = reflect(-V, surface.worldNormal);
    let env = sampleReflection(R).rgb;
    let smoothness = 1.0 - surface.roughness;
    return env * surface.reflectivity * smoothness * smoothness;
}

fn applyReflection(surface: SurfaceData, V: vec3<f32>, litColor: vec3<f32>) -> vec3<f32> {
    return litColor + reflectionColor(surface, V);
}
`;
