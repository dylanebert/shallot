import { mesh, surface, createSphere, createCone } from "@dylanebert/shallot";

export function registerContent() {
    mesh(createCone(), "cone");

    mesh(createSphere(48, 32), "blob-sphere");

    surface(
        {
            vertex: `
    let np = normal * 3.0 + vec3(scene.time * 0.4, scene.time * 0.3, scene.time * 0.2);
    let ni = floor(np);
    let nf = fract(np);
    let nu = nf * nf * (3.0 - 2.0 * nf);
    let seed = vec3<f32>(127.1, 311.7, 74.7);
    let c000 = fract(sin(dot(ni, seed)) * 43758.5);
    let c100 = fract(sin(dot(ni + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let c010 = fract(sin(dot(ni + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let c110 = fract(sin(dot(ni + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let c001 = fract(sin(dot(ni + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let c101 = fract(sin(dot(ni + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let c011 = fract(sin(dot(ni + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let c111 = fract(sin(dot(ni + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n1 = mix(mix(mix(c000, c100, nu.x), mix(c010, c110, nu.x), nu.y),
                 mix(mix(c001, c101, nu.x), mix(c011, c111, nu.x), nu.y), nu.z);

    let np2 = np * 2.0;
    let ni2 = floor(np2);
    let nf2 = fract(np2);
    let nu2 = nf2 * nf2 * (3.0 - 2.0 * nf2);
    let d000 = fract(sin(dot(ni2, seed)) * 43758.5);
    let d100 = fract(sin(dot(ni2 + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let d010 = fract(sin(dot(ni2 + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let d110 = fract(sin(dot(ni2 + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let d001 = fract(sin(dot(ni2 + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let d101 = fract(sin(dot(ni2 + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let d011 = fract(sin(dot(ni2 + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let d111 = fract(sin(dot(ni2 + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n2 = mix(mix(mix(d000, d100, nu2.x), mix(d010, d110, nu2.x), nu2.y),
                 mix(mix(d001, d101, nu2.x), mix(d011, d111, nu2.x), nu2.y), nu2.z);

    let np3 = np * 4.0;
    let ni3 = floor(np3);
    let nf3 = fract(np3);
    let nu3 = nf3 * nf3 * (3.0 - 2.0 * nf3);
    let e000 = fract(sin(dot(ni3, seed)) * 43758.5);
    let e100 = fract(sin(dot(ni3 + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let e010 = fract(sin(dot(ni3 + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let e110 = fract(sin(dot(ni3 + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let e001 = fract(sin(dot(ni3 + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let e101 = fract(sin(dot(ni3 + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let e011 = fract(sin(dot(ni3 + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let e111 = fract(sin(dot(ni3 + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n3 = mix(mix(mix(e000, e100, nu3.x), mix(e010, e110, nu3.x), nu3.y),
                 mix(mix(e001, e101, nu3.x), mix(e011, e111, nu3.x), nu3.y), nu3.z);

    let noiseVal = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
    let displacement = (noiseVal - 0.3) * 0.5;
    pos = pos + normal * displacement;`,
            fragment: `
    (*surface).roughness = 0.05;`,
        },
        "noise-blob",
    );

    surface(
        {
            properties: [{ name: "waveHeight", type: "f32" }],
            vertex: `
    let np = normal * 3.0 + vec3(scene.time * 0.4, scene.time * 0.3, scene.time * 0.2);
    let ni = floor(np);
    let nf = fract(np);
    let nu = nf * nf * (3.0 - 2.0 * nf);
    let seed = vec3<f32>(127.1, 311.7, 74.7);
    let c000 = fract(sin(dot(ni, seed)) * 43758.5);
    let c100 = fract(sin(dot(ni + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let c010 = fract(sin(dot(ni + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let c110 = fract(sin(dot(ni + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let c001 = fract(sin(dot(ni + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let c101 = fract(sin(dot(ni + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let c011 = fract(sin(dot(ni + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let c111 = fract(sin(dot(ni + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n1 = mix(mix(mix(c000, c100, nu.x), mix(c010, c110, nu.x), nu.y),
                 mix(mix(c001, c101, nu.x), mix(c011, c111, nu.x), nu.y), nu.z);

    let np2 = np * 2.0;
    let ni2 = floor(np2);
    let nf2 = fract(np2);
    let nu2 = nf2 * nf2 * (3.0 - 2.0 * nf2);
    let d000 = fract(sin(dot(ni2, seed)) * 43758.5);
    let d100 = fract(sin(dot(ni2 + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let d010 = fract(sin(dot(ni2 + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let d110 = fract(sin(dot(ni2 + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let d001 = fract(sin(dot(ni2 + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let d101 = fract(sin(dot(ni2 + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let d011 = fract(sin(dot(ni2 + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let d111 = fract(sin(dot(ni2 + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n2 = mix(mix(mix(d000, d100, nu2.x), mix(d010, d110, nu2.x), nu2.y),
                 mix(mix(d001, d101, nu2.x), mix(d011, d111, nu2.x), nu2.y), nu2.z);

    let np3 = np * 4.0;
    let ni3 = floor(np3);
    let nf3 = fract(np3);
    let nu3 = nf3 * nf3 * (3.0 - 2.0 * nf3);
    let e000 = fract(sin(dot(ni3, seed)) * 43758.5);
    let e100 = fract(sin(dot(ni3 + vec3(1.0, 0.0, 0.0), seed)) * 43758.5);
    let e010 = fract(sin(dot(ni3 + vec3(0.0, 1.0, 0.0), seed)) * 43758.5);
    let e110 = fract(sin(dot(ni3 + vec3(1.0, 1.0, 0.0), seed)) * 43758.5);
    let e001 = fract(sin(dot(ni3 + vec3(0.0, 0.0, 1.0), seed)) * 43758.5);
    let e101 = fract(sin(dot(ni3 + vec3(1.0, 0.0, 1.0), seed)) * 43758.5);
    let e011 = fract(sin(dot(ni3 + vec3(0.0, 1.0, 1.0), seed)) * 43758.5);
    let e111 = fract(sin(dot(ni3 + vec3(1.0, 1.0, 1.0), seed)) * 43758.5);
    let n3 = mix(mix(mix(e000, e100, nu3.x), mix(e010, e110, nu3.x), nu3.y),
                 mix(mix(e001, e101, nu3.x), mix(e011, e111, nu3.x), nu3.y), nu3.z);

    let noiseVal = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
    let displacement = (noiseVal - 0.3) * inst.waveHeight;
    pos = pos + normal * displacement;`,
            fragment: `
    (*surface).roughness = 0.05;`,
        },
        "wave-blob",
    );

    surface(
        {
            fragment: `
    let wp = (*surface).worldPos * 12.0;

    let p1 = vec3<f32>(wp.x * 2.0, wp.y * 0.4, wp.z * 2.0);
    let i1 = floor(p1); let f1 = fract(p1);
    let u1 = f1 * f1 * (3.0 - 2.0 * f1);
    let n1 = mix(
        mix(mix(fract(sin(dot(i1, vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i1 + vec3(1,0,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u1.x),
            mix(fract(sin(dot(i1 + vec3(0,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i1 + vec3(1,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u1.x), u1.y),
        mix(mix(fract(sin(dot(i1 + vec3(0,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i1 + vec3(1,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u1.x),
            mix(fract(sin(dot(i1 + vec3(0,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i1 + vec3(1,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u1.x), u1.y), u1.z);

    let p2 = p1 * 2.0;
    let i2 = floor(p2); let f2 = fract(p2);
    let u2 = f2 * f2 * (3.0 - 2.0 * f2);
    let n2 = mix(
        mix(mix(fract(sin(dot(i2, vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i2 + vec3(1,0,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u2.x),
            mix(fract(sin(dot(i2 + vec3(0,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i2 + vec3(1,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u2.x), u2.y),
        mix(mix(fract(sin(dot(i2 + vec3(0,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i2 + vec3(1,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u2.x),
            mix(fract(sin(dot(i2 + vec3(0,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i2 + vec3(1,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u2.x), u2.y), u2.z);

    let p3 = p1 * 4.0;
    let i3 = floor(p3); let f3 = fract(p3);
    let u3 = f3 * f3 * (3.0 - 2.0 * f3);
    let n3 = mix(
        mix(mix(fract(sin(dot(i3, vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i3 + vec3(1,0,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u3.x),
            mix(fract(sin(dot(i3 + vec3(0,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i3 + vec3(1,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5), u3.x), u3.y),
        mix(mix(fract(sin(dot(i3 + vec3(0,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i3 + vec3(1,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u3.x),
            mix(fract(sin(dot(i3 + vec3(0,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5),
                fract(sin(dot(i3 + vec3(1,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5), u3.x), u3.y), u3.z);

    let pattern = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

    let dark = vec3<f32>(0.15, 0.08, 0.03);
    let mid = vec3<f32>(0.35, 0.20, 0.10);
    let light = vec3<f32>(0.50, 0.35, 0.20);

    var barkColor = mix(dark, mid, smoothstep(0.3, 0.5, pattern));
    barkColor = mix(barkColor, light, smoothstep(0.55, 0.7, pattern));

    (*surface).baseColor = barkColor;
    (*surface).roughness = 0.9;`,
        },
        "bark",
    );
}
