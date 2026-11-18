import { mkdir, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { TIMEOUT } from "../src/project/testing.js";

// Run against a checkout, including an older revision, without changing its sources.
const [checkoutPath, outputPath, injection] = process.argv.slice(2);
if (!checkoutPath || !outputPath || (injection && injection !== "--reject-timestamps"))
    throw new Error(
        "Usage: bun scripts/profile-safari.oracle.ts <checkout> <new-output-directory> [--reject-timestamps]",
    );
const root = resolve(checkoutPath);
const output = resolve(outputPath);
await mkdir(output);
async function run(cwd: string, ...cmd: string[]) {
    const status = await Bun.spawn(cmd, { cwd, stdout: "inherit", stderr: "inherit" }).exited;
    if (status) throw new Error(`${cmd.join(" ")} exited ${status}`);
}
const names = (await readdir(resolve(root, "examples"), { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((a, b) => (a === "bloom" ? -1 : b === "bloom" ? 1 : a.localeCompare(b)));
const pages = names.map((name, index) => ({
    name,
    cwd: root,
    entry: `examples/${name}`,
    port: 4223 + index,
}));
const rejectTimestamps = `
window.__timestampRejection={sets:0,writes:0,resolves:0};
const timestampSets=new WeakSet();
const create=GPUDevice.prototype.createQuerySet;
GPUDevice.prototype.createQuerySet=function(descriptor){
    if(descriptor.type!=='timestamp')return create.call(this,descriptor);
    const set=create.call(this,{...descriptor,type:'occlusion'});
    timestampSets.add(set);__timestampRejection.sets++;return set;
};
for(const method of ['beginRenderPass','beginComputePass']){
    const begin=GPUCommandEncoder.prototype[method];
    GPUCommandEncoder.prototype[method]=function(descriptor){
        if(descriptor?.timestampWrites)__timestampRejection.writes++;
        return begin.call(this,descriptor);
    };
}
const resolveQueries=GPUCommandEncoder.prototype.resolveQuerySet;
GPUCommandEncoder.prototype.resolveQuerySet=function(set,first,count,buffer,offset){
    if(timestampSets.has(set)){
        __timestampRejection.resolves++;
        return resolveQueries.call(this,set,0,set.count+1,buffer,offset);
    }
    return resolveQueries.call(this,set,first,count,buffer,offset);
};`;
for (const page of pages) {
    const dist = resolve(output, `${page.name}-dist`);
    await run(page.cwd, "bunx", "vite", "build", page.entry, "--outDir", dist);
    const html = resolve(dist, "index.html");
    await Bun.write(
        html,
        (await Bun.file(html).text()).replace(
            "<head>",
            `<head><script>
window.__safariErrors=[];
addEventListener('error',e=>__safariErrors.push(e.message));
addEventListener('unhandledrejection',e=>__safariErrors.push(String(e.reason?.stack||e.reason)));
${injection ? rejectTimestamps : ""}
</script>`,
        ),
    );
}
const processes: ReturnType<typeof Bun.spawn>[] = [];
let session: string | undefined;
const driver = "http://127.0.0.1:4796";
async function request<T>(path: string, body?: unknown): Promise<T> {
    const response = await fetch(
        driver + path,
        body === undefined
            ? {}
            : {
                  method: "POST",
                  body: JSON.stringify(body),
              },
    );
    const result = (await response.json()) as { value: T & { error?: string; message?: string } };
    if (result.value?.error) throw new Error(JSON.stringify(result.value));
    return result.value;
}
async function waitFor(predicate: () => Promise<boolean>, label: string) {
    const start = performance.now();
    while (!(await predicate())) {
        if (performance.now() - start > TIMEOUT.startup) throw new Error(`Timed out: ${label}`);
        await Bun.sleep(100);
    }
}
const execute = <T>(script: string, args: unknown[] = []) =>
    request<T>(`/session/${session}/execute/sync`, { script, args });
const stateScript = `const p=document.querySelector('[data-shallot-profile]');return {
url:location.href,visibility:document.visibilityState,focus:document.hasFocus(),
profile:p?.dataset.gpuTiming||null,errors:window.__safariErrors,injection:window.__timestampRejection,
body:document.body.innerText,firstUpdate:performance.getEntriesByType('measure').some(e=>e.name.endsWith(':first-update')),
marks:performance.getEntries().filter(e=>e.entryType==='mark'||e.entryType==='measure').map(e=>[e.name,e.duration])};`;
try {
    for (const page of pages)
        processes.push(
            Bun.spawn(
                [
                    "bunx",
                    "vite",
                    "preview",
                    page.entry,
                    "--outDir",
                    resolve(output, `${page.name}-dist`),
                    "--port",
                    String(page.port),
                    "--strictPort",
                    "--host",
                    "127.0.0.1",
                ],
                { cwd: page.cwd, stdout: "inherit", stderr: "inherit" },
            ),
        );
    processes.push(
        Bun.spawn(["/usr/bin/safaridriver", "-p", "4796"], {
            stdout: "inherit",
            stderr: "inherit",
        }),
    );
    await waitFor(async () => {
        try {
            return (await fetch(`${driver}/status`)).ok;
        } catch {
            return false;
        }
    }, "safaridriver startup");
    const created = await request<{ sessionId: string; capabilities: unknown }>("/session", {
        capabilities: { alwaysMatch: { browserName: "safari" } },
    });
    session = created.sessionId;
    console.log(
        JSON.stringify({ root, injection: !!injection, capabilities: created.capabilities }),
    );
    await request(`/session/${session}/window/rect`, { x: 0, y: 0, width: 1000, height: 800 });
    for (const page of pages) {
        const url = `http://127.0.0.1:${page.port}/`;
        await waitFor(async () => {
            try {
                return (await fetch(url)).ok;
            } catch {
                return false;
            }
        }, `${page.name} preview startup`);
        await request(`/session/${session}/url`, { url });
        const visibility = await execute<string>("return document.visibilityState");
        if (visibility !== "visible") {
            const png = await request<string>(`/session/${session}/screenshot`);
            await Bun.write(resolve(output, `${page.name}-hidden.png`), Buffer.from(png, "base64"));
            await Bun.write(
                resolve(output, `${page.name}-hidden.json`),
                JSON.stringify(await execute(stateScript), null, 2),
            );
            throw new Error(
                `${page.name}: Safari document is ${visibility}; native animation frames may be suspended. Make the automation window visible and rerun with a new output directory.`,
            );
        }
        await waitFor(
            () =>
                execute<boolean>(
                    "return performance.getEntriesByName('shallot:handoff','mark').length>0 && performance.getEntriesByType('measure').some(e=>e.name.endsWith(':first-update'))",
                ),
            `${page.name} first update`,
        );
        const hasProfiler = await execute<boolean>(
            "return performance.getEntriesByType('measure').some(e=>e.name.startsWith('shallot:system:Profile/'))",
        );
        for (const open of hasProfiler ? [false, true] : [false]) {
            const shown = await execute<boolean>(
                "return !!document.querySelector('[data-shallot-profile]')",
            );
            if (shown !== open)
                await execute(
                    "for(const type of ['keydown','keyup'])window.dispatchEvent(new KeyboardEvent(type,{key:'F3',code:'F3',bubbles:true,cancelable:true}));",
                );
            await waitFor(
                () =>
                    execute<boolean>(
                        `return !!document.querySelector('[data-shallot-profile]')===${open}`,
                    ),
                `${page.name} profiler ${open ? "open" : "closed"}`,
            );
            const state = await execute<{ errors: string[]; visibility: string }>(stateScript);
            if (state.visibility !== "visible") throw new Error(`${page.name}: hidden document`);
            // Keep layout, but exclude DOM controls and the profiler from pixel evidence.
            const box = await execute<number[]>(`const style=document.createElement('style');
style.id='safari-scene-only';style.textContent='body * { visibility: hidden !important } canvas { visibility: visible !important }';
document.head.append(style);const r=document.querySelector('canvas').getBoundingClientRect();
return [r.x/innerWidth,r.y/innerHeight,r.width/innerWidth,r.height/innerHeight];`);
            const png = await request<string>(`/session/${session}/screenshot`);
            const label = `${page.name}-${open ? "open" : "closed"}`;
            await Bun.write(resolve(output, `${label}.png`), Buffer.from(png, "base64"));
            const deviation = await request<number[]>(`/session/${session}/execute/async`, {
                args: [png, box],
                script: `const [png,box,done]=arguments;const image=new Image();image.onload=()=>{
const c=document.createElement('canvas');c.width=image.width;c.height=image.height;
const ctx=c.getContext('2d');ctx.drawImage(image,0,0);
// Inset the canvas to exclude page edges and embedded-scene framing.
box[0]+=box[2]*.1;box[1]+=box[3]*.1;box[2]*=.8;box[3]*=.8;
const data=ctx.getImageData(Math.floor(c.width*box[0]),Math.floor(c.height*box[1]),Math.floor(c.width*box[2]),Math.floor(c.height*box[3])).data;
const sum=[0,0,0],squares=[0,0,0],n=data.length/4;
for(let i=0;i<data.length;i+=4)for(let j=0;j<3;j++){sum[j]+=data[i+j];squares[j]+=data[i+j]**2;}
done(sum.map((s,j)=>Math.sqrt(Math.max(0,squares[j]/n-(s/n)**2))));
};image.src='data:image/png;base64,'+png;`,
            });
            await Bun.write(
                resolve(output, `${label}.json`),
                JSON.stringify({ state, deviation }, null, 2),
            );
            console.log(JSON.stringify({ label, state, deviation }));
            await execute("document.querySelector('#safari-scene-only').remove()");
            if (state.errors.length)
                throw new Error(`${label}: page errors: ${state.errors.join("; ")}`);
            if (Math.max(...deviation) <= 10)
                throw new Error(`${label}: blank scene region (RGB deviations ${deviation})`);
        }
    }
} finally {
    if (session) await fetch(`${driver}/session/${session}`, { method: "DELETE" }).catch(() => {});
    for (const process of processes) process.kill();
    await Promise.all(processes.map((process) => process.exited));
}
