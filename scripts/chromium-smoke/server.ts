const html = `<!doctype html>
<html lang="en">
    <head><meta charset="UTF-8"><title>Bare WebGPU smoke</title></head>
    <body><canvas id="surface" width="64" height="64"></canvas></body>
</html>`;

Bun.serve({
    port: 4176,
    fetch: () => new Response(html, { headers: { "content-type": "text/html" } }),
});
