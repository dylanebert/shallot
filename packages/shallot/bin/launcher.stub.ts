// A vite-free stand-in for the editor dev server, used only as the launcher tests' `command` override.
// It binds the port the launcher allocated and answers every request with the dir it was launched for,
// so a test can prove the launcher spawned a server bound to the right dir without booting the editor.

const dir = process.argv[2] ?? "";
const port = Number(process.argv[3]);

Bun.serve({ port, fetch: () => new Response(dir) });
