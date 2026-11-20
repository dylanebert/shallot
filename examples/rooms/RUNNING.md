# Rooms

In this directory, run `bun server.ts`, then open **http://127.0.0.1:3001/** in every tab. Quick join picks a public room. To play together in private, create a room, copy its code, and have your friend choose **Enter code** at that same address.

For another device, use `bun server.ts --host 0.0.0.0` and open `http://<server-ip>:3001/` on both devices. A separately hosted copy can still use `?server=ws://<server-ip>:3001`. From an HTTPS page, use `wss://` and forward both WebSocket upgrades and `/rooms` through your TLS reverse proxy. `--origin` sets the allowed browser origin for cross-origin room reservation; the demo defaults to `*`.

On the static site or with `bunx vite`, rooms stay in one tab. Open an empty party slot to add a bot. **Drop connection**, then **Resume connection** within 10 seconds, to keep your seat.

Click the scene to look with the mouse; WASD moves and Space jumps. On touch, the floating stick moves, a drag looks, and Jump jumps.
