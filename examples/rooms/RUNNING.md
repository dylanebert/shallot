# Rooms

Open the example, then **Quick join**. In this tab, **Add bot** puts a thin client in your room. **Drop connection** cuts your link; **Resume connection** returns you to your avatar if you resume within 10 seconds. Leaving ends your session and removes the bots you added.

To play with a friend, run `bun server.ts` in this directory and open the example with `?server=ws://127.0.0.1:3001`. Create a private room, then give its code to a friend who has opened the example against the same server. For another device, bind the server with `--host 0.0.0.0` and give the friend a reachable server address. On an HTTPS page, use a WSS endpoint behind a TLS reverse proxy that forwards both WebSocket upgrades and `/rooms`. `--origin` sets the allowed browser origin for room listing and code reservation; the demo defaults to `*`. Private codes are invitations, not accounts or authentication.

Click the scene to look with the mouse; WASD moves and Space jumps. On touch, the floating stick moves, a drag looks, and Jump jumps.
