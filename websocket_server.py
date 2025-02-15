import asyncio
import websockets

# WebSocket Server Config
WEBSOCKET_PORT = 8765
connected_clients = set()

async def websocket_handler(websocket):
    """Handles incoming WebSocket connections and manages frame broadcasting."""
    connected_clients.add(websocket)
    print(f"[INFO] WebSocket client connected. Total clients: {len(connected_clients)}")

    try:
        async for message in websocket:
            print(f"[WebSocket] Received frame of size: {len(message)} bytes")

            # Broadcast the frame to all clients except the sender
            disconnected_clients = set()
            for client in connected_clients:
                if client != websocket:
                    try:
                        await client.send(message)
                    except:
                        print("[ERROR] Failed to send frame. Removing client.")
                        disconnected_clients.add(client)

            # Remove disconnected clients
            connected_clients.difference_update(disconnected_clients)

    except websockets.exceptions.ConnectionClosed:
        print("[INFO] WebSocket client disconnected.")
    finally:
        connected_clients.remove(websocket)
        print(f"[INFO] Remaining clients: {len(connected_clients)}")

async def start_websocket_server():
    """Starts the WebSocket server."""
    print(f"[INFO] WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
    async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT):
        await asyncio.Future()  # Keeps the server running indefinitely

if __name__ == "__main__":
    asyncio.run(start_websocket_server())
