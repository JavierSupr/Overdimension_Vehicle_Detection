import asyncio
import websockets
import base64
import json
import cv2

async def send_frame_via_websocket(websocket, frame, camera_name):
    """Send an encoded frame through WebSocket."""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    message = json.dumps({
        "type": "frame",
        "camera": camera_name,
        "data": frame_base64
    })
    await websocket.send(message)
    
    #except websockets.exceptions.ConnectionClosedError as e:
    #    print(f"WebSocket Connection closed with error: {e}")
    #except websockets.exceptions.ConnectionClosedOK:
    #    print("WebSocket closed normally")
    #except Exception as e:
    #    print(f"Unexpected error: {e}")
    #finally:
    #    print("WebSocket closed, releasing resources.")
    #
async def start_stream_server(frame1, frame2):
    print("try to start")
    async def handler(websocket, path):
        await video_stream(websocket, frame1, frame2)
        print("started")

    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()
    
