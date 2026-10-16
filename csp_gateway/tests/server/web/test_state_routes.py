import threading

from csp_gateway import GatewayStruct


class _Row(GatewayStruct):
    x: int = 0


def test_state_route_does_not_block_event_loop():
    """A slow state query must leave the event loop free to serve other requests."""
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from csp_gateway.server.web.routes.state import add_state_routes

    entered = threading.Event()
    release = threading.Event()

    class _Channels:
        def query(self, field, indexer=None, query=None):
            entered.set()
            release.wait(timeout=10)
            return [_Row(x=1)]

    app = FastAPI()
    app.gateway = type("_Gateway", (), {"channels": _Channels()})()
    router = APIRouter()
    add_state_routes(router, field="records", model=_Row)
    app.include_router(router, prefix="/state")

    @app.get("/ping")
    async def ping():
        return "pong"

    with TestClient(app) as client:
        state_response = []
        slow = threading.Thread(target=lambda: state_response.append(client.get("/state/records")))
        slow.start()
        try:
            assert entered.wait(timeout=5)
            # The state query is parked mid-handler; an async route on the same loop must still answer.
            assert client.get("/ping").json() == "pong"
            assert slow.is_alive()
        finally:
            release.set()
            slow.join(timeout=10)
    assert [r["x"] for r in state_response[0].json()] == [1]
