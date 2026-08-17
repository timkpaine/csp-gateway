"""Tests for the spaday UI's transports wire: namespacing and per-identity tenancy."""

import json
from typing import Any

import csp
import pytest
from csp import ts
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from csp_gateway import (
    Gateway,
    GatewayChannels,
    GatewayModule,
    GatewaySettings,
    GatewayStruct,
    MountRestRoutes,
)
from csp_gateway.server.middleware.api_key_external import MountExternalAPIKeyMiddleware
from csp_gateway.testing.mock_validators import mock_api_key_validator_by_user

pytest.importorskip("spaday")
pytest.importorskip("transports")


class Example(GatewayStruct):
    value: float = 0.0


class ExampleChannels(GatewayChannels):
    example: ts[Example] = None


class ExampleModule(GatewayModule):
    def connect(self, channels: ExampleChannels) -> None:
        channels.set_channel("example", csp.null_ts(Example))


class Rows(BaseModel):
    rows: list = []


class PublishingModule(GatewayModule):
    """Declares a live model and publishes a per-identity view of it."""

    # GatewayModule is a pydantic model, so the handle needs a declared field to live on.
    handle: Any = None

    def connect(self, channels: ExampleChannels) -> None: ...

    def ui(self, app) -> None:
        self.handle = app.model("staging", Rows)

    def publish_per_identity(self) -> None:
        # Each tenant only ever sees rows carrying its own user.
        self.handle.publish(lambda identity: {"rows": [{"user": (identity or {}).get("user", "anon")}]})


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class TestWireDeclaration:
    def test_no_model_declared_serves_no_websocket(self):
        gateway = Gateway(
            modules=[ExampleModule(), MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            client = TestClient(gateway.web_app.get_fastapi())
            with pytest.raises(WebSocketDisconnect) as excinfo, client.websocket_connect("/ws"):
                pass
            # 1000 is Starlette closing an unrouted websocket. Asserting the code keeps this from
            # passing on a 1008 rejection, which would mean the route exists but refuses every client.
            assert excinfo.value.code == 1000
        finally:
            gateway.stop()

    def test_declaring_a_model_serves_a_websocket(self):
        gateway = Gateway(
            modules=[ExampleModule(), PublishingModule(), MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            client = TestClient(gateway.web_app.get_fastapi())
            with client.websocket_connect("/ws") as ws:
                snapshot = json.loads(ws.receive_text())
            assert snapshot["t"] == "snapshot"
            assert snapshot["type"] == "Rows"
        finally:
            gateway.stop()

    def test_the_wire_carries_state_published_before_the_client_connected(self):
        publisher = PublishingModule()
        gateway = Gateway(
            modules=[ExampleModule(), publisher, MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            publisher.publish_per_identity()
            client = TestClient(gateway.web_app.get_fastapi())
            with client.websocket_connect("/ws") as ws:
                snapshot = json.loads(ws.receive_text())
            assert snapshot["value"] == {"Map": {"rows": {"List": [{"Map": {"user": {"Str": "anon"}}}]}}}
        finally:
            gateway.stop()

    def test_namespaces_cannot_collide(self):
        gateway = Gateway(
            modules=[ExampleModule(), PublishingModule(), MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            with pytest.raises(ValueError, match="already declared"):
                gateway.web_app.ui.model("staging", Rows)
        finally:
            gateway.stop()

    def test_page_wires_the_declared_namespace(self):
        gateway = Gateway(
            modules=[ExampleModule(), PublishingModule(), MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            page = TestClient(gateway.web_app.get_fastapi()).get("/").text
            assert "staging" in page
            assert "/ws" in page
        finally:
            gateway.stop()


class TestTenancy:
    """Per-identity tenancy is what keeps the wire from bypassing AuthFilterMiddleware."""

    @pytest.fixture(scope="class")
    def gateway(self):
        gateway = Gateway(
            modules=[
                ExampleModule(),
                PublishingModule(),
                MountRestRoutes(force_mount_all=True),
                MountExternalAPIKeyMiddleware(external_validator=mock_api_key_validator_by_user),
            ],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=_free_port(), UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield gateway
        finally:
            gateway.stop()

    def test_unauthenticated_connections_share_the_anonymous_tenant(self, gateway):
        import asyncio

        from csp_gateway.server.web.spaday_ui import _ANONYMOUS_TENANT

        ui = gateway.web_app.ui

        class _Sock:
            cookies: dict = {}
            headers: dict = {}
            query_params: dict = {}

        tenant, identity = asyncio.run(ui._tenant_for(_Sock()))
        assert tenant == _ANONYMOUS_TENANT
        assert identity is None

    def test_distinct_identities_get_distinct_tenants(self, gateway):
        ui = gateway.web_app.ui
        ui._host_tenant("alice", {"user": "alice"})
        ui._host_tenant("bob", {"user": "bob"})

        module = next(m for m in gateway.modules if isinstance(m, PublishingModule))
        module.publish_per_identity()

        alice_model, _ = ui._tenant_models["alice"]["staging"]
        bob_model, _ = ui._tenant_models["bob"]["staging"]
        assert alice_model.rows == [{"user": "alice"}]
        assert bob_model.rows == [{"user": "bob"}]

    def test_a_new_tenant_is_seeded_with_current_state(self, gateway):
        ui = gateway.web_app.ui
        module = next(m for m in gateway.modules if isinstance(m, PublishingModule))
        module.publish_per_identity()

        ui._host_tenant("carol", {"user": "carol"})
        carol_model, _ = ui._tenant_models["carol"]["staging"]
        assert carol_model.rows == [{"user": "carol"}]

    def test_publishing_before_the_loop_is_up_is_not_lost(self, gateway):
        ui = gateway.web_app.ui
        module = next(m for m in gateway.modules if isinstance(m, PublishingModule))
        ui._loop = None  # as during startup, before any connection
        module.handle.publish({"rows": [{"user": "seeded"}]})

        ui._host_tenant("dave", {"user": "dave"})
        dave_model, _ = ui._tenant_models["dave"]["staging"]
        assert dave_model.rows == [{"user": "seeded"}]
