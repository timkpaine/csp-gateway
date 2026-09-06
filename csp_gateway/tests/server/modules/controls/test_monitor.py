"""Tests for `MountProcessMonitor`, the spaday "Process" tab over the `stats` control."""

from datetime import timedelta

import csp
import pytest
from csp import ts
from fastapi.testclient import TestClient

from csp_gateway import (
    Gateway,
    GatewayChannels,
    GatewayModule,
    GatewaySettings,
    GatewayStruct,
    MountControls,
    MountProcessMonitor,
)
from csp_gateway.utils import Controls

pytest.importorskip("spaday")


class Example(GatewayStruct):
    value: float


class ExampleChannels(GatewayChannels):
    controls: ts[Controls] = None
    example: ts[Example] = None


class ExampleModule(GatewayModule):
    @csp.node
    def _produce(self, trigger: ts[bool]) -> ts[Example]:
        if csp.ticked(trigger):
            return Example(value=1.0)

    def connect(self, channels: ExampleChannels) -> None:
        channels.set_channel("example", self._produce(csp.timer(interval=timedelta(seconds=0.1), value=True)))


def _client(modules, free_port):
    gateway = Gateway(
        modules=modules,
        channels=ExampleChannels(),
        settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
    )
    gateway.start(rest=True, ui=True, _in_test=True)
    return gateway, TestClient(gateway.web_app.get_fastapi())


class TestProcessMonitor:
    @pytest.fixture(scope="class")
    def client(self, free_port):
        gateway, client = _client([ExampleModule(), MountControls(), MountProcessMonitor()], free_port)
        try:
            yield client
        finally:
            gateway.stop()

    def test_tab_is_registered(self, client):
        tree = client.get("/tree.json").text
        assert '"monitor"' in tree
        assert "Process" in tree

    def test_panel_reads_the_stats_control(self, client):
        tree = client.get("/tree.json").text
        assert "/api/v1/controls/stats" in tree

    def test_stats_endpoint_backs_the_panel(self, client):
        """The fields the panel binds are the ones the control actually reports."""
        data = client.get("/api/v1/controls/stats").json()[0]["data"]
        for key in ("host", "user", "pid", "cpu", "memory", "memory-total", "active_threads", "max_threads"):
            assert key in data


class TestProcessMonitorWithoutControls:
    """Without MountControls the stats route is absent; the panel must say so, not render blanks."""

    @pytest.fixture(scope="class")
    def client(self, free_port):
        gateway, client = _client([ExampleModule(), MountProcessMonitor()], free_port)
        try:
            yield client
        finally:
            gateway.stop()

    def test_no_stats_route(self, client):
        assert client.get("/api/v1/controls/stats").status_code == 404

    def test_panel_carries_the_missing_callout(self, client):
        assert "No stats available" in client.get("/tree.json").text
