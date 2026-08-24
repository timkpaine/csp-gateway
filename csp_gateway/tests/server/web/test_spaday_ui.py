"""Tests for the optional spaday UI provider (`Settings.UI_PROVIDER == "spaday"`)."""

import json
from datetime import timedelta
from enum import Enum, auto

import csp
import pytest
from csp import ts
from fastapi.testclient import TestClient

from csp_gateway import (
    ChannelSelection,
    Gateway,
    GatewayChannels,
    GatewayModule,
    GatewaySettings,
    GatewayStruct,
    MountPerspectiveTables,
    MountRestRoutes,
    MountSendForm,
)
from csp_gateway.server.middleware.api_key_external import MountExternalAPIKeyMiddleware
from csp_gateway.testing.mock_validators import mock_api_key_validator_by_user

pytest.importorskip("spaday")


class Example(GatewayStruct):
    value: float


class ExampleChannels(GatewayChannels):
    example: ts[Example] = None


class ExampleModule(GatewayModule):
    @csp.node
    def _produce(self, trigger: ts[bool]) -> ts[Example]:
        if csp.ticked(trigger):
            return Example(value=1.0)

    def connect(self, channels: ExampleChannels) -> None:
        channels.set_channel("example", self._produce(csp.timer(interval=timedelta(seconds=0.1), value=True)))


class TestSpadayAuth:
    """The spaday page and tree must sit behind the same auth as the default UI (review finding #1)."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        return Gateway(
            modules=[
                ExampleModule(),
                MountRestRoutes(force_mount_all=True),
                MountExternalAPIKeyMiddleware(external_validator=mock_api_key_validator_by_user),
            ],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_unauthenticated_does_not_serve_spaday(self, client: TestClient):
        # The auth middleware intercepts the page and tree (they carry the same dependencies as the
        # default UI), so an unauthenticated request never receives the spaday app or its tree.
        assert "spa-app" not in client.get("/").text
        assert not client.get("/tree.json").headers["content-type"].startswith("application/json")

    def test_authenticated_serves_spaday(self, client: TestClient):
        # With a valid key the page renders (200) and the tree is JSON — the provider-gated smoke test.
        page = client.get("/?token=alice_key")
        assert page.status_code == 200
        assert "spa-app" in page.text
        tree = client.get("/tree.json?token=alice_key")
        assert tree.status_code == 200
        assert tree.headers["content-type"].startswith("application/json")


class TwoSendChannels(GatewayChannels):
    alpha: ts[Example] = None
    beta: ts[Example] = None


class TwoSendModule(GatewayModule):
    @csp.node
    def _tick(self, trigger: ts[bool]) -> ts[Example]:
        if csp.ticked(trigger):
            return Example(value=1.0)

    def connect(self, channels: TwoSendChannels) -> None:
        trig = csp.timer(interval=timedelta(seconds=0.1), value=True)
        channels.set_channel("alpha", self._tick(trig))
        channels.set_channel("beta", self._tick(trig))
        channels.add_send_channel("alpha")
        channels.add_send_channel("beta")


class TestSpadaySendFormSelection:
    """The send panel only offers channels whose send route is mounted (review finding #4)."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        # `beta` has a send adapter but its send route is not mounted, so no form should be shown for it.
        return Gateway(
            modules=[
                TwoSendModule(),
                MountRestRoutes(mount_send=ChannelSelection(include=["alpha"])),
                MountSendForm(mount_send=ChannelSelection(include=["alpha"])),
            ],
            channels=TwoSendChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_only_mounted_send_channel_has_a_form(self, client: TestClient):
        tree = client.get("/tree.json").text
        assert "alpha" in tree
        assert "beta" not in tree


class SendableModule(GatewayModule):
    @csp.node
    def _tick(self, trigger: ts[bool]) -> ts[Example]:
        if csp.ticked(trigger):
            return Example(value=1.0)

    def connect(self, channels: ExampleChannels) -> None:
        channels.set_channel("example", self._tick(csp.timer(interval=timedelta(seconds=0.1), value=True)))
        channels.add_send_channel("example")


class TestSpadayRootPath:
    """Provider-generated URLs are prefixed with ROOT_PATH for reverse-proxy sub-path serving (finding #3)."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        return Gateway(
            modules=[SendableModule(), MountRestRoutes(force_mount_all=True), MountSendForm()],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday", ROOT_PATH="/watchtower"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_page_assets_prefixed(self, client: TestClient):
        # The page's own runtime/asset URLs (/js, wasm) carry the ROOT_PATH prefix.
        assert "/watchtower/js" in client.get("/").text

    def test_module_urls_prefixed(self, client: TestClient):
        # A module-generated action URL (the send POST) is prefixed in the component tree.
        assert "/watchtower/api/v1/send/example" in client.get("/tree.json").text


class Order(GatewayStruct):
    symbol: str
    secret: str


class BasketKey(Enum):
    A = auto()
    B = auto()


class DetailChannels(GatewayChannels):
    orders: ts[Order] = None
    basket: dict[BasketKey, ts[Order]] = None


class DetailModule(GatewayModule):
    @csp.node
    def _order(self, trigger: ts[bool]) -> ts[Order]:
        if csp.ticked(trigger):
            return Order(symbol="AAPL", secret="hidden")

    def connect(self, channels: DetailChannels) -> None:
        trig = csp.timer(interval=timedelta(seconds=0.1), value=True)
        channels.set_channel("orders", self._order(trig))
        channels.add_send_channel("orders")
        channels.set_channel("basket", self._order(trig), BasketKey.A)
        channels.set_channel("basket", self._order(trig), BasketKey.B)
        channels.add_send_channel("basket", BasketKey.A)
        channels.add_send_channel("basket", BasketKey.B)


class TestSpadaySendFormDetails:
    """Send-form field overrides and dict-basket keyed URLs (finding #4 test gaps)."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        return Gateway(
            modules=[
                DetailModule(),
                MountRestRoutes(force_mount_all=True),
                MountSendForm(form_overrides={"symbol": {"label": "Ticker"}, "secret": {"exclude": True}}),
            ],
            channels=DetailChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_field_overrides(self, client: TestClient):
        tree = client.get("/tree.json").text
        assert "Ticker" in tree  # `symbol` relabelled via form_overrides
        assert "secret" not in tree  # `secret` excluded via form_overrides (id/timestamp are always filtered)

    def test_keyed_basket_url(self, client: TestClient):
        # A dict-basket send targets /send/{channel}/{key}: the base URL + the per-channel key field bind.
        tree = client.get("/tree.json").text
        assert "/api/v1/send/basket" in tree
        assert "send_key_basket" in tree


class TestDefaultLayout:
    """The generated layout is a Perspective 5 whole-element config."""

    def test_one_datagrid_panel_per_table(self):
        from csp_gateway.server.web.spaday_ui import GatewayUI

        layout = GatewayUI._default_layout(["orders", "fills"])
        assert layout["layout"] == {"type": "tab-layout", "tabs": ["CSP_GATEWAY_0", "CSP_GATEWAY_1"], "selected": 0}
        assert layout["panels"] == {
            "CSP_GATEWAY_0": {"table": "orders", "plugin": "Datagrid", "title": "orders"},
            "CSP_GATEWAY_1": {"table": "fills", "plugin": "Datagrid", "title": "fills"},
        }

    def test_schemas_add_legacy_parity_sort_and_columns(self):
        # Legacy-UI parity: timestamp sorts descending and the id column is hidden.
        from csp_gateway.server.web.spaday_ui import GatewayUI

        layout = GatewayUI._default_layout(
            ["orders"],
            schemas={"orders": {"id": "string", "timestamp": "datetime", "price": "float"}},
        )
        panel = layout["panels"]["CSP_GATEWAY_0"]
        assert panel["sort"] == [["timestamp", "desc"]]
        assert panel["columns"] == ["timestamp", "price"]

    def test_schema_without_timestamp_or_id_stays_bare(self):
        from csp_gateway.server.web.spaday_ui import GatewayUI

        layout = GatewayUI._default_layout(["orders"], schemas={"orders": {"price": "float"}})
        panel = layout["panels"]["CSP_GATEWAY_0"]
        assert "sort" not in panel
        assert "columns" not in panel

    def test_custom_default_layout_replaces_generated_layout(self):
        from csp_gateway.server.web.spaday_ui import GatewayUI

        ui = object.__new__(GatewayUI)
        ui._store_seeds = {}
        ui._settings = GatewaySettings()
        custom = {
            "layout": {"type": "tab-layout", "tabs": ["custom"]},
            "panels": {"custom": {"table": "orders", "plugin": "X Bar"}},
        }

        panel = ui.perspective_panel(route="/perspective", tables=["orders"], default_layout=custom)
        node = panel.to_node()

        assert "X Bar" in json.dumps(node)
        assert "CSP_GATEWAY_0" not in json.dumps(node)


class TestDarkBoot:
    """The page seeds `dark` from the browser's prefers-color-scheme, like the legacy UI."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        return Gateway(
            modules=[ExampleModule(), MountRestRoutes(force_mount_all=True)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_dark_seed_is_client_evaluated(self, client: TestClient):
        page = client.get("/").text
        assert '"dark": (matchMedia("(prefers-color-scheme: dark)").matches)' in page

    def test_dark_choice_persists_across_reloads(self, client: TestClient):
        # A manual toggle is stored and overrides the browser preference on the next load (legacy parity).
        page = client.get("/").text
        assert 'localStorage.getItem("csp-gateway:dark")' in page
        assert 'store.subscribe("dark", (v) => { try { localStorage.setItem("csp-gateway:dark", JSON.stringify(v)); } catch {} });' in page


class TestSpadayPerspectiveLayoutActions:
    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        return Gateway(
            modules=[ExampleModule(), MountPerspectiveTables()],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )

    @pytest.fixture(scope="class")
    def client(self, gateway):
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_layout_actions_are_available_without_server_layouts(self, client: TestClient):
        tree = client.get("/tree.json").text
        assert "Custom Layout" in tree
        assert "gateway-layout-selector" in tree
        assert "layout_view" in tree
        assert "Save current layout" in tree
        assert "csp-gateway:save-layout" in tree
        assert "Download layout" in tree
        assert "csp-gateway:download-layout" in tree

    def test_layout_action_script_is_served(self, client: TestClient):
        page = client.get("/").text
        assert "/components/csp-gateway/actions.js" in page
        assert "/components/csp-gateway-perspective-charts/spaday-charts.js" in page
        assert "globalThis.cspGatewayCustomLayout" in page
        assert '"gateway-workspace"' in client.get("/tree.json").text
        script = client.get("/components/csp-gateway/actions.js")
        assert script.status_code == 200
        assert 'from "../../js/cdn/index.js"' in script.text
        assert '"csp-gateway:save-layout"' in script.text
        assert '"csp-gateway:download-layout"' in script.text
        assert '"csp_gateway_demo_config"' in script.text
        assert '"gateway-layout-selector"' in script.text
        assert 'new Event("input", { bubbles: true })' in script.text
        assert '"gateway-workspace"' in script.text
        assert client.get("/js/cdn/index.js").status_code == 200
        charts = client.get("/components/csp-gateway-perspective-charts/spaday-charts.js")
        assert charts.status_code == 200
        assert "X Bar" in charts.text
        assert "Treemap" in charts.text

    def test_layout_download_is_a_same_origin_attachment(self, client: TestClient):
        layout = {"layout": {"type": "tab-layout", "tabs": []}, "panels": {}}

        response = client.post("/api/v1/perspective/download-layout", data={"layout": json.dumps(layout)})

        assert response.status_code == 200
        assert response.json() == layout
        assert response.headers["content-disposition"] == 'attachment; filename="layout.json"'
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_layout_download_rejects_invalid_and_oversized_content(self, client: TestClient):
        invalid = client.post("/api/v1/perspective/download-layout", data={"layout": "1"})
        oversized = client.post(
            "/api/v1/perspective/download-layout",
            content=b"x",
            headers={"content-length": str(16 * 1024 * 1024 + 1)},
        )

        assert invalid.status_code == 400
        assert oversized.status_code == 413
