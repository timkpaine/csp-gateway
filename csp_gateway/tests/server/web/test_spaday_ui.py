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
        assert '"method": "saveClean"' in tree
        assert "Download layout" in tree
        assert '"kind": "download"' in tree

    def test_layout_actions_are_declarative(self, client: TestClient):
        # the handlers formerly shipped in actions.js are serializable actions in the tree now
        page = client.get("/").text
        assert "actions.js" not in page
        assert "cspGatewayCustomLayout" not in page
        assert 'localStorage.getItem("csp_gateway_demo_config")' in page
        assert client.get("/components/csp-gateway/actions.js").status_code == 404
        tree = client.get("/tree.json").text
        # save: clean-save the workspace, persist it, and switch the selector to the custom layout
        assert '"target": {"ref": "id", "id": "gateway-workspace"}, "method": "saveClean", "result": "custom_layout"' in tree
        assert '"kind": "set-storage", "key": "csp_gateway_demo_config", "value": {"expr": "field", "name": "custom_layout"}' in tree
        assert '"kind": "set-field", "field": "layout_view", "value": {"expr": "lit", "value": "Custom Layout"}' in tree
        # download: clean-save, then offer the result as a client-side file
        assert '"method": "saveClean", "result": "download_layout"' in tree
        assert (
            '"kind": "download", "filename": {"expr": "lit", "value": "layout.json"}, "value": {"expr": "field", "name": "download_layout"}' in tree
        )

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


class TestMainTabs:
    """The main window's tab layout: workspace + on-demand closeable tabs (graph, send)."""

    @pytest.fixture(scope="class")
    def gateway(self, free_port):
        from csp_gateway import MountChannelsGraph

        return Gateway(
            modules=[
                TwoSendModule(),
                MountRestRoutes(force_mount_all=True),
                MountSendForm(),
                MountChannelsGraph(),
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

    def test_main_region_is_a_tab_layout(self, client: TestClient):
        tree = json.loads(client.get("/tree.json").text)
        text = json.dumps(tree)
        # the workspace, the graph tab, and the send tab are frames of one regular-layout
        assert '"spaday-regular-layout"' in text
        assert '"workspace"' in text and '"channels-graph"' in text and '"send"' in text
        # the layout opens with just the workspace; other frames wait for their tab_button
        assert '{"Str": "tab-layout"}' in text
        assert text.count("regular-layout-frame") >= 3

    def test_tab_chrome_only_with_more_tabs(self, client: TestClient):
        tree = client.get("/tree.json").text
        # solo-mode class is computed from main_tabbed, synced from regular-layout-update
        assert "spa spa-solo" in tree
        assert "main_tabbed" in tree
        assert "regular-layout-update" in tree

    def test_graph_tab_renders_the_channels_graph(self, client: TestClient):
        tree = json.loads(client.get("/tree.json").text)
        text = json.dumps(tree)
        assert '"spaday-dagre"' in text
        # channels and modules become classed nodes with edges between them
        assert "gateway-channel" in text and "gateway-module" in text
        # classic dagre-d3 parity: diamond channels, classed setter/getter edges
        assert '"shape": {"Str": "diamond"}' in text
        assert "gateway-sets" in text and "gateway-gets" in text

    def test_main_layout_is_locked(self, client: TestClient):
        tree = client.get("/tree.json").text
        # tabs select and close but cannot be drag-rearranged
        assert '"locked": {"Bool": true}' in tree

    def test_workspace_tab_is_not_closeable(self, client: TestClient):
        tree = json.loads(client.get("/tree.json").text)

        def frames(node):
            if isinstance(node, dict):
                if node.get("tag") == "regular-layout-frame":
                    yield node
                for value in node.values():
                    yield from frames(value)
            elif isinstance(node, list):
                for value in node:
                    yield from frames(value)

        # nothing reopens a closed workspace, so its frame hides the close control;
        # reopenable tabs (graph, send) keep theirs
        by_name = {frame["props"]["name"]["Str"]: frame["props"] for frame in frames(tree)}
        assert "data-no-close" in by_name["workspace"]
        assert "data-no-close" not in by_name["channels-graph"]
        assert "data-no-close" not in by_name["send"]
        page = client.get("/").text
        assert "regular-layout-frame[data-no-close]::part(close) { display: none; }" in page

    def test_graph_edge_styles_match_the_classic_page(self, client: TestClient):
        page = client.get("/").text
        assert ".gateway-sets .spaday-dagre-edge-line { stroke: #f66;" in page
        assert "stroke-dasharray: 5 5" in page

    def test_plus_and_graph_buttons_open_tabs(self, client: TestClient):
        tree = client.get("/tree.json").text
        assert '"data-tab": {"Str": "send"}' in tree
        assert '"data-tab": {"Str": "channels-graph"}' in tree
        # the bottom drawer is gone in the spaday provider (send lives in a tab now)
        assert "gateway-send" not in tree
        # each button invokes the layout engine's openPanel with its own data-tab
        assert (
            '"target": {"ref": "id", "id": "gateway-main-layout"}, "method": "openPanel", '
            '"args": [{"expr": "event-prop", "path": "currentTarget.dataset.tab"}]'
        ) in tree

    def test_dagre_and_layout_packages_are_mounted(self, client: TestClient):
        page = client.get("/").text
        assert "/components/dagre/cdn/index.js" in page
        assert "/components/regular-layout/cdn/index.js" in page


class TestMainWithoutTabs:
    """With no registered tabs and no send panel, the main region stays the plain workspace."""

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

    def test_plain_main_without_tabs(self, client: TestClient):
        tree = client.get("/tree.json").text
        assert "spaday-regular-layout" not in tree
        assert "spaday-dagre" not in tree
