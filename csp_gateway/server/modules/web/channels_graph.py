from json import dumps
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import HTMLResponse

from csp_gateway.server import GatewayChannels, GatewayModule

# separate to avoid circular
from csp_gateway.server.web import GatewayWebApp

if TYPE_CHECKING:
    from csp_gateway.server.web.spaday_ui import GatewayUI

_GRAPH_TAB = "channels-graph"


class MountChannelsGraph(GatewayModule):
    route: str = "/channels_graph"

    _channels: GatewayChannels | None = None

    def connect(self, channels: GatewayChannels) -> None:
        # Keep the channels handle so the spaday UI can render the graph structure.
        self._channels = channels

    def rest(self, app: GatewayWebApp) -> None:
        api_router = app.get_router("api")
        app_router = app.get_router("app")

        # TODO subselect
        @api_router.get(
            self.route,
            response_model=dict[str, dict[str, list[str]]],
            tags=["Utility"],
        )
        def channels_graph_data(request: Request) -> dict[str, dict[str, list[str]]]:
            """
            This endpoint returns the structure of the GatewayChannels graph as a JSON object.
            It is used by the `Browse Channels Graph` endpoint to generate a nice, interactive view of the graph.

            Data is of the form:

            ```
            {
                "<channel name>": {
                    "getters": [`GatewayModule`s that pull from that channel],
                    "setters": [`GatewayModule`s that push to that channel]
                },
                ...
            }
            ```
            """
            return request.app.gateway.channels.graph()

        @app_router.get("/channels_graph", response_class=HTMLResponse, tags=["Utility"])
        def browse_channels_graph(request: Request):
            """
            This endpoint is a small webpage that shows the dependency relationship of the GatewayChannels graph powering this API.
            """
            channels_graph = request.app.gateway.channels.graph()
            return app.templates.TemplateResponse(
                request,
                "channels_graph.html.j2",
                context={"channels_graph": dumps(channels_graph)},
            )

    def _graph(self) -> dict[str, Any]:
        """The channels graph as a spaday-dagre node/edge config.

        Channels and modules become nodes (distinguished by class for styling); a module that
        sets a channel gets a module->channel edge, a getter a channel->module edge.
        """
        data = self._channels.graph() if self._channels is not None else {}
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        for channel, wiring in data.items():
            nodes[channel] = {"id": channel, "class": "gateway-channel"}
            for setter in wiring.get("setters", []):
                nodes.setdefault(setter, {"id": setter, "class": "gateway-module"})
                edges.append({"source": setter, "target": channel})
            for getter in wiring.get("getters", []):
                nodes.setdefault(getter, {"id": getter, "class": "gateway-module"})
                edges.append({"source": channel, "target": getter})
        return {"nodes": list(nodes.values()), "edges": edges}

    def ui(self, app: "GatewayUI") -> None:
        # The graph opens as a closeable tab in the main window, rendered by spaday-dagre (the
        # legacy standalone page at `route` remains served for this release).
        from spaday_dagre import Dagre

        def graph_tab():
            return Dagre().prop("graph", self._graph()).prop("layout", {"rankdir": "LR", "ranksep": 60}).style(padding="0.5rem")

        app.add_tab(_GRAPH_TAB, "Channels Graph", graph_tab)
        from csp_gateway.server.web.spaday_ui import Region

        app.add(Region.DRAWER_RIGHT, app.tab_button("Channels Graph", _GRAPH_TAB))
