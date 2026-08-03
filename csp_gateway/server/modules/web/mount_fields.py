from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from csp_gateway.server import ChannelSelection, GatewayChannels, GatewayModule
from csp_gateway.server.web import GatewayWebApp, get_default_responses


class MountFieldRestRoutes(GatewayModule):
    """Mount rest routes for specific non-csp fields of the GatewayChannels.

    This is not done generically across all static fields as they may not always be serializable.
    """

    requires: ChannelSelection | None = []
    fields: list[str] = Field(description="Static fields on the Channels that should be exposed via REST. These must be JSON serializable.")
    route: str = "/field"

    def connect(self, channels: GatewayChannels) -> None:
        # NO-OP
        ...

    def rest(self, app: GatewayWebApp) -> None:
        # Get API Router
        api_router: APIRouter = app.get_router("api")

        for field in self.fields:
            model = app.gateway.channels_model.get_outer_type(field)
            add_field_routes(api_router, field, self.route, model)

        @api_router.get(
            f"{self.route}",
            responses=get_default_responses(),
            response_model=list[str],
            include_in_schema=False,
        )
        async def get_field(request: Request) -> list[str]:
            """
            This endpoint will return a list of string values of all available channels under the `/field` route.
            """
            return self.fields


def add_field_routes(
    api_router: APIRouter,
    field: str,
    route: str,
    model: BaseModel | type,
) -> None:
    async def get_field(request: Request) -> model:  # type: ignore[misc, valid-type]
        """
        Get static field value on a static channel.
        """
        # Throw 404 if not a supported channel
        if not hasattr(request.app.gateway.channels, field):
            raise HTTPException(status_code=404, detail=f"Channel field not found: {field}")

        # Grab the request off the edge
        try:
            res = getattr(request.app.gateway.channels, field)
        except AttributeError:
            raise HTTPException(
                status_code=404,
                detail=f"Channel field not found: {field}",
            )

        return res

    api_router.get(
        f"{route}/{field}",
        responses=get_default_responses(),
        response_model=model,
        name=f"Get Channel field {field}",
    )(get_field)

    api_router.get(
        "/{}".format(field.replace("_", "-")),
        responses=get_default_responses(),
        response_model=model,
        include_in_schema=False,
    )(get_field)
