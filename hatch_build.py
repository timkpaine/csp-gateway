import os
from hatchling.metadata.plugin.interface import MetadataHookInterface

__all__ = ("CustomHook",)

class CustomHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        # TODO: make CLI after https://github.com/pypa/hatch/pull/1743
        extra = os.environ.get("CSP_GATEWAY_EXTRA", "")

        if extra and extra in metadata["optional-dependencies"]:
            metadata["name"] = f"csp-gateway-{extra}"
            metadata["dependencies"] = metadata["optional-dependencies"].pop(extra)
        else:
            metadata["name"] = "csp-gateway"
            metadata["dependencies"] = metadata["optional-dependencies"].get("server")
            