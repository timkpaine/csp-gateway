import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ccflow import RootModelRegistry, load_config as load_config_base

if TYPE_CHECKING:
    from csp_gateway import Gateway

log = logging.getLogger(__name__)

__all__ = (
    "load_config",
    "load_gateway",
)


def load_config(
    config_dir: str = "",
    config_name: str = "",
    overrides: Optional[List[str]] = None,
    *,
    overwrite: bool = True,
    basepath: str = "",
) -> RootModelRegistry:
    log.info("Loading csp-gateway config...")
    return load_config_base(
        root_config_dir=str(Path(__file__).resolve().parent),
        root_config_name="base",
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
        overwrite=overwrite,
        basepath=basepath,
    )


@wraps(load_config)
def load_gateway(*args, **kwargs) -> "Gateway":
    log.info("Loading csp-gateway config...")
    return load_config(*args, **kwargs)["gateway"]
