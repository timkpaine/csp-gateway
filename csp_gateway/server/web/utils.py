from collections.abc import Callable, Coroutine
from typing import Any

from fastapi.exceptions import RequestErrorModel

__all__ = (
    "Error404",
    "get_default_responses",
)

NoArgsNoReturnFuncT = Callable[[], None]
NoArgsNoReturnAsyncFuncT = Callable[[], Coroutine[Any, Any, None]]
NoArgsNoReturnDecorator = Callable[[NoArgsNoReturnFuncT | NoArgsNoReturnAsyncFuncT], NoArgsNoReturnAsyncFuncT]


class Error404(RequestErrorModel):  # type: ignore[misc, valid-type]
    detail: str = ""


def get_default_responses() -> dict[int | str, dict[str, Any]]:
    return {404: {"model": Error404}}
