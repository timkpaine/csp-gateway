import typing
from datetime import datetime, timezone

from atomic_counter import Counter

if typing.TYPE_CHECKING:
    from csp_gateway.server import GatewayModule


def get_counter(kind: "GatewayModule"):
    if not hasattr(get_counter, "id_map"):
        get_counter.map = {}
    if kind not in get_counter.map:
        nowish = datetime.now(timezone.utc)
        base = datetime(nowish.year, nowish.month, nowish.day)
        get_counter.map[kind] = Counter(int(base.timestamp()) * 1_000_000_000)
    return get_counter.map[kind]
