"""Tests for `MountOutputsFolder`'s listing and chunk endpoints (the spaday log viewer's backend)."""

import os
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
    MountOutputsFolder,
)


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


@pytest.fixture(scope="class")
def outputs_dir(tmp_path_factory):
    """An outputs tree, plus a sibling directory sharing its name as a prefix."""
    root = tmp_path_factory.mktemp("logviewer")
    outputs = root / "outputs"
    (outputs / "run" / "nested").mkdir(parents=True)
    (outputs / "run" / "app.log").write_text("".join(f"line {i}\n" for i in range(1000)))
    (outputs / "run" / "nested" / "config.yaml").write_text("a: 1\n")
    # `<dir>-evil` shares `<dir>` as a string prefix; a startswith() containment check lets it through.
    sibling = root / "outputs-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("do not serve me")
    return outputs


@pytest.fixture(scope="class")
def client(outputs_dir, free_port):
    gateway = Gateway(
        modules=[ExampleModule(), MountOutputsFolder(dir=str(outputs_dir), chunk_bytes=256)],
        channels=ExampleChannels(),
        settings=GatewaySettings(PORT=free_port),
    )
    gateway.start(rest=True, _in_test=True)
    try:
        yield TestClient(gateway.web_app.get_fastapi())
    finally:
        gateway.stop()


class TestOutputsApi:
    def test_lists_relative_paths(self, client):
        body = client.get("/outputs/_tree").json()
        assert body["paths"] == ["run/app.log", "run/nested/config.yaml"]
        assert body["truncated"] is False

    def test_caps_entries(self, client):
        body = client.get("/outputs/_tree?limit=1").json()
        assert body["paths"] == ["run/app.log"]
        assert body["truncated"] is True

    def test_tails_by_default(self, client, outputs_dir):
        size = (outputs_dir / "run" / "app.log").stat().st_size
        body = client.get("/outputs/_chunk?path=run/app.log").json()
        assert body["size"] == size
        assert body["end"] == size
        assert body["start"] == size - 256
        assert body["text"].endswith("line 999\n")
        assert len(body["text"].encode()) == 256

    def test_reads_an_explicit_range(self, client):
        body = client.get("/outputs/_chunk?path=run/app.log&start=0&end=7").json()
        assert body["text"] == "line 0\n"
        assert (body["start"], body["end"]) == (0, 7)

    def test_small_file_is_served_whole(self, client):
        body = client.get("/outputs/_chunk?path=run/nested/config.yaml").json()
        assert body["text"] == "a: 1\n"
        assert body["start"] == 0

    def test_rejects_escaping_paths(self, client):
        for path in ("../outputs-evil/secret.txt", "/etc/passwd", "run/../../outputs-evil/secret.txt"):
            assert client.get(f"/outputs/_chunk?path={path}").status_code == 404

    def test_missing_file(self, client):
        assert client.get("/outputs/_chunk?path=run/nope.log").status_code == 404

    def test_directory_is_not_a_chunk(self, client):
        assert client.get("/outputs/_chunk?path=run").status_code == 404

    def test_legacy_browser_still_serves(self, client):
        """The HTML listing the default UI provider links to is untouched."""
        assert client.get("/outputs").status_code == 200
        assert client.get("/outputs/run/app.log").status_code == 200


class TestSpadayViewer:
    """The spaday provider gets the in-page viewer instead of the link that navigated away."""

    @pytest.fixture(scope="class")
    def client(self, outputs_dir, free_port):
        pytest.importorskip("spaday")
        gateway = Gateway(
            modules=[ExampleModule(), MountOutputsFolder(dir=str(outputs_dir))],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port, UI_PROVIDER="spaday"),
        )
        gateway.start(rest=True, ui=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_page_carries_the_tree_and_reader(self, client):
        tree = client.get("/tree.json").text
        assert "spaday-tree" in tree
        assert "/outputs/_tree" in tree
        assert "/outputs/_chunk" in tree

    def test_panel_shows_the_serving_pid(self, client):
        """The HTML log page has always shown it; the tab must not lose it."""
        assert f"pid[{os.getpid()}]" in client.get("/tree.json").text

    def test_logs_open_in_a_tab_not_a_link(self, client):
        tree = client.get("/tree.json").text
        # The drawer button opens the registered tab; nothing should link out to the HTML browser.
        assert '"logs"' in tree
        assert '"href": "/outputs"' not in tree


class TestChunkWindowing:
    """Walking backwards must stay bounded: the Older gesture sends only an `end`."""

    @pytest.fixture(scope="class")
    def client(self, outputs_dir, free_port):
        gateway = Gateway(
            modules=[ExampleModule(), MountOutputsFolder(dir=str(outputs_dir), chunk_bytes=256)],
            channels=ExampleChannels(),
            settings=GatewaySettings(PORT=free_port),
        )
        gateway.start(rest=True, _in_test=True)
        try:
            yield TestClient(gateway.web_app.get_fastapi())
        finally:
            gateway.stop()

    def test_end_only_reads_one_chunk_not_everything_before(self, client):
        body = client.get("/outputs/_chunk?path=run/app.log&end=5000").json()
        assert (body["start"], body["end"]) == (4744, 5000)
        assert len(body["text"].encode()) == 256

    def test_end_only_clamps_at_the_start_of_file(self, client):
        body = client.get("/outputs/_chunk?path=run/app.log&end=100").json()
        assert body["start"] == 0
        assert len(body["text"].encode()) == 100

    def test_posting_the_trees_selection_shape_opens_the_tail(self, client, outputs_dir):
        size = (outputs_dir / "run" / "app.log").stat().st_size
        body = client.post("/outputs/_chunk", json={"paths": ["run/app.log"]}).json()
        assert body["path"] == "run/app.log"
        assert body["end"] == size

    def test_posting_an_empty_selection_is_not_an_error(self, client):
        assert client.post("/outputs/_chunk", json={"paths": []}).json()["text"] == ""

    def test_posting_a_directory_selection_is_not_an_error(self, client):
        """Expanding a directory in the tree emits a selection; it must not read as a failure."""
        body = client.post("/outputs/_chunk", json={"paths": ["run"]})
        assert body.status_code == 200
        assert body.json()["text"] == ""

    def test_posting_an_escaping_selection_is_still_rejected(self, client):
        assert client.post("/outputs/_chunk", json={"paths": ["../outputs-evil/secret.txt"]}).status_code == 404
