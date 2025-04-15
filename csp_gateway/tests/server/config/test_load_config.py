from csp_gateway import Gateway, load_config


class TestLoadConfig:
    def test_load_config(self):
        gw = load_config(overrides=["+gateway=demo", "+port=1234"], overwrite=True)["/gateway"]
        assert isinstance(gw, Gateway)
        assert gw.settings.PORT == 1234
