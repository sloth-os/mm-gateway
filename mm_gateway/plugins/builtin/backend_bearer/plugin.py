from dataclasses import replace


class BackendBearer:
    def __init__(self, config):
        self.token = config.get("token")
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("Configure a non-empty token.")

    async def prepare_backend(self, backend, context):
        return replace(backend, api_key=self.token)

    async def on_backend_request(self, request, context):
        request.headers["authorization"] = "Bearer " + self.token


def create_plugin(config):
    return BackendBearer(config)
