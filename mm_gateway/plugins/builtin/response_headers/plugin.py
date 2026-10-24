class ResponseHeaders:
    def __init__(self, config):
        self.headers = config.get("headers", {})
        if not isinstance(self.headers, dict):
            raise TypeError("headers must be an object")

    async def on_response(self, response, context):
        for name, value in self.headers.items():
            response.headers[name] = str(value)


def create_plugin(config):
    return ResponseHeaders(config)
