class RoutePreference:
    def __init__(self, config):
        self.preferred = config.get("preferred_backends", [])
        if not isinstance(self.preferred, list) or any(
            not isinstance(name, str) for name in self.preferred
        ):
            raise ValueError("preferred_backends must be a list of backend names")

    async def route(self, routing, context):
        order = {name: index for index, name in enumerate(self.preferred)}
        return [
            candidate.id
            for candidate in sorted(
                routing.candidates,
                key=lambda candidate: order.get(candidate.backend, len(order)),
            )
        ]


def create_plugin(config):
    return RoutePreference(config)
