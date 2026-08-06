"""Provider adapters.

Each module wraps one upstream SDK and exposes a class named in
``mm_gateway.registry._PROVIDER_CLASSES``. Providers are constructed lazily by
the registry only when their credentials are configured.
"""
