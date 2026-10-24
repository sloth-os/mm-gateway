"""Admin-only plugin store, local activation and GitHub installation."""

from fastapi import APIRouter, Depends, Request, Response

from mm_gateway.schemas.plugins import GitHubPluginInstall, PluginOptions, PluginStore
from mm_gateway.server.auth import require_management_key
from mm_gateway.server.routes.management_routes import WRITE_RESPONSES, IfMatch

router = APIRouter(
    prefix="/v1/management/plugins",
    tags=["management"],
    dependencies=[Depends(require_management_key)],
)


def store_response(response: Response, body: dict):
    response.headers["ETag"] = f'"{body["revision"]}"'
    return body


@router.get("", operation_id="getManagementPluginStore", response_model=PluginStore)
async def get_store(request: Request, response: Response):
    """Discover bundled, local and installed GitHub plugins without executing them."""
    return store_response(response, await request.app.state.plugins.snapshot())


@router.post(
    "/install",
    operation_id="installManagementGitHubPlugin",
    response_model=PluginStore,
    responses=WRITE_RESPONSES,
)
async def install_github(
    request: Request,
    response: Response,
    body: GitHubPluginInstall,
    if_match: IfMatch = None,
):
    """Fetch a public GitHub package, validate it, persist and atomically activate it."""
    return store_response(
        response, await request.app.state.plugins.install_github(body, if_match)
    )


@router.put(
    "/{plugin_id}",
    operation_id="putManagementPlugin",
    response_model=PluginStore,
    responses=WRITE_RESPONSES,
)
async def put_plugin(
    request: Request,
    response: Response,
    plugin_id: str,
    body: PluginOptions,
    if_match: IfMatch = None,
):
    """Install a catalog plugin or change its enabled state, config, order or hook policy."""
    return store_response(
        response, await request.app.state.plugins.put(plugin_id, body, if_match)
    )


@router.delete(
    "/{plugin_id}",
    operation_id="deleteManagementPlugin",
    response_model=PluginStore,
    responses=WRITE_RESPONSES,
)
async def delete_plugin(
    request: Request, response: Response, plugin_id: str, if_match: IfMatch = None
):
    """Uninstall, releasing old generations after their in-flight requests complete."""
    return store_response(
        response, await request.app.state.plugins.delete(plugin_id, if_match)
    )


@router.post(
    "/{plugin_id}/reload",
    operation_id="reloadManagementPlugin",
    response_model=PluginStore,
    responses=WRITE_RESPONSES,
)
async def reload_plugin(
    request: Request, response: Response, plugin_id: str, if_match: IfMatch = None
):
    """Reload from disk. A failed reload leaves the last working generation active."""
    return store_response(
        response, await request.app.state.plugins.reload(plugin_id, if_match)
    )


@router.post(
    "/{plugin_id}/update",
    operation_id="updateManagementGitHubPlugin",
    response_model=PluginStore,
    responses=WRITE_RESPONSES,
)
async def update_plugin(
    request: Request, response: Response, plugin_id: str, if_match: IfMatch = None
):
    """Re-fetch the installed GitHub ref and activate it only after validation."""
    return store_response(
        response, await request.app.state.plugins.update_github(plugin_id, if_match)
    )
