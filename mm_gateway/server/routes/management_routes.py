"""Authenticated operator APIs. Configuration writes require a current revision."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, Response

from mm_gateway.schemas.management import (
    ManagedBackend, ManagedKey, ManagedProxy, ManagementConfig, ManagementConfigResponse,
    ManagementMetrics, ManagementStatus, ManagementTaskList, ManagementUsageList, Modality, TaskStatus,
)
from mm_gateway.server.auth import require_management_key

router = APIRouter(prefix="/v1/management", tags=["management"],
                   dependencies=[Depends(require_management_key)])
IfMatch = Annotated[str | None, Header(alias="If-Match", description="Current configuration revision (ETag).")]
WRITE_RESPONSES = {412: {"description": "Configuration changed; reload before saving."},
                   428: {"description": "A current If-Match revision is required."}}


def config_response(request: Request, response: Response, body: dict) -> dict:
    response.headers["ETag"] = f'"{body["revision"]}"'
    return body


@router.get("/status", operation_id="getManagementStatus", response_model=ManagementStatus)
async def get_status(request: Request):
    return await request.app.state.management.status()


@router.get("/config", operation_id="getManagementConfig", response_model=ManagementConfigResponse)
async def get_config(request: Request, response: Response):
    return config_response(request, response, request.app.state.management.configuration())


@router.put("/config", operation_id="replaceManagementConfig", response_model=ManagementConfigResponse,
            responses=WRITE_RESPONSES)
async def replace_config(request: Request, response: Response, body: ManagementConfig, if_match: IfMatch = None):
    result = await request.app.state.management.update(body, if_match)
    return config_response(request, response, result)


@router.put("/backends/{name}", operation_id="putManagementBackend", response_model=ManagementConfigResponse,
            responses=WRITE_RESPONSES)
async def put_backend(request: Request, response: Response, name: str, body: ManagedBackend, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("backends", name, body, if_match)
    return config_response(request, response, result)


@router.delete("/backends/{name}", operation_id="deleteManagementBackend", response_model=ManagementConfigResponse,
               responses=WRITE_RESPONSES)
async def delete_backend(request: Request, response: Response, name: str, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("backends", name, None, if_match)
    return config_response(request, response, result)


@router.put("/keys/{key_id}", operation_id="putManagementKey", response_model=ManagementConfigResponse,
            responses=WRITE_RESPONSES)
async def put_key(request: Request, response: Response, key_id: str, body: ManagedKey, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("keys", key_id, body, if_match)
    return config_response(request, response, result)


@router.delete("/keys/{key_id}", operation_id="deleteManagementKey", response_model=ManagementConfigResponse,
               responses=WRITE_RESPONSES)
async def delete_key(request: Request, response: Response, key_id: str, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("keys", key_id, None, if_match)
    return config_response(request, response, result)


@router.put("/proxies/{domain}", operation_id="putManagementProxy", response_model=ManagementConfigResponse,
            responses=WRITE_RESPONSES)
async def put_proxy(request: Request, response: Response, domain: str, body: ManagedProxy, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("proxies", domain, body, if_match)
    return config_response(request, response, result)


@router.delete("/proxies/{domain}", operation_id="deleteManagementProxy", response_model=ManagementConfigResponse,
               responses=WRITE_RESPONSES)
async def delete_proxy(request: Request, response: Response, domain: str, if_match: IfMatch = None):
    result = await request.app.state.management.update_resource("proxies", domain, None, if_match)
    return config_response(request, response, result)


@router.get("/metrics", operation_id="getManagementMetrics", response_model=ManagementMetrics)
async def get_metrics(request: Request):
    return request.app.state.management.metrics()


@router.get("/tasks", operation_id="listManagementTasks", response_model=ManagementTaskList)
async def list_tasks(request: Request, modality: Modality | None = None, status: TaskStatus | None = None,
                     key_id: str | None = None, backend: str | None = None,
                     offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 50):
    tasks = await request.app.state.management.tasks()
    filters = {"modality": modality, "status": status, "owner_key_id": key_id, "backend": backend}
    tasks = [task for task in tasks if all(value is None or task[name] == value for name, value in filters.items())]
    return {"object": "list", "data": tasks[offset:offset + limit], "total": len(tasks), "offset": offset, "limit": limit}


@router.get("/usage", operation_id="listManagementUsage", response_model=ManagementUsageList)
async def list_usage(request: Request, key_id: str | None = None):
    return request.app.state.management.usage(key_id)
