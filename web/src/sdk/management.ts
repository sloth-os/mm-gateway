/** Management bindings extending the pinned sloth-os/mm-gateway-ts SDK.
 * Schema types are generated from the gateway's OpenAPI via npm run sdk:generate.
 * All calls share the upstream SDK's Configuration, Axios client and bearer token.
 */
import { MetaApi } from "@sloth-os/mm-gateway-ts";
import type { components } from "./schema";

type Schema = components["schemas"];
export type Config = Schema["ManagementConfig-Input"];
export type ConfigResponse = Schema["ManagementConfigResponse"];
export type Status = Schema["ManagementStatus"];
export type Metrics = Schema["ManagementMetrics"];
export type TaskList = Schema["ManagementTaskList"];
export type UsageList = Schema["ManagementUsageList"];
export type Backend = Schema["ManagedBackend"];
export type Key = Schema["ManagedKey"];
export type Proxy = Schema["ManagedProxy"];
export type TaskFilters = {
  status?: string;
  modality?: string;
  key_id?: string;
  backend?: string;
  offset?: number;
  limit?: number;
};

const ROOT = "/v1/management";

export class ManagementApi extends MetaApi {
  private async json<T>(
    method: string,
    path: string,
    data?: unknown,
    revision?: string,
    params?: object,
  ) {
    const options = this.configuration?.baseOptions || {};
    const configured = this.configuration?.accessToken;
    const token = await (typeof configured === "function"
      ? configured()
      : configured);
    return this.axios.request<T>({
      ...options,
      method,
      url: this.basePath + ROOT + path,
      data,
      params,
      headers: {
        ...options.headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(data ? { "Content-Type": "application/json" } : {}),
        ...(revision ? { "If-Match": `"${revision}"` } : {}),
      },
    });
  }

  getManagementStatus() {
    return this.json<Status>("GET", "/status");
  }
  getManagementConfig() {
    return this.json<ConfigResponse>("GET", "/config");
  }
  replaceManagementConfig(config: Config, revision: string) {
    return this.json<ConfigResponse>("PUT", "/config", config, revision);
  }
  getManagementMetrics() {
    return this.json<Metrics>("GET", "/metrics");
  }
  listManagementTasks(filters: TaskFilters = {}) {
    return this.json<TaskList>("GET", "/tasks", undefined, undefined, filters);
  }
  listManagementUsage(keyId?: string) {
    return this.json<UsageList>("GET", "/usage", undefined, undefined, {
      key_id: keyId,
    });
  }
  putManagementBackend(body: Backend, revision: string) {
    return this.json<ConfigResponse>(
      "PUT",
      `/backends/${encodeURIComponent(body.name)}`,
      body,
      revision,
    );
  }
  deleteManagementBackend(name: string, revision: string) {
    return this.json<ConfigResponse>(
      "DELETE",
      `/backends/${encodeURIComponent(name)}`,
      undefined,
      revision,
    );
  }
  putManagementKey(body: Key, revision: string) {
    return this.json<ConfigResponse>(
      "PUT",
      `/keys/${encodeURIComponent(body.id)}`,
      body,
      revision,
    );
  }
  deleteManagementKey(id: string, revision: string) {
    return this.json<ConfigResponse>(
      "DELETE",
      `/keys/${encodeURIComponent(id)}`,
      undefined,
      revision,
    );
  }
  putManagementProxy(body: Proxy, revision: string) {
    return this.json<ConfigResponse>(
      "PUT",
      `/proxies/${encodeURIComponent(body.domain || new URL(body.base_url).hostname)}`,
      body,
      revision,
    );
  }
  deleteManagementProxy(domain: string, revision: string) {
    return this.json<ConfigResponse>(
      "DELETE",
      `/proxies/${encodeURIComponent(domain)}`,
      undefined,
      revision,
    );
  }
}
