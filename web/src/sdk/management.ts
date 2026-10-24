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
export type ModelCatalog = Schema["BackendModelCatalog"];
export type ModelDiscovery = Schema["BackendModelDiscovery"];
export type PluginStoreData = Schema["PluginStore"];
export type PluginEntry = Schema["PluginEntry"];
export type PluginOptions = Schema["PluginOptions"];
export type GitHubPluginInstall = Schema["GitHubPluginInstall"];
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
    timeout?: number,
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
      ...(timeout ? { timeout } : {}),
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
  getManagementBuiltinModels(providerType: string) {
    return this.json<ModelCatalog>(
      "GET",
      `/backend-types/${encodeURIComponent(providerType)}/models`,
    );
  }
  discoverManagementBackendModels(body: Backend) {
    return this.json<ModelDiscovery>("POST", "/backends/discover-models", body);
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
  getManagementPluginStore() {
    return this.json<PluginStoreData>("GET", "/plugins");
  }
  putManagementPlugin(id: string, options: PluginOptions, revision: string) {
    return this.json<PluginStoreData>(
      "PUT",
      `/plugins/${encodeURIComponent(id)}`,
      options,
      revision,
    );
  }
  deleteManagementPlugin(id: string, revision: string) {
    return this.json<PluginStoreData>(
      "DELETE",
      `/plugins/${encodeURIComponent(id)}`,
      undefined,
      revision,
    );
  }
  reloadManagementPlugin(id: string, revision: string) {
    return this.json<PluginStoreData>(
      "POST",
      `/plugins/${encodeURIComponent(id)}/reload`,
      undefined,
      revision,
    );
  }
  installManagementGitHubPlugin(body: GitHubPluginInstall, revision: string) {
    // Fetching a repository can take longer than ordinary configuration edits.
    return this.json<PluginStoreData>(
      "POST",
      "/plugins/install",
      body,
      revision,
      undefined,
      180000,
    );
  }
  updateManagementGitHubPlugin(id: string, revision: string) {
    return this.json<PluginStoreData>(
      "POST",
      `/plugins/${encodeURIComponent(id)}/update`,
      undefined,
      revision,
      undefined,
      180000,
    );
  }
}
