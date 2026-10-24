import { useCallback, useEffect, useState, type FormEvent } from "react";
import axios from "axios";
import {
  Download,
  Puzzle,
  RefreshCw,
  Search,
  Settings2,
  Trash2,
  X,
} from "lucide-react";
import {
  ManagementApi,
  type PluginEntry,
  type PluginOptions,
  type PluginStoreData,
} from "./sdk/management";

function message(cause: unknown) {
  if (axios.isAxiosError(cause))
    return cause.response?.data?.detail || cause.message;
  return cause instanceof Error ? cause.message : "Plugin operation failed.";
}

const defaults: PluginOptions = {
  enabled: true,
  order: 0,
  config: {},
  timeout_ms: 1000,
  failure_mode: "closed",
};

export function PluginStore({
  api,
  updatedAt,
}: {
  api: ManagementApi;
  updatedAt: Date | null;
}) {
  const [store, setStore] = useState<PluginStoreData | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState("");
  const [installedOnly, setInstalledOnly] = useState(false);
  const [editing, setEditing] = useState<PluginEntry | "github" | null>(null);
  const [editingRevision, setEditingRevision] = useState("");
  const [options, setOptions] = useState<PluginOptions>(defaults);
  const [config, setConfig] = useState("{}");
  const [repository, setRepository] = useState("");
  const [ref, setRef] = useState("HEAD");
  const [subdirectory, setSubdirectory] = useState("");

  const load = useCallback(async () => {
    try {
      setStore((await api.getManagementPluginStore()).data);
    } catch (cause) {
      setError(message(cause));
    }
  }, [api]);

  useEffect(() => {
    if (busy || editing) return;
    let alive = true;
    api.getManagementPluginStore().then(
      (response) => {
        if (alive) setStore(response.data);
      },
      (cause) => {
        if (alive) setError(message(cause));
      },
    );
    return () => {
      alive = false;
    };
  }, [api, updatedAt, busy, editing]);

  async function mutate(
    operation: () => Promise<{ data: PluginStoreData }>,
    success: string,
  ) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      setStore((await operation()).data);
      setEditing(null);
      setNotice(success);
    } catch (cause) {
      setError(message(cause));
      await load();
    } finally {
      setBusy(false);
    }
  }

  function open(entry: PluginEntry | "github") {
    const next = entry === "github" ? defaults : entry.options || defaults;
    setOptions({ ...next });
    setConfig(JSON.stringify(next.config || {}, null, 2));
    setRepository("");
    setRef("HEAD");
    setSubdirectory("");
    setError("");
    setEditingRevision(store?.revision || "");
    setEditing(entry);
  }

  function save(event: FormEvent) {
    event.preventDefault();
    if (!editing || !store) return;
    try {
      const parsed = JSON.parse(config);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
        throw new Error("Plugin configuration must be a JSON object.");
      const next = { ...options, config: parsed };
      void mutate(
        () =>
          editing === "github"
            ? api.installManagementGitHubPlugin(
                { repository, ref, subdirectory, options: next },
                editingRevision,
              )
            : api.putManagementPlugin(editing.id, next, editingRevision),
        "Plugin settings applied.",
      );
    } catch (cause) {
      setError(message(cause));
    }
  }

  const entries = (store?.data || []).filter(
    (entry) =>
      (!installedOnly || entry.installed) &&
      `${entry.id} ${entry.manifest?.name} ${entry.manifest?.description}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );

  return (
    <>
      {error && !editing && (
        <div className="alert error" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="toast" role="status">
          {notice}
        </div>
      )}
      <div className="section-toolbar">
        <div className="search">
          <Search size={16} />
          <input
            aria-label="Search plugins"
            placeholder="Search plugins…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="button-row">
          <label className="check-field">
            <input
              type="checkbox"
              checked={installedOnly}
              onChange={(e) => setInstalledOnly(e.target.checked)}
            />{" "}
            Installed only
          </label>
          <button
            className="secondary"
            disabled={busy}
            onClick={() => void load()}
          >
            <RefreshCw size={15} /> Refresh store
          </button>
          <button
            className="primary"
            disabled={busy || !store}
            onClick={() => open("github")}
          >
            <Download size={16} /> Install from GitHub
          </button>
        </div>
      </div>
      {store && (
        <div className="plugin-store-note">
          <span>
            {store.data.filter((entry) => entry.active).length} active ·{" "}
            {store.watching ? "Hot reload on" : "Manual reload"} ·{" "}
            {store.persistent
              ? "Installs saved"
              : "Settings last until restart"}
          </span>
          <details>
            <summary>Develop a plugin</summary>
            <p>
              Create a package with{" "}
              <code>
                mm-gateway-plugin new my-plugin --directory {store.directory}
              </code>
              , then run{" "}
              <code>mm-gateway-plugin check {store.directory}/my-plugin</code>.
              Install it below and edit its files to hot reload.
            </p>
            <p>
              Local source directory: <code>{store.directory}</code>. Publish
              the package to a public GitHub repository to share it.
            </p>
          </details>
        </div>
      )}
      {!store && <div className="empty">Loading Plugin Store…</div>}
      <div className="resource-grid plugin-grid">
        {entries.map((entry) => (
          <article className="resource-card plugin-card" key={entry.id}>
            <div className="resource-top">
              <span className="resource-icon">
                <Puzzle size={20} />
              </span>
              <span
                className={`badge ${entry.active ? "positive" : entry.error ? "negative" : ""}`}
              >
                <span />
                {entry.active
                  ? "active"
                  : entry.installed
                    ? entry.options?.enabled
                      ? "unavailable"
                      : "disabled"
                    : "available"}
              </span>
            </div>
            <h3>{entry.manifest?.name || entry.id}</h3>
            <p className="resource-description">
              {entry.manifest?.description ||
                "Package unavailable. Restore its source or uninstall it."}
            </p>
            <div className="plugin-facts">
              <span>
                {entry.id} · {entry.manifest?.version || "unknown version"} ·{" "}
                {entry.source.kind}
              </span>
              {entry.manifest?.author && (
                <span>By {entry.manifest.author}</span>
              )}
              <span>
                {entry.manifest?.hooks
                  ?.map((hook) => hook.replace("on_", "").replaceAll("_", " "))
                  .join(" · ")}
              </span>
              {!!entry.manifest?.providers?.length && (
                <span>
                  Provider types: {entry.manifest.providers.join(", ")}
                </span>
              )}
              {entry.installed && (
                <span>
                  {entry.calls} hook calls · {entry.failures} failures · order{" "}
                  {entry.options?.order}
                </span>
              )}
              {entry.source.commit && (
                <span>
                  Commit <code>{entry.source.commit.slice(0, 12)}</code> ·{" "}
                  {entry.source.ref}
                </span>
              )}
              {entry.source.repository && (
                <a
                  href={entry.source.repository}
                  target="_blank"
                  rel="noreferrer"
                >
                  View repository ↗
                </a>
              )}
            </div>
            {entry.error && (
              <p className="field-error" role="status">
                {entry.error}
                {entry.active &&
                  " The last working generation is still active."}
              </p>
            )}
            <div className="plugin-actions">
              {!entry.installed ? (
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() => open(entry)}
                >
                  <Download size={14} /> Install
                </button>
              ) : (
                <>
                  <button
                    className="secondary"
                    disabled={busy}
                    onClick={() => open(entry)}
                  >
                    <Settings2 size={14} /> Configure
                  </button>
                  <button
                    className="secondary"
                    disabled={busy || !store}
                    onClick={() =>
                      void mutate(
                        () =>
                          api.putManagementPlugin(
                            entry.id,
                            {
                              ...defaults,
                              ...entry.options,
                              enabled: !entry.options?.enabled,
                            },
                            store!.revision,
                          ),
                        "Plugin state applied.",
                      )
                    }
                  >
                    {entry.options?.enabled ? "Disable" : "Enable"}
                  </button>
                  <button
                    className="secondary"
                    disabled={busy || !entry.options?.enabled}
                    onClick={() =>
                      void mutate(
                        () =>
                          api.reloadManagementPlugin(entry.id, store!.revision),
                        "Plugin reloaded.",
                      )
                    }
                  >
                    <RefreshCw size={14} /> Reload
                  </button>
                  {entry.source.kind === "github" && (
                    <button
                      className="secondary"
                      disabled={busy}
                      onClick={() =>
                        void mutate(
                          () =>
                            api.updateManagementGitHubPlugin(
                              entry.id,
                              store!.revision,
                            ),
                          "GitHub plugin updated.",
                        )
                      }
                    >
                      <Download size={14} /> Update
                    </button>
                  )}
                  <button
                    className="icon-button danger"
                    aria-label={`Uninstall ${entry.id}`}
                    disabled={busy}
                    onClick={() =>
                      void mutate(
                        () =>
                          api.deleteManagementPlugin(entry.id, store!.revision),
                        "Plugin uninstalled.",
                      )
                    }
                  >
                    <Trash2 size={16} />
                  </button>
                </>
              )}
            </div>
          </article>
        ))}
      </div>
      {store && !entries.length && (
        <div className="empty">
          No matching plugins. Add a local package or install from GitHub.
        </div>
      )}
      {store?.errors.map((item) => (
        <div className="alert error" key={item.path}>
          <span>
            <code>{item.path}</code>: {item.error}
          </span>
        </div>
      ))}
      {editing && (
        <div className="modal-overlay">
          <div
            className="editor-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="plugin-editor-title"
          >
            <form onSubmit={save}>
              <div className="panel-heading">
                <h2 id="plugin-editor-title">
                  {editing === "github"
                    ? "Install from GitHub"
                    : `${editing.installed ? "Configure" : "Install"} ${editing.manifest?.name || editing.id}`}
                </h2>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Close plugin editor"
                  disabled={busy}
                  onClick={() => setEditing(null)}
                >
                  <X size={20} />
                </button>
              </div>
              <div className="editor-body">
                {error && (
                  <>
                    <div className="alert error" role="alert">
                      {error}
                    </div>
                    <button
                      type="button"
                      className="secondary"
                      disabled={busy}
                      onClick={() => {
                        if (!store) return;
                        if (editing !== "github")
                          open(
                            store.data.find(
                              (entry) => entry.id === editing.id,
                            ) || editing,
                          );
                        else {
                          setEditingRevision(store.revision);
                          setError("");
                        }
                      }}
                    >
                      Reload saved settings
                    </button>
                  </>
                )}
                {editing === "github" && (
                  <>
                    <p className="editor-note">
                      Plugins run Python code with gateway permissions. Install
                      repositories you trust. Dependencies must already be
                      installed in the gateway environment.
                    </p>
                    <label className="field">
                      GitHub repository URL
                      <input
                        type="url"
                        required
                        placeholder="https://github.com/owner/gateway-plugins"
                        value={repository}
                        onChange={(e) => setRepository(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      Branch, tag or commit
                      <input
                        required
                        value={ref}
                        onChange={(e) => setRef(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      Plugin subdirectory
                      <input
                        placeholder="Leave empty for repository root"
                        value={subdirectory}
                        onChange={(e) => setSubdirectory(e.target.value)}
                      />
                    </label>
                  </>
                )}
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={options.enabled}
                    onChange={(e) =>
                      setOptions({ ...options, enabled: e.target.checked })
                    }
                  />{" "}
                  Enabled
                </label>
                <label className="field">
                  Plugin configuration (JSON)
                  <textarea
                    className="code-editor"
                    rows={7}
                    value={config}
                    spellCheck={false}
                    onChange={(e) => setConfig(e.target.value)}
                  />
                </label>
                <p className="field-hint">
                  Package defaults apply to omitted fields. Keep existing
                  [redacted] values to preserve secrets.
                </p>
                <label className="field">
                  Execution order
                  <input
                    type="number"
                    min={-10000}
                    max={10000}
                    required
                    value={options.order}
                    onChange={(e) =>
                      setOptions({ ...options, order: Number(e.target.value) })
                    }
                  />
                </label>
                <p className="field-hint">
                  Request hooks run from lowest to highest order; response hooks
                  run in reverse.
                </p>
                <label className="field">
                  Hook timeout (ms)
                  <input
                    type="number"
                    min={10}
                    max={30000}
                    required
                    value={options.timeout_ms}
                    onChange={(e) =>
                      setOptions({
                        ...options,
                        timeout_ms: Number(e.target.value),
                      })
                    }
                  />
                </label>
                <label className="field">
                  On hook failure
                  <select
                    value={options.failure_mode}
                    onChange={(e) =>
                      setOptions({
                        ...options,
                        failure_mode: e.target.value as "closed" | "open",
                      })
                    }
                  >
                    <option value="closed">Stop request</option>
                    <option value="open">
                      Continue without this hook's changes
                    </option>
                  </select>
                </label>
              </div>
              <div className="dialog-footer">
                <button
                  type="button"
                  className="secondary"
                  disabled={busy}
                  onClick={() => setEditing(null)}
                >
                  Cancel
                </button>
                <button type="submit" className="primary" disabled={busy}>
                  {busy
                    ? "Applying…"
                    : editing === "github" || !editing.installed
                      ? "Install plugin"
                      : "Save plugin"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
