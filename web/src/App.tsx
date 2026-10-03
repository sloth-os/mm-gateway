import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import axios from "axios";
import { Configuration } from "@sloth-os/mm-gateway-ts";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Clock3,
  Coins,
  Copy,
  Database,
  ExternalLink,
  KeyRound,
  Layers3,
  Loader2,
  LogOut,
  Network,
  Plus,
  RefreshCw,
  Search,
  Server,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import {
  ManagementApi,
  type Backend,
  type Config,
  type ConfigResponse,
  type Key,
  type Metrics,
  type Proxy,
  type Status,
  type TaskFilters,
  type TaskList,
  type UsageList,
} from "./sdk/management";

type Page =
  | "Overview"
  | "Metrics"
  | "Tasks"
  | "Usage"
  | "Backends"
  | "API keys"
  | "Proxies"
  | "Routing";
type Section = "backends" | "keys" | "proxies";
type Resource = Backend | Key | Proxy;
type Snapshot = {
  status: Status;
  metrics: Metrics;
  usage: UsageList;
  tasks: TaskList;
};
type Editor = {
  section: Section;
  original: string | null;
  draft: Record<string, unknown>;
};
const icons = {
  Overview: Layers3,
  Metrics: Activity,
  Tasks: Clock3,
  Usage: Coins,
  Backends: Server,
  "API keys": KeyRound,
  Proxies: Network,
  Routing: SlidersHorizontal,
};
const pages = Object.keys(icons) as Page[];
// The console is served at <gateway prefix>/admin/ (or admin/index.html).
// Resolve the gateway at runtime so one build works behind any proxy prefix.
const gatewayUrl = new URL("../", window.location.href).href.replace(/\/$/, "");
const usd = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 4,
  }).format(value);
const number = (value: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
const date = (value: number) => new Date(value * 1000).toLocaleString();
const identity = (section: Section, resource: Record<string, unknown>) =>
  String(
    section === "backends"
      ? resource.name
      : section === "keys"
        ? resource.id
        : resource.domain || new URL(String(resource.base_url)).hostname,
  );

function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const problem = error.response?.data;
    const details = Array.isArray(problem?.errors)
      ? problem.errors
          .map(
            (entry: { loc: string[]; msg: string }) =>
              `${entry.loc.join(".")}: ${entry.msg}`,
          )
          .join(" · ")
      : "";
    return (problem?.detail || error.message) + (details ? ` ${details}` : "");
  }
  return error instanceof Error
    ? error.message
    : "The request could not be completed.";
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <Database size={26} />
      <p>{children}</p>
    </div>
  );
}

function Badge({ value }: { value: string }) {
  return (
    <span
      className={`badge ${["active", "succeeded", "healthy"].includes(value) ? "positive" : ["failed", "rate limited", "degraded"].includes(value) ? "negative" : ""}`}
    >
      <span />
      {value}
    </span>
  );
}

function selectionState(health: Metrics["selection"][number]) {
  if (health.rate_limited) return "rate limited";
  if (health.success_rate === null) return "unobserved";
  return health.success_rate < 0.9 ? "degraded" : "healthy";
}

function App() {
  const [api, setApi] = useState<ManagementApi | null>(null);
  const [token, setToken] = useState("");
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [configuration, setConfiguration] = useState<ConfigResponse | null>(
    null,
  );
  const [page, setPage] = useState<Page>("Overview");
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [editor, setEditor] = useState<Editor | null>(null);
  const [deletion, setDeletion] = useState<{
    section: Section;
    id: string;
  } | null>(null);
  const [search, setSearch] = useState("");
  const [taskFilters, setTaskFilters] = useState<TaskFilters>({
    offset: 0,
    limit: 20,
  });
  const [taskPage, setTaskPage] = useState<TaskList | null>(null);
  const [rawMetrics, setRawMetrics] = useState<string | null>(null);
  const requestInFlight = useRef(false);
  const connection = useRef(0);

  const refresh = useCallback(
    async (client: ManagementApi, initial = false) => {
      if (requestInFlight.current && !initial) return;
      const generation = connection.current;
      requestInFlight.current = true;
      setRefreshing(true);
      try {
        const [status, metrics, usage, tasks, config] = await Promise.all([
          client.getManagementStatus(),
          client.getManagementMetrics(),
          client.listManagementUsage(),
          client.listManagementTasks({ limit: 8 }),
          initial ? client.getManagementConfig() : Promise.resolve(null),
        ]);
        if (generation !== connection.current) return;
        setSnapshot({
          status: status.data,
          metrics: metrics.data,
          usage: usage.data,
          tasks: tasks.data,
        });
        if (config) setConfiguration(config.data);
        setUpdatedAt(new Date());
        setError("");
      } catch (cause) {
        if (generation === connection.current) {
          setError(errorMessage(cause));
          if (initial) throw cause;
        }
      } finally {
        if (generation === connection.current) {
          requestInFlight.current = false;
          setRefreshing(false);
        }
      }
    },
    [],
  );

  async function connect(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const client = new ManagementApi(
      new Configuration({
        basePath: gatewayUrl,
        accessToken: token.trim(),
        baseOptions: { timeout: 15000 },
      }),
    );
    try {
      await refresh(client, true);
      setApi(client);
      setToken("");
    } catch {
      /* The refresh displays the gateway's problem detail. */
    } finally {
      setBusy(false);
    }
  }

  function disconnect() {
    connection.current += 1;
    requestInFlight.current = false;
    setApi(null);
    setSnapshot(null);
    setConfiguration(null);
    setToken("");
    setTaskPage(null);
    setRawMetrics(null);
    setEditor(null);
    setDeletion(null);
    setNotice("");
    setError("");
    setUpdatedAt(null);
  }

  useEffect(() => {
    if (!api || !autoRefresh) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh(api);
    }, 10000);
    return () => window.clearInterval(timer);
  }, [api, autoRefresh, refresh]);

  useEffect(() => {
    if (!api || page !== "Tasks") return;
    let alive = true;
    const load = async () => {
      try {
        const response = await api.listManagementTasks(taskFilters);
        if (alive) setTaskPage(response.data);
      } catch (cause) {
        if (alive) setError(errorMessage(cause));
      }
    };
    void load();
    return () => {
      alive = false;
    };
  }, [api, page, taskFilters, updatedAt]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 6000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function reloadConfig() {
    if (!api) return;
    try {
      setConfiguration((await api.getManagementConfig()).data);
      setError("");
      setNotice("Configuration reloaded.");
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  async function mutate(operation: () => Promise<{ data: ConfigResponse }>) {
    if (!api) return;
    setBusy(true);
    setError("");
    try {
      const result = await operation();
      setConfiguration(result.data);
      setEditor(null);
      setDeletion(null);
      setNotice("Changes applied to the gateway.");
      await refresh(api);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  function openEditor(section: Section, item?: Resource) {
    const draft: Record<string, unknown> = item
      ? structuredClone(item)
      : section === "backends"
        ? {
            name: "",
            type: "openai",
            enabled: true,
            api_key: "",
            base_url: null,
            tags: [],
            extra: {},
            credentials: [],
          }
        : section === "keys"
          ? {
              id: "",
              key: "",
              enabled: true,
              allow_tags: [],
              deny_tags: [],
              allow_backends: [],
              budget: null,
              extra: {},
            }
          : {
              base_url: "",
              enabled: true,
              tags: [],
              headers: {},
              accounts: [],
              timeout: 120,
              outbound_proxy: null,
            };
    setEditor({
      section,
      original: item ? identity(section, draft) : null,
      draft,
    });
  }

  function saveResource(draft: Record<string, unknown>) {
    if (!editor || !api || !configuration) return;
    if (
      editor.original &&
      editor.original !== identity(editor.section, draft)
    ) {
      setError(
        "Keep the existing resource identity. Create a new resource to use a different identity.",
      );
      return;
    }
    const revision = configuration.revision;
    void mutate(() =>
      editor.section === "backends"
        ? api.putManagementBackend(draft as Backend, revision)
        : editor.section === "keys"
          ? api.putManagementKey(draft as Key, revision)
          : api.putManagementProxy(draft as Proxy, revision),
    );
  }

  function deleteResource() {
    if (!deletion || !api || !configuration) return;
    const { section, id } = deletion;
    const revision = configuration.revision;
    void mutate(() =>
      section === "backends"
        ? api.deleteManagementBackend(id, revision)
        : section === "keys"
          ? api.deleteManagementKey(id, revision)
          : api.deleteManagementProxy(id, revision),
    );
  }

  async function showPrometheus() {
    if (!api) return;
    try {
      setRawMetrics((await api.getMetrics({ responseType: "text" })).data);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }

  if (!api || !snapshot || !configuration)
    return (
      <div className="login-shell">
        <div className="login-art" aria-hidden="true">
          <div className="orb" />
          <div className="orb second" />
          <span className="art-word">
            One gateway.
            <br />
            Every modality.
          </span>
          <div className="art-grid" />
        </div>
        <div className="login-content">
          <Logo />
          <div className="login-form">
            <span className="eyebrow">GATEWAY CONSOLE</span>
            <h1>
              Everything,
              <br />
              under control.
            </h1>
            <p className="muted">
              Manage your providers, tune routing, and see how your gateway is
              performing.
            </p>
            <form onSubmit={connect}>
              <label htmlFor="management-token">Management token</label>
              <input
                id="management-token"
                type="password"
                autoComplete="off"
                required
                value={token}
                onChange={(event) => setToken(event.target.value)}
                placeholder="Enter your management API key"
              />
              {error && (
                <div className="alert error" role="alert">
                  {error}
                </div>
              )}
              <button className="primary login-button" disabled={busy}>
                {busy ? (
                  <Loader2 className="spin" size={18} />
                ) : (
                  <>
                    Connect to gateway <ArrowRight size={18} />
                  </>
                )}
              </button>
            </form>
            <p className="login-note">
              <ShieldCheck size={15} /> Your token stays in this browser tab’s
              memory.
            </p>
            <div className="origin">
              <span className="dot" />
              {window.location.host}
            </div>
          </div>
          <span className="login-footer">mm-gateway / management console</span>
        </div>
      </div>
    );

  const { status, metrics, usage, tasks } = snapshot;
  const config = configuration.config;
  const calls = metrics.counters
    .filter((item) => item.name === "gateway_requests_total")
    .reduce((total, item) => total + item.value, 0);
  const duration = metrics.histograms.filter(
    (item) => item.name === "gateway_request_duration_seconds",
  );
  const observations = duration.reduce((total, item) => total + item.count, 0);
  const latency = observations
    ? duration.reduce((total, item) => total + item.sum, 0) / observations
    : null;
  const spent = usage.data.reduce(
    (total, item) => total + item.usage.key.spent_usd,
    0,
  );
  const active = status.backends.filter((backend) => backend.active).length;
  const descriptions: Record<Page, string> = {
    Overview: "A clear view of your multimodal gateway.",
    Metrics: "Live request counts, latency, and routing health.",
    Tasks: "Track generation across every modality.",
    Usage: "Spend, reservations, and budgets across your API keys.",
    Backends: "Connect providers and manage upstream accounts.",
    "API keys": "Control access, routing permissions, and spend limits.",
    Proxies: "Manage pass-through services and their account pools.",
    Routing: "Tune default selection, routing profiles, and model policies.",
  };

  const taskTable = (list: TaskList, compact = false) =>
    list.data.length ? (
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Task</th>
              <th>Model</th>
              <th>Status</th>
              {!compact && <th>Owner / backend</th>}
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {list.data.map((task) => (
              <tr key={task.id}>
                <td>
                  <div className="task-id" title={task.id}>
                    {task.id.slice(0, 20)}
                    <button
                      className="icon-button"
                      aria-label={`Copy ${task.id}`}
                      onClick={() => {
                        void navigator.clipboard
                          .writeText(task.id)
                          .then(() => setNotice("Task ID copied."))
                          .catch((cause) => setError(errorMessage(cause)));
                      }}
                    >
                      <Copy size={13} />
                    </button>
                  </div>
                  <span className="cell-secondary">{task.modality}</span>
                </td>
                <td>{task.model}</td>
                <td>
                  <Badge value={task.status} />
                </td>
                {!compact && (
                  <td>
                    {task.owner_key_id}
                    <span className="cell-secondary">{task.backend}</span>
                  </td>
                )}
                <td className="cell-date">{date(task.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ) : (
      <Empty>No tasks yet. Accepted generation tasks will appear here.</Empty>
    );

  const resourceList = (section: Section) => {
    const resources = (config[section] || []) as Resource[];
    const filtered = resources.filter((resource) =>
      JSON.stringify(resource).toLowerCase().includes(search.toLowerCase()),
    );
    return (
      <>
        <div className="section-toolbar">
          <div className="search">
            <Search size={16} />
            <input
              aria-label="Search resources"
              placeholder={`Search ${page.toLowerCase()}…`}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <div className="button-row">
            <button className="secondary" onClick={() => void reloadConfig()}>
              <RefreshCw size={15} /> Reload configuration
            </button>
            <button className="primary" onClick={() => openEditor(section)}>
              <Plus size={17} /> Add{" "}
              {section === "keys"
                ? "API key"
                : section === "proxies"
                  ? "proxy"
                  : "backend"}
            </button>
          </div>
        </div>
        {status.revision !== configuration.revision && (
          <div className="alert">
            Configuration has changed since you loaded it. Reload configuration
            before editing.
          </div>
        )}
        <div className="resource-grid">
          {filtered.map((resource) => {
            const item = resource as Record<string, unknown>;
            const id = identity(section, item);
            const live =
              section === "backends"
                ? status.backends.find((b) => b.name === id)
                : section === "proxies"
                  ? status.proxies.find((p) => p.domain === id)
                  : null;
            const tags = (item.tags || item.allow_tags || []) as string[];
            return (
              <article className="resource-card" key={id}>
                <div className="resource-top">
                  <div className="resource-icon">
                    {section === "backends" ? (
                      <Server size={20} />
                    ) : section === "keys" ? (
                      <KeyRound size={20} />
                    ) : (
                      <Network size={20} />
                    )}
                  </div>
                  <Badge
                    value={
                      item.enabled === false
                        ? "disabled"
                        : section === "keys" || live?.active
                          ? "active"
                          : "unconfigured"
                    }
                  />
                </div>
                <h3>{id}</h3>
                <p className="resource-description">
                  {section === "backends"
                    ? String(item.type)
                    : section === "proxies"
                      ? String(item.base_url)
                      : "Gateway API key"}
                </p>
                <div className="tags">
                  {tags.length ? (
                    tags.map((tag) => <span key={tag}>{tag}</span>)
                  ) : (
                    <span>
                      {section === "keys" &&
                      ((item.allow_backends as string[]) || []).length
                        ? "Backend permissions"
                        : section === "keys"
                          ? "All backends"
                          : "No tags"}
                    </span>
                  )}
                </div>
                <div className="resource-detail">
                  {section === "keys" ? (
                    <>
                      Budget{" "}
                      <strong>
                        {(resource as Key).budget?.limit_usd != null
                          ? `${usd((resource as Key).budget!.limit_usd!)} / ${(resource as Key).budget!.period || "month"}`
                          : "Unlimited"}
                      </strong>
                    </>
                  ) : (
                    <>
                      Accounts <strong>{live?.accounts.length || 0}</strong>
                    </>
                  )}
                </div>
                <div className="resource-actions">
                  <button onClick={() => openEditor(section, resource)}>
                    <Settings2 size={15} /> Configure
                  </button>
                  <button
                    className="icon-button danger"
                    aria-label={`Delete ${id}`}
                    onClick={() => setDeletion({ section, id })}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
        {!filtered.length && (
          <Empty>
            {resources.length
              ? "No matching resources."
              : `No ${page.toLowerCase()} configured. Add one to get started.`}
          </Empty>
        )}
      </>
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Logo />
        <div className="workspace">
          <div className="workspace-avatar">G</div>
          <div>
            <strong>Gateway workspace</strong>
            <span>{window.location.host}</span>
          </div>
          <span className="dot" />
        </div>
        <nav aria-label="Main navigation">
          {pages.map((name, index) => {
            const Icon = icons[name];
            return (
              <div key={name}>
                {index === 4 && <div className="nav-caption">MANAGEMENT</div>}
                <button
                  aria-current={page === name ? "page" : undefined}
                  className={page === name ? "selected" : ""}
                  onClick={() => {
                    setPage(name);
                    setSearch("");
                  }}
                >
                  <Icon size={18} />
                  {name}
                  {page === name && <span className="nav-dot" />}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="gateway-online">
            <span className="dot" />
            <div>
              Gateway online
              <span>
                v{status.version} ·{" "}
                {number(Math.floor(status.uptime_seconds / 60))}m uptime
              </span>
            </div>
          </div>
          <button onClick={disconnect}>
            <LogOut size={17} /> Disconnect
          </button>
          <a href={`${gatewayUrl}/docs`} target="_blank" rel="noreferrer">
            <CircleHelp size={17} /> API documentation{" "}
            <ExternalLink size={13} />
          </a>
        </div>
      </aside>
      <main>
        <div className="topbar">
          <span className="breadcrumb">
            Workspace <ChevronRight size={13} /> <strong>{page}</strong>
          </span>
          <div className="topbar-right">
            <span className="admin-label">
              <ShieldCheck size={14} /> Administrator
            </span>
            <span className="avatar">A</span>
            <button
              className="icon-button mobile-disconnect"
              aria-label="Disconnect"
              onClick={disconnect}
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
        <div className="content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                {pages.indexOf(page) < 4 ? "OBSERVABILITY" : "MANAGEMENT"}
              </span>
              <h1>{page}</h1>
              <p>{descriptions[page]}</p>
            </div>
            <div className="refresh-controls">
              <label>
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(event) => setAutoRefresh(event.target.checked)}
                />{" "}
                Auto-refresh <span>10s</span>
              </label>
              <button
                className="secondary"
                disabled={refreshing}
                onClick={() => void refresh(api)}
              >
                <RefreshCw size={15} className={refreshing ? "spin" : ""} />
                Refresh
              </button>
            </div>
          </div>
          {error && (
            <div className="alert error" role="alert">
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="toast" role="status">
              <CheckCircle2 size={17} />
              {notice}
            </div>
          )}
          {!configuration.persistent && pages.indexOf(page) >= 4 && (
            <div className="persistence-note">
              <Clock3 size={16} /> Changes apply immediately and last until this
              gateway restarts. Configure a management storage path to keep
              changes.
            </div>
          )}
          {page === "Overview" && (
            <>
              <div className="overview-banner">
                <div>
                  <span className="banner-label">
                    <span className="dot" /> SYSTEM STATUS
                  </span>
                  <h2>Your gateway at a glance.</h2>
                  <p>
                    {active} active backend{active === 1 ? "" : "s"} serving
                    image, video, music, and audio.
                  </p>
                </div>
                <div className="banner-symbol" aria-hidden="true">
                  <Layers3 size={72} strokeWidth={1} />
                </div>
                <span className="banner-chip">LIVE</span>
              </div>
              <div className="stats">
                <Stat
                  label="Provider calls"
                  value={number(calls)}
                  icon={<Activity size={18} />}
                  detail="Since process startup"
                />
                <Stat
                  label="Average call latency"
                  value={latency === null ? "—" : `${number(latency)}s`}
                  icon={<Clock3 size={18} />}
                  detail="Across provider operations"
                />
                <Stat
                  label="Current period spend"
                  value={usd(spent)}
                  icon={<Coins size={18} />}
                  detail="Across configured API keys"
                />
                <Stat
                  label="Active backends"
                  value={`${active} / ${status.backends.length}`}
                  icon={<Server size={18} />}
                  detail={`${status.enabled_keys_count} enabled API keys`}
                />
              </div>
              <div className="overview-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Calls by modality</h2>
                      <span>Process lifetime · provider operations</span>
                    </div>
                    <Activity size={17} />
                  </div>
                  <div className="modality-chart">
                    {["image", "video", "music", "audio", "voice"].map(
                      (modality) => {
                        const count = metrics.counters
                          .filter(
                            (c) =>
                              c.name === "gateway_requests_total" &&
                              c.labels.modality === modality,
                          )
                          .reduce((sum, c) => sum + c.value, 0);
                        return (
                          <div className="bar-row" key={modality}>
                            <span>{modality}</span>
                            <div className="bar-track">
                              <div
                                style={{
                                  width: `${calls ? (count / calls) * 100 : 0}%`,
                                }}
                              />
                            </div>
                            <strong>{number(count)}</strong>
                          </div>
                        );
                      },
                    )}
                  </div>
                  {calls === 0 && (
                    <p className="chart-note">
                      Traffic will appear after the first provider call.
                    </p>
                  )}
                </section>
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Routing health</h2>
                      <span>Recent outcomes · live selection state</span>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => setPage("Metrics")}
                    >
                      View metrics <ArrowRight size={14} />
                    </button>
                  </div>
                  {metrics.selection.length ? (
                    <div className="health-list">
                      {metrics.selection.slice(0, 4).map((health, index) => (
                        <div key={index}>
                          <span className="provider-avatar">
                            {health.backend.charAt(0).toUpperCase()}
                          </span>
                          <div>
                            <strong>{health.backend}</strong>
                            <span>
                              {health.account} · {health.modality}
                            </span>
                          </div>
                          <Badge value={selectionState(health)} />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <Empty>No routing observations yet.</Empty>
                  )}
                </section>
              </div>
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Recent tasks</h2>
                    <span>Latest generation activity</span>
                  </div>
                  <button
                    className="text-button"
                    onClick={() => setPage("Tasks")}
                  >
                    View all tasks <ArrowRight size={14} />
                  </button>
                </div>
                {taskTable(tasks, true)}
              </section>
            </>
          )}
          {page === "Metrics" && (
            <>
              <div className="section-toolbar">
                <p className="muted">
                  {metrics.enabled
                    ? `Collected ${new Date(metrics.collected_at).toLocaleTimeString()}`
                    : "Metrics are disabled on this gateway."}
                </p>
                <button
                  className="secondary"
                  onClick={() => void showPrometheus()}
                >
                  <ArrowDownToLine size={15} /> Prometheus exposition
                </button>
              </div>
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Provider & account health</h2>
                    <span>
                      Success and latency values decay over time. Attempts are
                      decayed estimates.
                    </span>
                  </div>
                </div>
                {metrics.selection.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Backend / account</th>
                          <th>Model / modality</th>
                          <th>Success rate</th>
                          <th>Latency</th>
                          <th>Attempts</th>
                          <th>Cooldown</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.selection.map((health, i) => (
                          <tr key={i}>
                            <td>
                              {health.backend}
                              <span className="cell-secondary">
                                {health.account}
                              </span>
                            </td>
                            <td>
                              {health.model || "All models"}
                              <span className="cell-secondary">
                                {health.modality}
                              </span>
                            </td>
                            <td>
                              {health.success_rate === null
                                ? "—"
                                : `${number(health.success_rate * 100)}%`}
                            </td>
                            <td>
                              {health.latency_s === null
                                ? "—"
                                : `${number(health.latency_s)}s`}
                            </td>
                            <td>{health.attempts}</td>
                            <td>
                              {health.rate_limited ? (
                                `${Math.ceil(health.cooldown_remaining_s)}s`
                              ) : (
                                <span>Ready</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No provider health observations yet.</Empty>
                )}
              </section>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Counters</h2>
                </div>
                {metrics.counters.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Labels</th>
                          <th>Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.counters.map((item, i) => (
                          <tr key={i}>
                            <td className="mono">{item.name}</td>
                            <td>
                              <Labels labels={item.labels} />
                            </td>
                            <td className="numeric">{number(item.value)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No counters recorded yet.</Empty>
                )}
              </section>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Duration summaries</h2>
                </div>
                {metrics.histograms.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Metric / labels</th>
                          <th>Count</th>
                          <th>Mean</th>
                          <th>Min</th>
                          <th>Max</th>
                          <th>Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.histograms.map((item, i) => (
                          <tr key={i}>
                            <td className="mono">
                              {item.name}
                              <Labels labels={item.labels} />
                            </td>
                            <td>{number(item.count)}</td>
                            <td>{number(item.mean)}s</td>
                            <td>{number(item.min)}s</td>
                            <td>{number(item.max)}s</td>
                            <td>{number(item.sum)}s</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No durations recorded yet.</Empty>
                )}
              </section>
            </>
          )}
          {page === "Tasks" && (
            <>
              <div className="section-toolbar task-filters">
                <select
                  aria-label="Filter modality"
                  value={taskFilters.modality || ""}
                  onChange={(event) =>
                    setTaskFilters({
                      ...taskFilters,
                      modality: event.target.value || undefined,
                      offset: 0,
                    })
                  }
                >
                  <option value="">All modalities</option>
                  {["image", "video", "music", "audio", "voice"].map(
                    (value) => (
                      <option key={value}>{value}</option>
                    ),
                  )}
                </select>
                <select
                  aria-label="Filter status"
                  value={taskFilters.status || ""}
                  onChange={(event) =>
                    setTaskFilters({
                      ...taskFilters,
                      status: event.target.value || undefined,
                      offset: 0,
                    })
                  }
                >
                  <option value="">All statuses</option>
                  {[
                    "pending",
                    "running",
                    "succeeded",
                    "failed",
                    "cancelled",
                    "expired",
                  ].map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
                <select
                  aria-label="Filter owner"
                  value={taskFilters.key_id || ""}
                  onChange={(event) =>
                    setTaskFilters({
                      ...taskFilters,
                      key_id: event.target.value || undefined,
                      offset: 0,
                    })
                  }
                >
                  <option value="">All API keys</option>
                  {(config.keys || []).map((key) => (
                    <option key={key.id}>{key.id}</option>
                  ))}
                </select>
              </div>
              <section className="panel">
                {taskPage ? taskTable(taskPage) : <Empty>Loading tasks…</Empty>}
                {taskPage && (
                  <div className="pagination">
                    <span>
                      {number(taskPage.total)} tasks · showing{" "}
                      {taskPage.total ? taskPage.offset + 1 : 0}–
                      {taskPage.offset + taskPage.data.length}
                    </span>
                    <div className="button-row">
                      <button
                        className="secondary"
                        aria-label="Previous page"
                        disabled={taskPage.offset === 0}
                        onClick={() =>
                          setTaskFilters({
                            ...taskFilters,
                            offset: Math.max(
                              0,
                              taskPage.offset - taskPage.limit,
                            ),
                          })
                        }
                      >
                        <ChevronLeft size={16} />
                      </button>
                      <button
                        className="secondary"
                        aria-label="Next page"
                        disabled={
                          taskPage.offset + taskPage.limit >= taskPage.total
                        }
                        onClick={() =>
                          setTaskFilters({
                            ...taskFilters,
                            offset: taskPage.offset + taskPage.limit,
                          })
                        }
                      >
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </section>
            </>
          )}
          {page === "Usage" && (
            <>
              <div className="stats usage-stats">
                <Stat
                  label="Current period spend"
                  value={usd(spent)}
                  icon={<Coins size={18} />}
                  detail="Each key’s configured budget period"
                />
                <Stat
                  label="Reserved"
                  value={usd(
                    usage.data.reduce(
                      (total, item) => total + item.usage.key.reserved_usd,
                      0,
                    ),
                  )}
                  icon={<Clock3 size={18} />}
                  detail="Held for unfinished tasks"
                />
                <Stat
                  label="Settled tasks"
                  value={number(
                    usage.data.reduce(
                      (total, item) => total + (item.usage.key.tasks || 0),
                      0,
                    ),
                  )}
                  icon={<CheckCircle2 size={18} />}
                  detail="Tasks recorded in the cost ledger"
                />
              </div>
              {usage.data.length ? (
                usage.data.map((item) => (
                  <section className="panel" key={item.key_id}>
                    <div className="panel-heading">
                      <div>
                        <h2>{item.key_id}</h2>
                        <span>
                          {item.usage.period.kind} budget period · USD
                        </span>
                      </div>
                      <Badge value={item.enabled ? "active" : "disabled"} />
                    </div>
                    <div className="budget-row">
                      <div>
                        <span>Spent</span>
                        <strong>{usd(item.usage.key.spent_usd)}</strong>
                      </div>
                      <div>
                        <span>Reserved</span>
                        <strong>{usd(item.usage.key.reserved_usd)}</strong>
                      </div>
                      <div>
                        <span>Limit</span>
                        <strong>
                          {item.usage.key.limit_usd == null
                            ? "Unlimited"
                            : usd(item.usage.key.limit_usd)}
                        </strong>
                      </div>
                      <div>
                        <span>Remaining</span>
                        <strong>
                          {item.usage.key.remaining_usd == null
                            ? "Unlimited"
                            : usd(item.usage.key.remaining_usd)}
                        </strong>
                      </div>
                    </div>
                    {(item.usage.scopes || []).length > 0 && (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Scope</th>
                              <th>Spent</th>
                              <th>Reserved</th>
                              <th>Remaining</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(item.usage.scopes || []).map((scope) => (
                              <tr key={scope.scope}>
                                <td>{scope.scope}</td>
                                <td>{usd(scope.spent_usd)}</td>
                                <td>{usd(scope.reserved_usd)}</td>
                                <td>
                                  {scope.remaining_usd == null
                                    ? "Unlimited"
                                    : usd(scope.remaining_usd)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {(item.usage.models || []).length > 0 && (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Modality</th>
                              <th>Spend</th>
                              <th>Tasks</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(item.usage.models || []).map((model, i) => (
                              <tr key={i}>
                                <td>{model.model}</td>
                                <td>{model.modality}</td>
                                <td>{usd(model.spent_usd)}</td>
                                <td>{model.tasks}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </section>
                ))
              ) : (
                <Empty>No API keys configured.</Empty>
              )}
            </>
          )}
          {page === "Backends" && resourceList("backends")}
          {page === "API keys" && resourceList("keys")}
          {page === "Proxies" && resourceList("proxies")}
          {page === "Routing" && (
            <RoutingEditor
              key={configuration.revision}
              config={config}
              busy={busy}
              onReload={() => void reloadConfig()}
              onSave={(next) =>
                void mutate(() =>
                  api.replaceManagementConfig(next, configuration.revision),
                )
              }
            />
          )}
          <footer className="content-footer">
            <span>
              <span className="dot" /> Connected to {window.location.host}
            </span>
            <span>
              {updatedAt
                ? `Updated ${updatedAt.toLocaleTimeString()}`
                : "Waiting for gateway"}
            </span>
          </footer>
        </div>
      </main>
      {editor && (
        <ResourceEditor
          editor={editor}
          types={status.backend_types}
          busy={busy}
          error={error}
          onClose={() => setEditor(null)}
          onSave={saveResource}
        />
      )}
      {deletion && (
        <div className="modal-overlay">
          <div
            className="confirm-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-title"
          >
            <div className="delete-symbol">
              <Trash2 size={24} />
            </div>
            <h2 id="delete-title">Delete {deletion.id}?</h2>
            <p>
              This removes the resource from the gateway configuration. Existing
              task records remain available in the management console.
            </p>
            {error && (
              <div className="alert error" role="alert">
                {error}
              </div>
            )}
            <div className="button-row">
              <button
                className="secondary"
                disabled={busy}
                onClick={() => setDeletion(null)}
              >
                Cancel
              </button>
              <button
                className="destructive"
                disabled={busy}
                onClick={deleteResource}
              >
                {busy ? "Deleting…" : "Delete resource"}
              </button>
            </div>
          </div>
        </div>
      )}
      {rawMetrics !== null && (
        <div className="modal-overlay">
          <div
            className="metrics-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="prometheus-title"
          >
            <div className="panel-heading">
              <h2 id="prometheus-title">Prometheus exposition</h2>
              <button
                className="icon-button"
                aria-label="Close metrics"
                onClick={() => setRawMetrics(null)}
              >
                <X size={20} />
              </button>
            </div>
            <pre>{rawMetrics || "No metrics recorded."}</pre>
            <div className="dialog-footer">
              <button
                className="secondary"
                onClick={() => {
                  const url = URL.createObjectURL(
                    new Blob([rawMetrics], { type: "text/plain" }),
                  );
                  const anchor = document.createElement("a");
                  anchor.href = url;
                  anchor.download = "mm-gateway-metrics.txt";
                  anchor.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <ArrowDownToLine size={16} /> Download metrics
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Logo() {
  return (
    <div className="logo">
      <span>
        <svg width="24" height="24" viewBox="0 0 40 40" fill="none">
          <path
            d="M5 29V12l9 11 6-11 6 11 9-11v17"
            stroke="currentColor"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <strong>
        mm-gateway<span>console</span>
      </strong>
    </div>
  );
}
function Stat({
  label,
  value,
  icon,
  detail,
}: {
  label: string;
  value: string;
  icon: ReactNode;
  detail: string;
}) {
  return (
    <article className="stat">
      <div className="stat-label">
        {label}
        {icon}
      </div>
      <strong>{value}</strong>
      <span className="stat-detail">{detail}</span>
    </article>
  );
}
function Labels({ labels }: { labels: Record<string, string> }) {
  return (
    <div className="metric-labels">
      {Object.entries(labels).map(([name, value]) => (
        <span key={name}>
          {name}: {value}
        </span>
      ))}
    </div>
  );
}

function ResourceEditor({
  editor,
  types,
  busy,
  error,
  onClose,
  onSave,
}: {
  editor: Editor;
  types: string[];
  busy: boolean;
  error: string;
  onClose: () => void;
  onSave: (draft: Record<string, unknown>) => void;
}) {
  const [draft, setDraft] = useState(editor.draft);
  const [mode, setMode] = useState<"form" | "json">("form");
  const [raw, setRaw] = useState(JSON.stringify(editor.draft, null, 2));
  const [localError, setLocalError] = useState("");
  const dialog = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  const busyRef = useRef(busy);
  closeRef.current = onClose;
  busyRef.current = busy;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLInputElement>("input")?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape" && !busyRef.current) closeRef.current();
      if (event.key === "Tab" && dialog.current) {
        const elements = Array.from(
          dialog.current.querySelectorAll<HTMLElement>(
            "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea",
          ),
        );
        const first = elements[0],
          last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      previous?.focus();
    };
  }, []);
  const set = (name: string, value: unknown) =>
    setDraft((current) => ({ ...current, [name]: value }));
  const field = (
    name: string,
    label: string,
    secret = false,
    locked = false,
  ) => (
    <label className="field" key={name}>
      {label}
      <input
        aria-label={label}
        type={secret ? "password" : "text"}
        disabled={locked && !!editor.original}
        value={
          secret && draft[name] === "[redacted]"
            ? ""
            : String(draft[name] ?? "")
        }
        placeholder={
          secret && editor.original
            ? "Leave blank to preserve credential"
            : undefined
        }
        onChange={(event) =>
          set(
            name,
            secret && !event.target.value && editor.original
              ? "[redacted]"
              : event.target.value || null,
          )
        }
      />
    </label>
  );
  const tags = (name: string, label: string) => (
    <label className="field" key={name}>
      {label}
      <input
        value={((draft[name] as string[]) || []).join(", ")}
        onChange={(event) =>
          set(
            name,
            event.target.value
              .split(",")
              .map((value) => value.trim())
              .filter(Boolean),
          )
        }
        placeholder="Comma-separated values"
      />
    </label>
  );
  const jsonField = (name: string, label: string) => (
    <JsonField
      key={name}
      label={label}
      value={draft[name]}
      onChange={(value) => set(name, value)}
    />
  );

  function switchMode(next: "form" | "json") {
    try {
      if (next === "form") {
        const parsed = JSON.parse(raw);
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
          throw new Error("The resource must be a JSON object.");
        setDraft(parsed);
      } else setRaw(JSON.stringify(draft, null, 2));
      setMode(next);
      setLocalError("");
    } catch (cause) {
      setLocalError(errorMessage(cause));
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    try {
      const value = mode === "json" ? JSON.parse(raw) : draft;
      identity(editor.section, value);
      onSave(value);
    } catch (cause) {
      setLocalError(errorMessage(cause));
    }
  }
  const title = `${editor.original ? "Configure" : "Add"} ${editor.section === "keys" ? "API key" : editor.section === "proxies" ? "proxy" : "backend"}`;
  return (
    <div className="modal-overlay">
      <div
        className="editor-dialog"
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="editor-title"
      >
        <form onSubmit={submit}>
          <div className="panel-heading">
            <div>
              <h2 id="editor-title">{title}</h2>
              <span>Changes apply to the running gateway.</span>
            </div>
            <button
              type="button"
              className="icon-button"
              aria-label="Close editor"
              disabled={busy}
              onClick={onClose}
            >
              <X size={21} />
            </button>
          </div>
          <div className="editor-tabs">
            <button
              type="button"
              className={mode === "form" ? "active" : ""}
              onClick={() => switchMode("form")}
            >
              Details
            </button>
            <button
              type="button"
              className={mode === "json" ? "active" : ""}
              onClick={() => switchMode("json")}
            >
              Full JSON
            </button>
          </div>
          <div className="editor-body">
            {(localError || error) && (
              <div className="alert error" role="alert">
                {localError || error}
              </div>
            )}
            <p className="editor-note">
              Existing secrets appear as <code>[redacted]</code>. Keep that
              value to preserve them.
            </p>
            {mode === "json" ? (
              <label className="field">
                Resource JSON
                <textarea
                  className="code-editor"
                  rows={20}
                  spellCheck={false}
                  value={raw}
                  onChange={(event) => setRaw(event.target.value)}
                />
              </label>
            ) : (
              <>
                {editor.section === "backends" && (
                  <>
                    {field("name", "Backend name", false, true)}
                    <label className="field">
                      Provider type
                      <select
                        value={String(draft.type)}
                        onChange={(event) => set("type", event.target.value)}
                      >
                        {types.map((type) => (
                          <option key={type}>{type}</option>
                        ))}
                      </select>
                    </label>
                    {field("api_key", "Provider API key", true)}
                    {field("base_url", "Base URL")}
                    {tags("tags", "Routing tags")}
                    {jsonField("credentials", "Account pool (JSON)")}
                    {jsonField("extra", "Provider options (JSON)")}
                  </>
                )}
                {editor.section === "keys" && (
                  <>
                    {field("id", "Key ID", false, true)}
                    {field("key", "API token", true)}
                    {tags("allow_tags", "Allowed tags")}
                    {tags("allow_backends", "Allowed backend names")}
                    {tags(
                      "deny_tags",
                      "Denied backend names, proxy domains, or provider types",
                    )}
                    <p className="field-hint">
                      Empty allowed tags and backends permit every backend. Deny
                      rules still apply.
                    </p>
                    {jsonField("budget", "Budget (JSON)")}
                    <p className="field-hint">
                      Example:{" "}
                      {JSON.stringify({
                        limit_usd: 50,
                        period: "month",
                        scopes_limit_usd: 10,
                      })}
                      . Use null for no budget.
                    </p>
                    <details>
                      <summary>Routing defaults & additional options</summary>
                      {["image", "video", "music", "audio"].map((modality) =>
                        field(
                          `default_${modality}_backend`,
                          `Default ${modality} backend`,
                        ),
                      )}
                      {["image", "video", "music", "audio"].map((modality) =>
                        field(
                          `default_${modality}_tag`,
                          `Default ${modality} tag`,
                        ),
                      )}
                      {jsonField("extra", "Additional options (JSON)")}
                    </details>
                  </>
                )}
                {editor.section === "proxies" && (
                  <>
                    {field("base_url", "Upstream base URL", false, true)}
                    {tags("tags", "Routing tags")}
                    <label className="field">
                      Timeout (seconds)
                      <input
                        type="number"
                        min="0.1"
                        step="any"
                        value={Number(draft.timeout)}
                        onChange={(event) =>
                          set("timeout", Number(event.target.value))
                        }
                      />
                    </label>
                    {field("outbound_proxy", "Outbound proxy URL", true)}
                    {jsonField("headers", "Shared headers (JSON)")}
                    {jsonField("accounts", "Account pool (JSON)")}
                  </>
                )}
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={draft.enabled !== false}
                    onChange={(event) => set("enabled", event.target.checked)}
                  />{" "}
                  Enabled
                </label>
              </>
            )}
          </div>
          <div className="dialog-footer">
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={onClose}
            >
              Cancel
            </button>
            <button className="primary" disabled={busy}>
              {busy ? (
                <Loader2 className="spin" size={16} />
              ) : (
                <Check size={16} />
              )}{" "}
              Save changes
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function JsonField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [raw, setRaw] = useState(JSON.stringify(value ?? null, null, 2));
  const [error, setError] = useState("");
  return (
    <label className="field">
      {label}
      <textarea
        aria-label={label}
        rows={4}
        className="code-editor"
        value={raw}
        onChange={(event) => {
          setRaw(event.target.value);
          try {
            onChange(JSON.parse(event.target.value));
            setError("");
            event.target.setCustomValidity("");
          } catch {
            setError("Enter valid JSON.");
            event.target.setCustomValidity("Enter valid JSON.");
          }
        }}
      />
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}

function RoutingEditor({
  config,
  busy,
  onReload,
  onSave,
}: {
  config: Config;
  busy: boolean;
  onReload: () => void;
  onSave: (config: Config) => void;
}) {
  const [draft, setDraft] = useState(config);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSave(draft);
      }}
    >
      <div className="section-toolbar">
        <p className="muted">Routing changes affect new generation requests.</p>
        <div className="button-row">
          <button type="button" className="secondary" onClick={onReload}>
            <RefreshCw size={15} />
            Reload configuration
          </button>
          <button className="primary" disabled={busy}>
            <Check size={16} />
            Save routing
          </button>
        </div>
      </div>
      <section className="panel routing-panel">
        <h2>Default selection</h2>
        <p className="muted">
          Choose how the gateway ranks eligible candidates.
        </p>
        <div className="optimize-options">
          {(["balanced", "cost", "latency"] as const).map((optimize) => (
            <label
              className={
                draft.routing_default_optimize === optimize ? "chosen" : ""
              }
              key={optimize}
            >
              <input
                type="radio"
                aria-label={
                  optimize.charAt(0).toUpperCase() + optimize.slice(1)
                }
                name="optimize"
                value={optimize}
                checked={draft.routing_default_optimize === optimize}
                onChange={() =>
                  setDraft({ ...draft, routing_default_optimize: optimize })
                }
              />
              <strong>
                {optimize.charAt(0).toUpperCase() + optimize.slice(1)}
              </strong>
              <span>
                {optimize === "balanced"
                  ? "Balance health, latency, and cost."
                  : optimize === "cost"
                    ? "Prefer lower estimated cost."
                    : "Prefer faster provider responses."}
              </span>
            </label>
          ))}
        </div>
        <JsonField
          label="Routing profiles (JSON)"
          value={draft.routing_profiles}
          onChange={(value) =>
            setDraft({
              ...draft,
              routing_profiles: value as Config["routing_profiles"],
            })
          }
        />
        <p className="field-hint">
          Example:{" "}
          {JSON.stringify({
            fast: { optimize: "latency", tags: ["production"] },
          })}
        </p>
        <JsonField
          label="Model catalog overrides (JSON)"
          value={draft.catalog_models}
          onChange={(value) =>
            setDraft({
              ...draft,
              catalog_models: value as Config["catalog_models"],
            })
          }
        />
        <label className="field">
          Global outbound proxy
          <input
            type="password"
            placeholder="Optional HTTP / SOCKS proxy URL"
            value={draft.outbound_proxy || ""}
            onChange={(event) =>
              setDraft({ ...draft, outbound_proxy: event.target.value || null })
            }
          />
        </label>
        <label className="check-field">
          <input
            type="checkbox"
            checked={draft.budget_allow_unpriced || false}
            onChange={(event) =>
              setDraft({
                ...draft,
                budget_allow_unpriced: event.target.checked,
              })
            }
          />
          Allow unpriced models for budgeted requests
        </label>
      </section>
    </form>
  );
}

export default App;
