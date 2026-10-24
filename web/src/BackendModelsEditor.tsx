import { useEffect, useRef, useState } from "react";
import { Loader2, Plus, RefreshCw, Trash2 } from "lucide-react";
import axios from "axios";
import {
  ManagementApi,
  type Backend,
  type ModelCatalog,
  type ModelDiscovery,
} from "./sdk/management";
import { ConfigurationExample } from "./ConfigurationExample";

type Modality = "image" | "video" | "music" | "audio";
type Models = Partial<Record<Modality, string[]>>;
const modalities: Modality[] = ["image", "video", "music", "audio"];
const validId = (id: string) =>
  id.length > 0 &&
  id.length <= 256 &&
  !/\s/.test(id) &&
  id.toLowerCase() !== "auto";
const message = (error: unknown) =>
  axios.isAxiosError(error)
    ? error.response?.data?.detail || error.message
    : error instanceof Error
      ? error.message
      : "Could not load models.";

export function BackendModelsEditor({
  api,
  backend,
  onChange,
}: {
  api: ManagementApi;
  backend: Backend;
  onChange: (backend: Backend) => void;
}) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [discovered, setDiscovered] = useState<ModelDiscovery | null>(null);
  const [discovering, setDiscovering] = useState(false);
  const [error, setError] = useState("");
  const [modality, setModality] = useState<Modality>("image");
  const [manual, setManual] = useState("");
  const [alias, setAlias] = useState("");
  const [target, setTarget] = useState("");
  const [search, setSearch] = useState("");
  const discoveryRequest = useRef(0);
  const discoveryIdentity = JSON.stringify([
    backend.type,
    backend.base_url,
    backend.api_key,
    backend.credentials,
    backend.extra,
  ]);

  useEffect(() => {
    let alive = true;
    void api
      .getManagementBuiltinModels(backend.type)
      .then(({ data }) => {
        if (!alive) return;
        setCatalog(data);
        const supported = modalities.filter((item) => item in data.models);
        setModality(supported[0] || "image");
      })
      .catch((cause) => {
        if (alive) setCatalogError(message(cause));
      });
    return () => {
      alive = false;
    };
  }, [api, backend.type]);

  useEffect(() => {
    discoveryRequest.current += 1;
    setDiscovered(null);
    setDiscovering(false);
    return () => {
      discoveryRequest.current += 1;
    };
  }, [discoveryIdentity]);

  const defaults: Models = { ...catalog?.models };
  for (const item of modalities) {
    const pinned = backend.extra?.[`${item}_model`];
    if (typeof pinned === "string" && item in defaults)
      defaults[item] = [...new Set([...(defaults[item] || []), pinned])];
  }
  if (backend.extra?.audio_only)
    for (const item of ["image", "video", "music"] as const)
      delete defaults[item];
  const automatic = backend.models == null;
  const active: Models = automatic ? defaults : backend.models!;
  const supported = catalog
    ? modalities.filter((item) => item in defaults)
    : modalities;
  const chosenModality = supported.includes(modality)
    ? modality
    : supported[0] || "image";
  const activeIds = [...new Set(Object.values(active).flat())];
  const aliases = backend.model_aliases || {};
  const chosenTarget = activeIds.includes(target) ? target : activeIds[0] || "";

  function toggle(item: Modality, id: string, enabled: boolean) {
    const next = {
      ...active,
      [item]: enabled
        ? [...new Set([...(active[item] || []), id])]
        : (active[item] || []).filter((entry) => entry !== id),
    };
    onChange({ ...backend, models: next });
    setError("");
  }

  async function discover() {
    if (!backend.name) {
      setError(
        "Enter a backend name and provider credentials before listing upstream models.",
      );
      return;
    }
    const request = ++discoveryRequest.current;
    setDiscovering(true);
    setError("");
    try {
      const result = await api.discoverManagementBackendModels(backend);
      if (request === discoveryRequest.current) setDiscovered(result.data);
    } catch (cause) {
      if (request === discoveryRequest.current) setError(message(cause));
    } finally {
      if (request === discoveryRequest.current) setDiscovering(false);
    }
  }

  return (
    <section className="backend-models" aria-label="Backend models">
      <h3>Active models</h3>
      <p className="field-hint">
        Choose which models clients can use on this backend. Selected IDs also
        drive automatic routing. New model IDs can be added without a gateway
        release.
      </p>
      <label className="check-field">
        <input
          type="checkbox"
          checked={automatic}
          disabled={!catalog}
          onChange={(event) =>
            onChange({
              ...backend,
              models: event.target.checked ? null : defaults,
            })
          }
        />
        Use provider defaults
      </label>
      <p className="field-hint">
        {automatic
          ? "Provider defaults are enabled. Selecting or clearing a model switches to an explicit allowlist."
          : "Only selected models are allowed. Unselected models are blocked, including through aliases. An empty selection disables all models."}
      </p>
      {catalogError && (
        <p className="field-error" role="alert">
          {catalogError}
        </p>
      )}
      {!catalog && !catalogError && (
        <p className="field-hint">Loading built-in models…</p>
      )}
      {supported.map((item) => {
        const options = [
          ...new Set([...(defaults[item] || []), ...(active[item] || [])]),
        ];
        return (
          <fieldset className="model-options" key={item}>
            <legend>
              {item.charAt(0).toUpperCase() + item.slice(1)} models
            </legend>
            {options.length ? (
              options.map((id) => (
                <label className="model-option" key={id}>
                  <input
                    type="checkbox"
                    aria-label={`Activate ${item} model ${id}`}
                    checked={(active[item] || []).includes(id)}
                    onChange={(event) => toggle(item, id, event.target.checked)}
                  />
                  <code>{id}</code>
                  <span>
                    {(defaults[item] || []).includes(id)
                      ? "Built-in"
                      : "Custom"}
                  </span>
                </label>
              ))
            ) : (
              <p className="field-hint">
                No built-in models. Add an upstream or manual model below.
              </p>
            )}
          </fieldset>
        );
      })}
      {!automatic && !activeIds.length && (
        <p className="model-notice">
          No models are active. This backend will not accept new generation
          requests.
        </p>
      )}

      <div className="model-toolbar">
        <h4>Discover upstream models</h4>
        <button
          type="button"
          className="secondary"
          disabled={discovering}
          onClick={() => void discover()}
        >
          {discovering ? (
            <Loader2 className="spin" size={15} />
          ) : (
            <RefreshCw size={15} />
          )}{" "}
          List models from backend
        </button>
      </div>
      <p className="field-hint">
        Uses the draft credentials and Base URL without saving, or the first
        configured account in a pool. Discovery only suggests models; select one
        to activate it. If the upstream has no models API, use the built-in list
        or manual entry.
      </p>
      <label className="field">
        Model modality
        <select
          aria-label="Model modality"
          value={chosenModality}
          onChange={(event) => setModality(event.target.value as Modality)}
        >
          {supported.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      <p className="field-hint">
        Used for manual models and upstream IDs whose modality is unknown.
        Choose the generation endpoint supported by that model.
      </p>
      {discovered && (
        <div className="discovered-models">
          <label className="field">
            Search upstream models
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Filter model IDs"
            />
          </label>
          <p className="field-hint">
            {discovered.data.length} models returned by the upstream.
          </p>
          <div className="model-results">
            {discovered.data
              .filter((model) =>
                model.id.toLowerCase().includes(search.toLowerCase()),
              )
              .map((model) => {
                const item =
                  model.modalities?.find((entry) =>
                    supported.includes(entry),
                  ) || chosenModality;
                return (
                  <label className="model-option" key={model.id}>
                    <input
                      type="checkbox"
                      aria-label={`Activate discovered ${item} model ${model.id}`}
                      checked={(active[item] || []).includes(model.id)}
                      onChange={(event) =>
                        toggle(item, model.id, event.target.checked)
                      }
                    />
                    <code>{model.id}</code>
                    <span>{item}</span>
                  </label>
                );
              })}
          </div>
          {!discovered.data.length && (
            <p className="field-hint">
              The upstream returned no models. You can still add a model
              manually.
            </p>
          )}
        </div>
      )}
      <div className="model-entry">
        <label className="field">
          Manual model ID
          <input
            value={manual}
            placeholder="e.g. provider-image-next"
            onChange={(event) => setManual(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="secondary"
          disabled={!manual.trim()}
          onClick={() => {
            const id = manual.trim();
            if (!validId(id)) {
              setError("Enter a model ID without spaces; 'auto' is reserved.");
              return;
            }
            toggle(chosenModality, id, true);
            setManual("");
          }}
        >
          <Plus size={15} /> Add model
        </button>
      </div>

      <h3>Model aliases</h3>
      <p className="field-hint">
        Give an active model a stable public name. Requests using the alias are
        sent to its target model on this backend. Clients still need permission
        to use the backend.
      </p>
      {Object.entries(aliases).map(([name, model]) => (
        <div className="model-alias" key={name}>
          <code>{name}</code>
          <span>→</span>
          <select
            aria-label={`Target for alias ${name}`}
            value={model}
            onChange={(event) =>
              onChange({
                ...backend,
                model_aliases: { ...aliases, [name]: event.target.value },
              })
            }
          >
            {[...new Set([model, ...activeIds])].map((id) => (
              <option key={id} value={id}>
                {id}
                {activeIds.includes(id) ? "" : " (inactive)"}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="icon-button danger"
            aria-label={`Remove alias ${name}`}
            onClick={() => {
              const next = { ...aliases };
              delete next[name];
              onChange({ ...backend, model_aliases: next });
            }}
          >
            <Trash2 size={15} />
          </button>
        </div>
      ))}
      <div className="alias-entry">
        <label className="field">
          Model alias
          <input
            value={alias}
            placeholder="e.g. production-image"
            onChange={(event) => setAlias(event.target.value)}
          />
        </label>
        <label className="field">
          Alias target model
          <select
            aria-label="Alias target model"
            value={chosenTarget}
            onChange={(event) => setTarget(event.target.value)}
            disabled={!activeIds.length}
          >
            {!activeIds.length && (
              <option value="">Activate a model first</option>
            )}
            {activeIds.map((id) => (
              <option value={id} key={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary"
          disabled={!alias.trim() || !chosenTarget}
          onClick={() => {
            const name = alias.trim();
            if (
              !validId(name) ||
              name === chosenTarget ||
              name in aliases ||
              activeIds.includes(name)
            ) {
              setError(
                "Choose a unique alias, different from existing models. 'auto' is reserved.",
              );
              return;
            }
            onChange({
              ...backend,
              model_aliases: { ...aliases, [name]: chosenTarget },
            });
            setAlias("");
            setError("");
          }}
        >
          <Plus size={15} /> Add alias
        </button>
      </div>
      {Object.values(aliases).some((model) => !activeIds.includes(model)) && (
        <p className="field-error" role="alert">
          An alias targets an inactive model. Remove that alias or activate its
          target before saving.
        </p>
      )}
      {error && (
        <div className="alert error" role="alert">
          {error}
        </div>
      )}
      <ConfigurationExample
        title="Model selection & alias example"
        value={{
          models: { [chosenModality]: [chosenTarget || "provider-image-next"] },
          model_aliases: {
            "production-model": chosenTarget || "provider-image-next",
          },
        }}
      >
        <p>
          Clients send <code>{'{"model":"production-model"}'}</code>. The
          gateway sends{" "}
          <code>
            {JSON.stringify({ model: chosenTarget || "provider-image-next" })}
          </code>{" "}
          to this backend. Keep the alias and change its target when you release
          a new model.
        </p>
        <p>
          This allowlist enables only the selected model and modality. Remove an
          alias before deactivating its target. Discovery never activates models
          automatically.
        </p>
      </ConfigurationExample>
    </section>
  );
}
