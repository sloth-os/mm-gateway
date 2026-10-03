import { test, expect, type Page } from "@playwright/test";

async function connect(page: Page) {
  await page.goto("admin/");
  await page.getByLabel("Management token").fill("management-test");
  await page.getByRole("button", { name: "Connect to gateway" }).click();
  await expect(
    page.getByRole("heading", { name: "Overview", exact: true }),
  ).toBeVisible();
}

test("rejects generation tokens and keeps admin credentials in tab memory", async ({
  page,
}) => {
  await page.goto("admin/");
  await page.getByLabel("Management token").fill("browser-generation");
  await page.getByRole("button", { name: "Connect to gateway" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "management bearer token",
  );
  await connect(page);
  expect(
    await page.evaluate(() => [localStorage.length, sessionStorage.length]),
  ).toEqual([0, 0]);
  await page.getByRole("button", { name: "Disconnect", exact: true }).click();
  await expect(page.getByLabel("Management token")).toHaveValue("");
});

test("shows real gateway metrics, tasks, usage and Prometheus through the SDK", async ({
  page,
  request,
}) => {
  const create = await request.post("v1/images", {
    headers: { Authorization: "Bearer browser-generation" },
    data: { input: [{ type: "text", text: "browser integration task" }] },
  });
  expect(create.status()).toBe(202);
  const task = await create.json();
  await connect(page);
  await expect(
    page.getByText("test-provider", { exact: true }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Tasks", exact: true }).click();
  await expect(
    page.locator(".task-id", { hasText: task.id.slice(0, 20) }),
  ).toBeVisible();
  await page.getByLabel("Filter status").selectOption("failed");
  await expect(page.getByText("No tasks yet.", { exact: false })).toBeVisible();
  await page.getByLabel("Filter status").selectOption("succeeded");
  await expect(
    page.locator(".task-id", { hasText: task.id.slice(0, 20) }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Metrics", exact: true }).click();
  await expect(
    page.getByText("gateway_requests_total", { exact: true }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Prometheus exposition" }).click();
  await expect(page.getByRole("dialog").locator("pre")).toContainText(
    "gateway_async_tasks_submitted_total",
  );
  await page.getByRole("button", { name: "Close metrics" }).click();
  await page.getByRole("button", { name: "Usage", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "browser", exact: true }),
  ).toBeVisible();
});

test("creates, configures, disables, and deletes a backend", async ({
  page,
}) => {
  await connect(page);
  await page.getByRole("button", { name: "Backends", exact: true }).click();
  await page.getByRole("button", { name: "Add backend" }).click();
  await page
    .getByLabel("Backend name", { exact: true })
    .fill("browser-backend");
  await page.getByLabel("Provider type").selectOption("fake");
  await page
    .getByLabel("Provider API key", { exact: true })
    .fill("browser-provider-secret");
  await page
    .getByLabel("Routing tags", { exact: true })
    .fill("development, browser");
  await page.getByRole("button", { name: "Save changes" }).click();
  const card = page.locator(".resource-card").filter({
    has: page.getByRole("heading", { name: "browser-backend", exact: true }),
  });
  await expect(card).toBeVisible();
  await expect(card.getByText("active", { exact: true })).toBeVisible();
  await card.getByRole("button", { name: "Configure" }).click();
  await expect(
    page.getByLabel("Provider API key", { exact: true }),
  ).toHaveValue("");
  await page.getByLabel("Enabled", { exact: true }).uncheck();
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(card.getByText("disabled", { exact: true })).toBeVisible();
  await card
    .getByRole("button", { name: "Delete browser-backend", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Delete resource", exact: true })
    .click();
  await expect(card).toHaveCount(0);
});

test("manages a budgeted API key without revealing its token", async ({
  page,
  request,
}) => {
  await connect(page);
  await page.getByRole("button", { name: "API keys", exact: true }).click();
  await page.getByRole("button", { name: "Add API key" }).click();
  await page.getByLabel("Key ID", { exact: true }).fill("browser-key");
  await page
    .getByLabel("API token", { exact: true })
    .fill("browser-client-secret");
  await page
    .getByLabel("Budget (JSON)", { exact: true })
    .fill('{"period":"day","limit_usd":12}');
  await page.getByRole("button", { name: "Save changes" }).click();
  const card = page.locator(".resource-card").filter({
    has: page.getByRole("heading", { name: "browser-key", exact: true }),
  });
  await expect(card).toContainText("$12.00 / day");
  expect(await page.locator("body").innerText()).not.toContain(
    "browser-client-secret",
  );
  const usage = await request.get("v1/usage", {
    headers: { Authorization: "Bearer browser-client-secret" },
  });
  expect((await usage.json()).key.limit_usd).toBe(12);
  await card
    .getByRole("button", { name: "Delete browser-key", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Delete resource", exact: true })
    .click();
  await expect(card).toHaveCount(0);
  expect(
    (
      await request.get("v1/usage", {
        headers: { Authorization: "Bearer browser-client-secret" },
      })
    ).status(),
  ).toBe(401);
});

test("manages proxy pools and applies routing policy", async ({
  page,
  request,
}) => {
  await connect(page);
  await page.getByRole("button", { name: "Proxies", exact: true }).click();
  await page.getByRole("button", { name: "Add proxy" }).click();
  await page
    .getByLabel("Upstream base URL", { exact: true })
    .fill("https://browser.example.test/v1");
  await page
    .getByLabel("Account pool (JSON)", { exact: true })
    .fill('[{"id":"primary","headers":{"X-API-Key":"private-proxy-key"}}]');
  await page.getByRole("button", { name: "Save changes" }).click();
  const card = page.locator(".resource-card").filter({
    has: page.getByRole("heading", {
      name: "browser.example.test",
      exact: true,
    }),
  });
  await expect(card).toContainText("active");
  await card
    .getByRole("button", { name: "Delete browser.example.test", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Delete resource", exact: true })
    .click();
  await expect(card).toHaveCount(0);
  await page.getByRole("button", { name: "Routing", exact: true }).click();
  await page.getByRole("radio", { name: "Cost", exact: true }).check();
  await page
    .getByLabel("Routing profiles (JSON)", { exact: true })
    .fill('{"fast":{"optimize":"latency","tags":["test"]}}');
  await page.getByRole("button", { name: "Save routing", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Changes applied");
  const response = await request.get("v1/management/config", {
    headers: { Authorization: "Bearer management-test" },
  });
  const config = (await response.json()).config;
  expect(config.routing_default_optimize).toBe("cost");
  expect(config.routing_profiles.fast.optimize).toBe("latency");
});

test("shows a revision conflict and lets the operator reload", async ({
  page,
  request,
}) => {
  await connect(page);
  await page.getByRole("button", { name: "API keys", exact: true }).click();
  const admin = { Authorization: "Bearer management-test" };
  const old = await request.get("v1/management/config", {
    headers: admin,
  });
  await request.put("v1/management/config", {
    headers: { ...admin, "If-Match": old.headers().etag },
    data: (await old.json()).config,
  });
  await page.getByRole("button", { name: "Add API key" }).click();
  await page.getByLabel("Key ID", { exact: true }).fill("conflict-key");
  await page.getByLabel("API token", { exact: true }).fill("conflict-token");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByRole("dialog").getByRole("alert")).toContainText(
    "Configuration changed",
  );
  await page.getByRole("button", { name: "Close editor" }).click();
  await page.getByRole("button", { name: "Reload configuration" }).click();
  await expect(page.getByRole("status")).toContainText(
    "Configuration reloaded",
  );
});

test("renders a usable dashboard on desktop and mobile", async ({ page }) => {
  await connect(page);
  await page.screenshot({
    path: "test-results/overview-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Overview", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/overview-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "API keys", exact: true }).click();
  await page.getByRole("button", { name: "Add API key" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "Close editor" }).click();
});
