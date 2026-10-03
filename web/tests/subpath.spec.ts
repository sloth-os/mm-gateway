import { test, expect } from "@playwright/test";

test("keeps assets, SDK traffic and API docs under the proxy prefix", async ({
  page,
  request,
  baseURL,
}) => {
  const requests: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin === new URL(baseURL!).origin) requests.push(url.pathname);
  });
  expect((await request.get("/admin/")).status()).toBe(404);
  await page.goto("admin/?from=proxy#overview");
  await expect(page.getByLabel("Management token")).toBeVisible();
  expect(
    await page.evaluate(() => document.styleSheets.length),
  ).toBeGreaterThan(0);
  const favicon = await page
    .locator('link[rel="icon"]')
    .evaluate((link) => (link as HTMLLinkElement).href);
  expect((await request.get(favicon)).headers()["content-type"]).toContain(
    "image/svg+xml",
  );
  await page.getByLabel("Management token").fill("management-test");
  await page.getByRole("button", { name: "Connect to gateway" }).click();
  await expect(
    page.getByRole("heading", { name: "Overview", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "API documentation" }),
  ).toHaveAttribute("href", `${baseURL}docs`);
  const docs = await request.get("docs");
  expect(docs.status()).toBe(200);
  expect(await docs.text()).toContain("/nested/gateway/openapi.json");
  const schema = await request.get("openapi.json");
  expect(schema.status()).toBe(200);
  expect((await schema.json()).servers).toContainEqual({
    url: "/nested/gateway",
  });
  expect(
    requests.some(
      (path) => path.includes("/admin/assets/") && path.endsWith(".js"),
    ),
  ).toBe(true);
  expect(requests.some((path) => path.includes("/v1/management/status"))).toBe(
    true,
  );
  expect(requests.every((path) => path.startsWith("/nested/gateway/"))).toBe(
    true,
  );
});

test("preserves the prefix and query on the admin trailing-slash redirect", async ({
  page,
}) => {
  await page.goto("admin?from=redirect");
  await expect(page).toHaveURL(
    "http://127.0.0.1:8766/nested/gateway/admin/?from=redirect",
  );
  await expect(page.getByLabel("Management token")).toBeVisible();
  await page.goto("http://127.0.0.1:8766/nested/gateway");
  await expect(page).toHaveURL("http://127.0.0.1:8766/nested/gateway/admin/");
});
