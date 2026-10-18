import { defineConfig } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const production = process.env.MANAGEMENT_TEST_PRODUCTION === "1";
const managementConfigDirectory = mkdtempSync(join(tmpdir(), "mm-gateway-browser-"));

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  metadata: { managementConfigDirectory },
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 1000 },
  },
  projects: [
    {
      name: "root",
      testMatch: "**/console.spec.ts",
      use: {
        baseURL: production
          ? "http://127.0.0.1:8765/"
          : "http://127.0.0.1:5173/",
      },
    },
    ...(production
      ? [
          {
            name: "subpath",
            use: { baseURL: "http://127.0.0.1:8766/nested/gateway/" },
          },
        ]
      : []),
  ],
  webServer: [
    {
      command: `${process.env.GATEWAY_TEST_PYTHON || "../.venv/bin/python"} tests/gateway.py`,
      url: "http://127.0.0.1:8765/health",
      env: { MANAGEMENT_CONFIG_PATH: join(managementConfigDirectory, "root", "management.json") },
      reuseExistingServer: false,
    },
    ...(production
      ? [
          {
            command: `${process.env.GATEWAY_TEST_PYTHON || "../.venv/bin/python"} tests/gateway.py`,
            url: "http://127.0.0.1:8767/health",
            env: {
              GATEWAY_TEST_PORT: "8767",
              GATEWAY_TEST_ROOT_PATH: "/nested/gateway",
              MANAGEMENT_CONFIG_PATH: join(managementConfigDirectory, "subpath", "management.json"),
            },
            reuseExistingServer: false,
          },
          {
            command: process.env.CADDY_TEST_BINARY
              ? '"$CADDY_TEST_BINARY" run --config tests/Caddyfile'
              : "node tests/prefix-proxy.mjs",
            env: {
              XDG_CONFIG_HOME: join(tmpdir(), "mm-gateway-caddy-test", "config"),
              XDG_DATA_HOME: join(tmpdir(), "mm-gateway-caddy-test", "data"),
            },
            url: "http://127.0.0.1:8766/nested/gateway/health",
            reuseExistingServer: !process.env.CI,
          },
        ]
      : []),
    ...(!production
      ? [
          {
            command: "npm run dev",
            url: "http://127.0.0.1:5173/admin/",
            env: { GATEWAY_URL: "http://127.0.0.1:8765" },
            reuseExistingServer: !process.env.CI,
          },
        ]
      : []),
  ],
});
