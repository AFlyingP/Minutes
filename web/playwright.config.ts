import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 30000,
  retries: 0,
  workers: 1,
  reporter: "line",
  use: { baseURL: "http://127.0.0.1:8000", browserName: "chromium" },
  webServer: {
    command: "uv run minutes serve --corpus fixture",
    cwd: "..",
    url: "http://127.0.0.1:8000/api/health",
    timeout: 60000,
    reuseExistingServer: false,
    env: { MINUTES_LLM_MODE: "stub", MINUTES_MODELS_MODE: "stub", MINUTES_CORPUS: "fixture" },
  },
});
