import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist" },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.tsx"],
    testTimeout: 10000,
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/**/*.test.tsx"],
      thresholds: { lines: 70 },
    },
  },
});
