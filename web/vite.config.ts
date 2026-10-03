import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "serve" ? "/admin/" : "./",
  build: { outDir: "../mm_gateway/server/static/admin", emptyOutDir: true },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1": {
        target: process.env.GATEWAY_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/metrics": {
        target: process.env.GATEWAY_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: process.env.GATEWAY_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
}));
