import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solid()],
  server: {
    port: 5173,
    strictPort: true
  },
  build: {
    outDir: '../ui-solid-dist',
    emptyOutDir: true,
    target: 'es2020'
  }
});

