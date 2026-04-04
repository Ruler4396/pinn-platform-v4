import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/pinn-flow-visual-demo-v4/',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 4176
  },
  preview: {
    host: '0.0.0.0',
    port: 4176
  }
});
