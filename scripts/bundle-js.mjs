#!/usr/bin/env node
/**
 * Kawkab AI frontend bundler.
 *
 * Reads the load order from scripts/frontend-manifest.json (NOT from
 * index.html -- see CLAUDE.md "Frontend build" section for why), bundles
 * those IIFE files in order with esbuild, and writes
 * src/kawkab/web/dist/app.bundle.min.js (+ .map).
 *
 * Does NOT touch index.html. index.html's <script> tags (the bundle tag,
 * qwebchannel.js, app-3d.js as a module, app-briefing.js -- wait, no:
 * app-briefing.js is in the manifest, not a separate tag) are the
 * permanent, hand-maintained source of truth for what loads outside the
 * bundle. A previous version of this script derived its file list BY
 * PARSING those tags out of index.html, then DELETED them from
 * index.html after bundling -- which meant a second run had nothing left
 * to parse and would have silently emitted a near-empty bundle. Keeping
 * the manifest and index.html independently maintained (each an explicit,
 * versioned file) makes every run idempotent: run this as many times as
 * you want, index.html never changes.
 *
 * Usage: node scripts/bundle-js.mjs
 */
import * as esbuild from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(__dirname, '../src/kawkab/web');
const manifestPath = resolve(__dirname, 'frontend-manifest.json');
const distDir = resolve(webDir, 'dist');

const manifest = JSON.parse(await readFile(manifestPath, 'utf-8'));
const scriptSrcs = manifest.bundle;
if (!Array.isArray(scriptSrcs) || scriptSrcs.length === 0) {
  throw new Error(`${manifestPath} has no "bundle" array -- nothing to bundle.`);
}

console.log(`Bundling ${scriptSrcs.length} files from frontend-manifest.json (in order):`);

let raw = '';
for (const src of scriptSrcs) {
  const filePath = resolve(webDir, src);
  let content;
  try {
    content = await readFile(filePath, 'utf-8');
  } catch (e) {
    throw new Error(`Manifest references ${src}, but it does not exist at ${filePath}: ${e.message}`);
  }
  content = content.replace(/\/\/# sourceMappingURL.*/g, '');
  raw += `// ${src}\n${content}\n\n`;
  console.log(`  + ${src} (${(content.length / 1024).toFixed(1)} KB)`);
}

await mkdir(distDir, { recursive: true });

const result = await esbuild.transform(raw, {
  minify: true,
  target: 'es2020',
  sourcemap: true,
});

const bundlePath = resolve(distDir, 'app.bundle.min.js');
await writeFile(bundlePath, result.code);
if (result.map) {
  // esbuild's transform() already returns `map` as sourcemap JSON *text*
  // (not an object) when sourcemap:true is passed -- JSON.stringify()-ing
  // it again here would double-encode it into a JSON string containing an
  // escaped JSON string, which is unreadable by source-map tooling. Write
  // it as-is.
  await writeFile(resolve(distDir, 'app.bundle.min.js.map'), result.map);
}

const savedBytes = Buffer.byteLength(raw) - result.code.length;
const savedPct = ((1 - result.code.length / Buffer.byteLength(raw)) * 100).toFixed(1);
console.log(`\n✓ Bundle written to dist/app.bundle.min.js`);
console.log(`  ${scriptSrcs.length} files concatenated`);
console.log(`  ${Buffer.byteLength(raw).toLocaleString()} B → ${result.code.length.toLocaleString()} B (${savedPct}% reduction, saved ${savedBytes.toLocaleString()} B)`);
console.log(`index.html is unchanged -- it already references dist/app.bundle.min.js directly.`);
