import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const webRoot = path.resolve(import.meta.dirname, '../..');
const node = '/Users/rccpony/.nvm/versions/node/v24.19.0/bin/node';
const playwright = path.join(webRoot, 'node_modules', '@playwright', 'test', 'cli.js');

function filesUnder(root: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(root).sort()) {
    const absolute = path.join(root, entry);
    if (statSync(absolute).isDirectory()) files.push(...filesUnder(absolute));
    else files.push(absolute);
  }
  return files;
}

function hashTree(relativeRoot: string) {
  const root = path.join(webRoot, relativeRoot);
  const hash = createHash('sha256');
  for (const file of filesUnder(root)) {
    hash.update(path.relative(root, file));
    hash.update('\0');
    hash.update(readFileSync(file));
    hash.update('\0');
  }
  return hash.digest('hex');
}

function hashFrontendSource() {
  const hash = createHash('sha256');
  for (const relativeRoot of ['src', 'index.html']) {
    const absolute = path.join(webRoot, relativeRoot);
    const files = statSync(absolute).isDirectory() ? filesUnder(absolute) : [absolute];
    for (const file of files) {
      hash.update(path.relative(webRoot, file));
      hash.update('\0');
      hash.update(readFileSync(file));
      hash.update('\0');
    }
  }
  return hash.digest('hex');
}

function snapshot() {
  return { source: hashFrontendSource(), dist: hashTree('dist') };
}

function runVariant(variant: 'baseline' | 'extension') {
  const result = spawnSync(node, [playwright, 'test', '--config', 'playwright.workbench.config.ts'], {
    cwd: webRoot,
    env: { ...process.env, WORKBENCH_REGISTRY_VARIANT: variant },
    stdio: 'inherit',
  });
  if (result.status !== 0)
    throw new Error(`${variant} registry Playwright run failed with exit ${String(result.status)}`);
}

const before = snapshot();
runVariant('baseline');
const afterBaseline = snapshot();
runVariant('extension');
const afterExtension = snapshot();

if (
  JSON.stringify(before) !== JSON.stringify(afterBaseline) ||
  JSON.stringify(before) !== JSON.stringify(afterExtension)
) {
  throw new Error(
    `frontend source or dist changed across registry runs: ${JSON.stringify({ before, afterBaseline, afterExtension })}`,
  );
}
console.log(JSON.stringify({ sameFrontendBuild: true, hashes: before }));
