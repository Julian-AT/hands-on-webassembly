import { createHash } from 'node:crypto';
import { readFile, readdir, stat, writeFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { prepareAssets, hashFile } from './assets.mjs';
import { ROOT as root, WEB as web } from '../paths.mjs';
const inputs = [
  'web/app',
  'web/lib',
  'runtime/launcher-session.mjs',
  'scripts/web',
  'scripts/paths.mjs',
  'manifests',
  'package.json',
  'package-lock.json',
  'web/package.json',
  'web/next.config.mjs',
  'vercel.json',
];
const locked = {};
async function visit(path) {
  const file = resolve(root, path);
  if ((await stat(file)).isDirectory()) {
    for (const name of (await readdir(file)).sort())
      await visit(`${path}/${name}`);
  } else locked[path] = await hashFile(file);
}
for (const path of inputs) await visit(path);
const identity = createHash('sha256')
  .update(
    JSON.stringify(
      Object.fromEntries(
        Object.entries(locked).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
      ),
    ),
  )
  .digest('hex');
await prepareAssets(root);
await rm(resolve(web, '.next'), { recursive: true, force: true });
await rm(resolve(web, 'out'), { recursive: true, force: true });
const result = spawnSync(
  process.execPath,
  [resolve(root, 'node_modules/next/dist/bin/next'), 'build', '--webpack'],
  {
    cwd: web,
    stdio: 'inherit',
    env: {
      ...process.env,
      COURSE_BUILD_ID: identity,
      NEXT_TELEMETRY_DISABLED: '1',
    },
  },
);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status || 1);
const expected = JSON.parse(
  await readFile(resolve(root, 'manifests/application-files.json'), 'utf8'),
);
const startupUi = JSON.parse(
  await readFile(resolve(root, 'manifests/startup-ui.json'), 'utf8'),
);
const published = { ...expected.files, ...startupUi.files };
for (const [name, info] of Object.entries(published))
  if (
    (await stat(resolve(web, 'out', name))).size !== info.bytes ||
    (await hashFile(resolve(web, 'out', name))) !== info.sha256
  )
    throw new Error(`Export changed verified application asset ${name}`);
const files = {};
async function output(path = '') {
  for (const name of (await readdir(resolve(web, 'out', path))).sort()) {
    const key = path ? `${path}/${name}` : name;
    const file = resolve(web, 'out', key);
    const info = await stat(file);
    if (info.isDirectory()) await output(key);
    else files[key] = { bytes: info.size, sha256: await hashFile(file) };
  }
}
await output();
await writeFile(
  resolve(web, 'out/release-manifest.json'),
  JSON.stringify(
    {
      build_id: identity,
      application_build_id: JSON.parse(
        await readFile(resolve(web, 'lib/assignment-sessions.json'), 'utf8'),
      ).build_id,
      files,
    },
    null,
    2,
  ) + '\n',
);
console.log(
  `Static export ${identity}: ${Object.keys(files).length} files; no runtime functions configured`,
);
