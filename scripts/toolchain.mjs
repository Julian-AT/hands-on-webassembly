import { createHash } from 'node:crypto';
import {
  readFile,
  writeFile,
  mkdir,
  readdir,
  access,
  rm,
  statfs,
  rename,
} from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pins = JSON.parse(
  await readFile(resolve(root, 'manifests/toolchain.json'), 'utf8'),
);
const key = `${process.platform}-${process.arch}`;
const checksum = pins.node.archives[key];
if (!checksum) throw new Error(`Unsupported toolchain platform ${key}`);
const disk = await statfs(root);
const reserve = 15 * 1024 ** 3,
  estimate = 3 * 1024 ** 3;
if (disk.bavail * disk.bsize < reserve + estimate)
  throw new Error(
    'Insufficient space: keep 15 GiB reserve plus 3 GiB next-run footprint',
  );
const cache = resolve(root, '.cache/toolchain');
await mkdir(cache, { recursive: true });
async function download(url, path, algorithm, digest, encoding = 'hex') {
  let data;
  let fetched = false;
  try {
    data = await readFile(path);
  } catch {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Toolchain HTTP ${response.status}`);
    data = Buffer.from(await response.arrayBuffer());
    fetched = true;
  }
  if (createHash(algorithm).update(data).digest(encoding) !== digest)
    throw new Error(`Toolchain checksum rejected: ${url}`);
  if (fetched) {
    const temporary = `${path}.${process.pid}.tmp`;
    await writeFile(temporary, data);
    await rename(temporary, path);
  }
}
function run(command, args, env = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, ...env },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}
const nodeName = `node-v${pins.node.version}-${key}`;
const nodeArchive = resolve(cache, `${nodeName}.tar.gz`),
  nodeDir = resolve(cache, nodeName);
await download(
  `https://nodejs.org/dist/v${pins.node.version}/${nodeName}.tar.gz`,
  nodeArchive,
  'sha256',
  checksum,
);
try {
  await access(resolve(nodeDir, 'bin/node'));
} catch {
  run('tar', ['-xzf', nodeArchive, '-C', cache]);
}
const node = resolve(nodeDir, 'bin/node');
if (
  spawnSync(node, ['--version'], { encoding: 'utf8' }).stdout.trim() !==
  `v${pins.node.version}`
)
  throw new Error('Wrong Node version');
const npmArchive = resolve(cache, `npm-${pins.npm.version}.tgz`),
  npmDir = resolve(cache, `npm-${pins.npm.version}`);
await download(
  `https://registry.npmjs.org/npm/-/npm-${pins.npm.version}.tgz`,
  npmArchive,
  'sha512',
  pins.npm.integrity.slice(7),
  'base64',
);
try {
  await access(resolve(npmDir, 'package/bin/npm-cli.js'));
} catch {
  await mkdir(npmDir, { recursive: true });
  run('tar', ['-xzf', npmArchive, '-C', npmDir]);
}
const npm = resolve(npmDir, 'package/bin/npm-cli.js');
const env = {
  PATH: `${resolve(nodeDir, 'bin')}:${process.env.PATH}`,
  npm_config_cache: resolve(cache, 'npm-cache'),
  PIP_CACHE_DIR: resolve(root, '.cache/pip'),
  NEXT_TELEMETRY_DISABLED: '1',
};
if (
  spawnSync(node, [npm, '--version'], {
    encoding: 'utf8',
    env: { ...process.env, ...env },
  }).stdout.trim() !== pins.npm.version
)
  throw new Error('Wrong npm version');
const mode = process.argv[2];
if (mode === 'install') run(node, [npm, 'ci', '--no-audit', '--no-fund'], env);
else if (mode === 'lock')
  run(
    node,
    [
      npm,
      'install',
      '--package-lock-only',
      '--ignore-scripts',
      '--no-audit',
      '--no-fund',
    ],
    env,
  );
else if (mode === 'build') run(node, ['scripts/web/build.mjs'], env);
else if (mode === 'test') {
  run(
    resolve(root, '.cache/env/native/bin/python'),
    ['-m', 'unittest', 'discover', '-s', 'tests/python'],
    env,
  );
  run(
    node,
    [
      '--test',
      ...(await readdir(resolve(root, 'tests/javascript')))
        .filter((name) => name.endsWith('.test.mjs'))
        .map((name) => `tests/javascript/${name}`),
    ],
    env,
  );
} else if (mode === 'setup') {
  run(node, [npm, 'ci', '--no-audit', '--no-fund'], env);
  run(node, ['scripts/setup.mjs'], env);
} else if (mode === 'check') {
  run(node, [npm, 'run', 'format:check'], env);
  run(node, [npm, 'run', 'lint'], env);
  run(resolve(root, '.cache/env/tools/bin/ruff'), ['check', '.'], env);
  run(
    resolve(root, '.cache/env/tools/bin/ruff'),
    ['format', '--check', '.'],
    env,
  );
  run(node, ['scripts/quality/hygiene.mjs'], env);
} else if (mode === 'preview')
  run(node, ['scripts/web/preview.mjs', 'web/out'], env);
else if (mode === 'proof') {
  run(
    resolve(root, '.cache/env/native/bin/python'),
    ['scripts/proof/inventory.py'],
    env,
  );
  run(
    resolve(root, '.cache/env/native/bin/python'),
    ['scripts/proof/gate.py'],
    env,
  );
} else if (mode === 'archive-verify')
  run(
    resolve(root, '.cache/env/native/bin/python'),
    ['scripts/archive/history.py', 'download'],
    env,
  );
else if (mode === 'staged')
  run(
    node,
    [resolve(root, 'node_modules/lint-staged/bin/lint-staged.js')],
    env,
  );
else if (mode === 'commit-msg')
  run(
    node,
    [
      resolve(root, 'node_modules/@commitlint/cli/cli.js'),
      '--edit',
      process.argv[3],
    ],
    env,
  );
else if (mode === 'format') {
  run(node, [npm, 'run', 'format'], env);
  run(resolve(root, '.cache/env/tools/bin/ruff'), ['format', '.'], env);
} else if (mode === 'commit-check')
  run(node, ['scripts/quality/commits.mjs'], env);
else throw new Error('Unknown toolchain command');
