import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from './paths.mjs';
function run(command, args) {
  const r = spawnSync(command, args, { cwd: ROOT, stdio: 'inherit' });
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status || 1);
}
mkdirSync(resolve(ROOT, '.cache/env'), { recursive: true });
const python = process.env.PYTHON || 'python3.12';
for (const [environment, lock] of [
  ['native', 'native'],
  ['tools', 'development'],
]) {
  const folder = resolve(ROOT, `.cache/env/${environment}`);
  const executable = resolve(folder, 'bin/python');
  if (!existsSync(executable)) run(python, ['-m', 'venv', folder]);
  run(executable, [
    '-c',
    "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'",
  ]);
  run(executable, [
    '-m',
    'pip',
    'install',
    '--disable-pip-version-check',
    '-r',
    `requirements/${lock}.lock`,
  ]);
}

import { prepareAssets } from './web/assets.mjs';
import { symlinkSync, rmSync } from 'node:fs';
await prepareAssets(ROOT);
mkdirSync(resolve(ROOT, 'artifacts'), { recursive: true });
const assets = resolve(ROOT, 'artifacts/assets');
if (!existsSync(assets))
  symlinkSync(resolve(ROOT, 'web/public/assets'), assets, 'dir');
run(resolve(ROOT, '.cache/env/native/bin/python'), [
  'scripts/archive/originals.py',
]);
run(resolve(ROOT, '.cache/env/native/bin/python'), [
  'scripts/quality/prepare_tests.py',
]);
