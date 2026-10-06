import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtemp,
  mkdir,
  writeFile,
  readFile,
  cp,
  symlink,
  rm,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import staged, { editable } from '../../lint-staged.config.mjs';
import { ROOT } from '../../scripts/paths.mjs';

test('staged checks exclude originals, immutable contracts, vendors and generated files', () => {
  for (const name of [
    'assignments/1/app.py',
    'runtime/session-bridge.js',
    'wasm/packages/native_tsne/a.py',
    'manifests/startup-ui.json',
    'proof/contracts/legacy-requirements.json',
    'proof/locks/runtime-packages/manifest.json',
    'tests/fixtures/upstream.js',
    'proof/source.lock.json',
    'proof/hardware-contract.json',
    'requirements/native.lock',
    '.cache/input.js',
    'artifacts/proof/report.json',
    'web/public/app.js',
    'web/.next/file.js',
    'web/out/index.html',
  ])
    assert.equal(editable(resolve(ROOT, name)), false, name);
  assert.deepEqual(staged([resolve(ROOT, 'assignments/1/app.py')]), []);
  assert.ok(
    staged([resolve(ROOT, 'scripts/quality/check.py')]).some((command) =>
      command.includes('ruff check'),
    ),
  );
});

test('real Husky hooks reject invalid messages and malformed staged code, accept a valid commit', async () => {
  const root = await mkdtemp(join(tmpdir(), 'course-hooks-'));
  const git = (args) =>
    spawnSync('git', args, {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, HUSKY: '1' },
    });
  try {
    for (const folder of ['scripts/quality', 'manifests', '.husky'])
      await mkdir(join(root, folder), { recursive: true });
    for (const name of [
      'scripts/toolchain.mjs',
      'scripts/paths.mjs',
      'scripts/quality/hygiene.mjs',
      'manifests/toolchain.json',
      'lint-staged.config.mjs',
      'commitlint.config.mjs',
      'eslint.config.mjs',
      'prettier.config.mjs',
      '.prettierignore',
      '.gitignore',
      '.husky/pre-commit',
      '.husky/commit-msg',
    ])
      await cp(join(ROOT, name), join(root, name));
    await symlink(
      join(ROOT, 'node_modules'),
      join(root, 'node_modules'),
      'dir',
    );
    await symlink(join(ROOT, '.cache'), join(root, '.cache'), 'dir');
    assert.equal(git(['init', '-q']).status, 0);
    git(['config', 'user.name', 'Hook Test']);
    git(['config', 'user.email', 'test@example.invalid']);
    const husky = spawnSync(
      process.execPath,
      [join(ROOT, 'node_modules/husky/bin.js')],
      { cwd: root, encoding: 'utf8' },
    );
    assert.equal(husky.status, 0, husky.stderr);
    await writeFile(join(root, 'example.mjs'), 'export const value = 1;\n');
    git(['add', 'example.mjs']);
    const bad = git(['commit', '-m', 'invalid message']);
    assert.notEqual(bad.status, 0);
    assert.match(bad.stdout + bad.stderr, /subject-empty|type-empty/);
    await writeFile(join(root, 'example.mjs'), 'export const = ;\n');
    git(['add', 'example.mjs']);
    const malformed = git(['commit', '-m', 'test: enforce staged checks']);
    assert.notEqual(malformed.status, 0);
    assert.match(
      malformed.stdout + malformed.stderr,
      /SyntaxError|Parsing error/,
    );
    await writeFile(join(root, 'example.mjs'), 'export const value = 1;\n');
    git(['add', 'example.mjs']);
    const valid = git(['commit', '-m', 'test: verify hooks']);
    assert.equal(valid.status, 0, valid.stdout + valid.stderr);
    await mkdir(join(root, 'web/out'), { recursive: true });
    await writeFile(join(root, 'web/out/generated.js'), 'generated');
    git(['add', '-f', 'web/out/generated.js']);
    const generated = git(['commit', '-m', 'build: accidentally stage output']);
    assert.notEqual(generated.status, 0);
    assert.match(
      generated.stdout + generated.stderr,
      /Generated or obsolete file/,
    );
    assert.equal(
      await readFile(join(root, 'example.mjs'), 'utf8'),
      'export const value = 1;\n',
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
