import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import {
  safePath,
  extractArchive,
  prepareAssets,
} from '../../scripts/web/assets.mjs';
function tar(entries) {
  const blocks = [];
  for (const { name, data = 'test', type = '0' } of entries) {
    const bytes = Buffer.from(data),
      header = Buffer.alloc(512);
    header.write(name, 0, 100);
    header.write('0000644\0', 100);
    header.write('0000000\0', 108);
    header.write('0000000\0', 116);
    header.write(bytes.length.toString(8).padStart(11, '0') + '\0', 124);
    header.write('00000000000\0', 136);
    header.fill(32, 148, 156);
    header.write(type, 156);
    header.write('ustar\0', 257);
    header.write(
      header
        .reduce((s, x) => s + x, 0)
        .toString(8)
        .padStart(6, '0') + '\0 ',
      148,
    );
    blocks.push(
      header,
      bytes,
      Buffer.alloc((512 - (bytes.length % 512)) % 512),
    );
  }
  blocks.push(Buffer.alloc(1024));
  return gzipSync(Buffer.concat(blocks));
}
const manifest = {
  files: {
    'assets/data.txt': {
      bytes: 4,
      sha256: createHash('sha256').update('test').digest('hex'),
    },
  },
};
test('safe paths reject traversal and noncanonical entries', () => {
  for (const name of [
    '../escape',
    '/escape',
    'a/../escape',
    'a\\b',
    'a//b',
    './a',
    'a/',
    'a\0b',
  ])
    assert.throws(() => safePath(name));
  assert.equal(safePath('assets/data.txt'), 'assets/data.txt');
});
test('complete verified archive extracts exact file values', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'course-archive-'));
  try {
    const file = join(dir, 'asset.tgz');
    await writeFile(file, tar([{ name: 'assets/data.txt' }]));
    assert.equal(await extractArchive(file, join(dir, 'stage'), manifest), 1);
    assert.equal(
      await readFile(join(dir, 'stage/assets/data.txt'), 'utf8'),
      'test',
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test('rejects missing, duplicate, unexpected, traversal, link, corrupt and wrong-size entries', async () => {
  const fixtures = [
    [],
    [{ name: 'assets/data.txt' }, { name: 'assets/data.txt' }],
    [{ name: 'extra.txt' }],
    [{ name: '../escape' }],
    [{ name: 'assets/data.txt', type: '2' }],
    [{ name: 'assets/data.txt', data: 'fail' }],
    [{ name: 'assets/data.txt', data: 'wrong size' }],
  ];
  for (const entries of fixtures) {
    const dir = await mkdtemp(join(tmpdir(), 'course-archive-'));
    try {
      const file = join(dir, 'asset.tgz');
      await writeFile(file, tar(entries));
      await assert.rejects(extractArchive(file, join(dir, 'stage'), manifest));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

test('publishes the approved startup UI change while preserving the other application bytes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'course-loading-ui-'));
  try {
    const input =
      '<div id="root"></div><section id="course-startup" role="status"><p>Loading</p><button hidden>Reload to retry</button></section>';
    const expected =
      '<div id="root"></div><section id="course-startup" hidden role="status"><p>Loading</p><button hidden>Reload to retry</button></section>';
    const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
    const files = {},
      hashes = {},
      overrides = {},
      entries = [];
    for (let unit = 1; unit <= 7; unit++) {
      const name = `unit${unit}/index.html`;
      files[name] = { bytes: Buffer.byteLength(input), sha256: hash(input) };
      hashes[name] = hash(input);
      overrides[name] = {
        original_sha256: hash(input),
        bytes: Buffer.byteLength(expected),
        sha256: hash(expected),
      };
      entries.push({ name, data: input });
    }
    files['assets/data.txt'] = manifest.files['assets/data.txt'];
    hashes['assets/data.txt'] = manifest.files['assets/data.txt'].sha256;
    entries.push({ name: 'assets/data.txt' });
    const archive = tar(entries),
      hashesText = JSON.stringify(hashes),
      manifestHash = hash(hashesText);
    await mkdir(join(root, 'manifests'));
    await mkdir(join(root, '.cache/assets'), { recursive: true });
    await writeFile(
      join(root, 'manifests/application-files.json'),
      JSON.stringify({ files, file_count: 8, manifest_sha256: manifestHash }),
    );
    await writeFile(
      join(root, 'manifests/application-hashes.json'),
      hashesText,
    );
    await writeFile(
      join(root, 'manifests/assets-release.json'),
      JSON.stringify({
        sha256: hash(archive),
        bytes: archive.length,
        manifest_sha256: manifestHash,
      }),
    );
    await writeFile(
      join(root, 'manifests/startup-ui.json'),
      JSON.stringify({ files: overrides }),
    );
    await writeFile(
      join(root, '.cache/assets', `assets-${hash(archive)}.tar.gz`),
      archive,
    );
    await prepareAssets(root);
    for (let unit = 1; unit <= 7; unit++)
      assert.equal(
        await readFile(join(root, `web/public/unit${unit}/index.html`), 'utf8'),
        expected,
      );
    assert.equal(
      await readFile(join(root, 'web/public/assets/data.txt'), 'utf8'),
      'test',
    );
    await writeFile(
      join(root, 'web/public', 'previous-publication.txt'),
      'keep this publication',
    );
    overrides['unit7/index.html'].sha256 = '0'.repeat(64);
    await writeFile(
      join(root, 'manifests/startup-ui.json'),
      JSON.stringify({ files: overrides }),
    );
    await assert.rejects(
      prepareAssets(root),
      /Startup UI output binding rejected/,
    );
    assert.equal(
      await readFile(
        join(root, 'web/public', 'previous-publication.txt'),
        'utf8',
      ),
      'keep this publication',
    );
    assert.equal(
      await readFile(join(root, 'web/public/unit1/index.html'), 'utf8'),
      expected,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
