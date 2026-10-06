import { execFileSync, spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { ROOT } from '../paths.mjs';
const commits = execFileSync('git', ['rev-list', 'HEAD'], {
  cwd: ROOT,
  encoding: 'utf8',
})
  .trim()
  .split('\n');
for (const commit of commits) {
  const message = execFileSync('git', ['show', '-s', '--format=%B', commit], {
    cwd: ROOT,
    encoding: 'utf8',
  });
  const result = spawnSync(
    process.execPath,
    [resolve(ROOT, 'node_modules/@commitlint/cli/cli.js')],
    { cwd: ROOT, input: message, encoding: 'utf8' },
  );
  if (result.status !== 0) {
    process.stderr.write(result.stdout + result.stderr);
    process.exit(result.status || 1);
  }
}
console.log(`Conventional Commits: ${commits.length} messages checked`);
