import { execFileSync } from 'node:child_process';
import { lstatSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from '../paths.mjs';
const staged = process.argv.includes('--staged');
const args = staged
  ? ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z']
  : ['ls-files', '--cached', '--others', '--exclude-standard', '-z'];
const files = execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' })
  .split('\0')
  .filter(Boolean);
const forbidden =
  /(^|\/)(\.DS_Store|__pycache__|node_modules|\.git)(\/|$)|^(artifacts|\.cache|release)\/|^web\/(out|public|\.next)\//;
for (const name of files) {
  if (forbidden.test(name))
    throw new Error(`Generated or obsolete file cannot be committed: ${name}`);
  const info = lstatSync(resolve(ROOT, name));
  if (info.isSymbolicLink())
    throw new Error(`Source tree must not contain symlinks: ${name}`);
  if (info.size > 5 * 1024 ** 2)
    throw new Error(
      `Use a verified release archive for files over 5 MiB: ${name}`,
    );
}
console.log(`Artifact hygiene: ${files.length} source files checked`);
