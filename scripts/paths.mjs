import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const WEB = resolve(ROOT, 'web');
export const CACHE = resolve(ROOT, '.cache');
export const ARTIFACTS = resolve(ROOT, 'artifacts');
