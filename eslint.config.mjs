import next from 'eslint-config-next/core-web-vitals';
export default [
  {
    ignores: [
      '.cache/**',
      'artifacts/**',
      'node_modules/**',
      'web/.next/**',
      'web/out/**',
      'web/public/**',
      'assignments/**',
      'wasm/**',
      'runtime/**',
      'proof/harness/**',
      'tests/fixtures/**',
    ],
  },
  ...next.map((config) => ({ ...config, files: ['web/**/*.{js,jsx,mjs}'] })),
  {
    files: ['**/*.{js,mjs}'],
    languageOptions: { ecmaVersion: 'latest', sourceType: 'module' },
    rules: {
      'no-constant-condition': 'error',
      'no-dupe-keys': 'error',
      'no-unreachable': 'error',
    },
  },
  { settings: { next: { rootDir: 'web/' } } },
];
