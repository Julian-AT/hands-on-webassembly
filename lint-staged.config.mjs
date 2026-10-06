import { relative } from 'node:path';
const protectedPaths =
  /^(assignments|runtime|wasm|manifests|requirements|proof\/(harness|contracts|locks)|tests\/fixtures|artifacts|\.cache)\//;
export function editable(file) {
  const name = relative(process.cwd(), file).replaceAll('\\', '/');
  return (
    !protectedPaths.test(name) &&
    !name.startsWith('web/public/') &&
    !name.startsWith('web/out/') &&
    !name.startsWith('web/.next/') &&
    !name.endsWith('.lock.json') &&
    name !== 'proof/hardware-contract.json' &&
    name !== 'iml_env.yaml' &&
    name !== 'package-lock.json'
  );
}
export default (files) => {
  const selected = files.filter(editable);
  const quote = (name) => "'" + name.replaceAll("'", "'\\''") + "'";
  const commands = [];
  const py = selected.filter((name) => name.endsWith('.py'));
  const js = selected.filter((name) => /\.(mjs|js|jsx)$/.test(name));
  const formatted = selected.filter((name) =>
    /\.(mjs|js|jsx|json|md|css|yml|yaml)$/.test(name),
  );
  if (py.length)
    commands.push(
      `.cache/env/tools/bin/ruff check ${py.map(quote).join(' ')}`,
      `.cache/env/tools/bin/ruff format ${py.map(quote).join(' ')}`,
    );
  if (formatted.length)
    commands.push(`prettier --write ${formatted.map(quote).join(' ')}`);
  if (js.length) commands.push(`eslint ${js.map(quote).join(' ')}`);
  return commands;
};
