export function selection(url) {
  const parsed = new URL(url);
  const values = parsed.searchParams.getAll("unit");
  const valid = values.length === 1 && /^[1-7]$/.test(values[0]);
  const unit = valid ? Number(values[0]) : 1;
  const normalize = values.length > 0 && !valid;
  if (normalize) parsed.searchParams.set("unit", "1");
  return {unit, normalize, url: parsed};
}
export function selectionUrl(url, unit) {
  if (!Number.isInteger(unit) || unit < 1 || unit > 7) throw new Error("Invalid assignment");
  const parsed = new URL(url);
  parsed.searchParams.set("unit", String(unit));
  return parsed;
}
