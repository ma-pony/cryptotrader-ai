const ABSOLUTE_HTTP_URL = /^https?:\/\//i;

export function buildApiUrl(path: string, base: string): string {
  if (ABSOLUTE_HTTP_URL.test(path)) return path;

  const normalizedBase = base.replace(/\/+$/, '');
  const normalizedPath = path.replace(/^\/+/, '');
  return `${normalizedBase}/${normalizedPath}`;
}
