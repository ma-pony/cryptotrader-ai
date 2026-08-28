import { describe, expect, it } from 'vitest';

import { buildApiUrl } from './api-url';

describe('buildApiUrl', () => {
  it('keeps API requests same-origin when the configured base is empty', () => {
    expect(buildApiUrl('/api/signal-profile', '')).toBe('/api/signal-profile');
    expect(buildApiUrl('api/chat/stream', '')).toBe('/api/chat/stream');
  });

  it('joins API paths to an explicit absolute deployment override', () => {
    expect(buildApiUrl('/api/signal-profile', 'https://api.example.com/')).toBe(
      'https://api.example.com/api/signal-profile',
    );
  });

  it('leaves an already absolute request URL unchanged', () => {
    expect(buildApiUrl('https://files.example.com/report.json', '')).toBe(
      'https://files.example.com/report.json',
    );
  });

  it('normalizes repeated leading slashes without creating a protocol-relative URL', () => {
    expect(buildApiUrl('//api/chat/stream', '')).toBe('/api/chat/stream');
    expect(buildApiUrl('///api/chat/stream', 'https://api.example.com/')).toBe(
      'https://api.example.com/api/chat/stream',
    );
  });
});
