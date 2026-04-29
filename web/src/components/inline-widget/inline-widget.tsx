import { useMemo, type FC } from 'react';

interface InlineWidgetProps {
  html: string;
  heightPx?: number | undefined;
}

// Deep-review FE-C1 (XSS hardening, 2026-04-24):
//
// Previous implementation used iframe.contentDocument.write(html) which bypasses
// the sandbox — the parent JS context writes the document before the browser
// applies the sandbox flags, so `allow-scripts` alone provided no origin
// isolation. Any compromised backend SSE payload could execute in the parent
// origin.
//
// Fixed by using the iframe's ``srcdoc`` attribute instead, which is subject to
// the sandbox from the moment the document loads. The sandbox is narrowed to
// ``allow-scripts`` only — NO ``allow-same-origin``, so even scripts inside the
// frame cannot touch the parent document, localStorage, cookies, or fetch with
// ambient credentials.
const SANDBOX_FLAGS = 'allow-scripts';

const IFRAME_BOILERPLATE_START =
  '<!DOCTYPE html><html><head><meta charset="utf-8">' +
  '<style>body{margin:0;font-family:system-ui,sans-serif;color:#e5e7eb;background:transparent;}</style>' +
  '</head><body>';
const IFRAME_BOILERPLATE_END = '</body></html>';

export const InlineWidget: FC<InlineWidgetProps> = ({ html, heightPx = 200 }) => {
  const srcDoc = useMemo(
    () => `${IFRAME_BOILERPLATE_START}${html}${IFRAME_BOILERPLATE_END}`,
    [html],
  );

  return (
    <iframe
      srcDoc={srcDoc}
      sandbox={SANDBOX_FLAGS}
      title="Inline widget"
      className="w-full rounded border border-border bg-transparent"
      style={{ height: `${String(heightPx)}px` }}
    />
  );
};
