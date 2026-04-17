import { useRef, useEffect, type FC } from 'react';

interface InlineWidgetProps {
  html: string;
  heightPx?: number | undefined;
}

const SANDBOX_FLAGS = 'allow-scripts';

export const InlineWidget: FC<InlineWidgetProps> = ({ html, heightPx = 200 }) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;
    const doc = iframe.contentDocument;
    if (!doc) return;
    doc.open();
    doc.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;font-family:system-ui,sans-serif;color:#e5e7eb;background:transparent;}</style></head><body>${html}</body></html>`);
    doc.close();
  }, [html]);

  return (
    <iframe
      ref={iframeRef}
      sandbox={SANDBOX_FLAGS}
      title="Inline widget"
      className="w-full rounded border border-border bg-transparent"
      style={{ height: `${String(heightPx)}px` }}
    />
  );
};
