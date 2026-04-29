import type { VisualAnalysisResult } from '@/types/chart-analysis';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

interface AiAnalysisPanelProps {
  result: VisualAnalysisResult;
  onStop: () => void;
  onRetry: () => void;
}

export function AiAnalysisPanel({ result, onStop, onRetry }: AiAnalysisPanelProps) {
  const { t } = useTranslation('market');

  if (result.status === 'idle') return null;

  return (
    <div role="region" aria-label="AI Analysis" className="mt-4 rounded-lg border border-border bg-card p-4">
      {result.screenshotFailed && (
        <p className="mb-2 text-sm text-yellow-500">{t('ai_analysis.screenshot_failed')}</p>
      )}
      {result.contextNotice === 'image_too_large' && (
        <p className="mb-2 text-sm text-yellow-500">{t('ai_analysis.image_too_large')}</p>
      )}

      {result.status === 'loading' && (
        <div className="space-y-2">
          <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
          <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
          <div className="h-4 w-2/3 animate-pulse rounded bg-muted" />
        </div>
      )}

      {(result.status === 'streaming' || result.status === 'done') && (
        <div aria-live="polite" className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{result.contentMd}</ReactMarkdown>
        </div>
      )}

      {result.status === 'error' && (
        <div className="space-y-2">
          <p className="text-sm text-destructive">{t('ai_analysis.status_error')}</p>
          <button
            type="button"
            onClick={onRetry}
            className="text-sm text-primary underline"
          >
            {t('ai_analysis.fast_btn')}
          </button>
        </div>
      )}

      {result.status === 'streaming' && (
        <button
          type="button"
          onClick={onStop}
          className="mt-2 text-sm text-muted-foreground underline"
        >
          {t('ai_analysis.stop_btn')}
        </button>
      )}

      {result.status === 'done' && (
        <div className="mt-2 flex items-center gap-2">
          <span className="inline-block rounded bg-blue-500/20 px-2 py-0.5 text-xs text-blue-400">
            {result.screenshotFailed ? t('ai_analysis.source_numerical') : t('ai_analysis.source_visual')}
          </span>
        </div>
      )}
    </div>
  );
}
