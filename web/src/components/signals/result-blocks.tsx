import { useId, useState } from 'react';
import type { ResultBlock } from '@/types/api';

const scalar = (value: string | number | boolean | null | undefined) =>
  value == null ? '暂无数据' : typeof value === 'boolean' ? (value ? '是' : '否') : String(value);
const timeLabel = (time: string) => new Date(time).toLocaleString('zh-CN', { hour12: false });
const colors = ['var(--color-primary, #4f86d9)', '#ca8a04', '#16a394', '#a36dc2'];

function SavedSeries({ block }: { block: Extract<ResultBlock, { kind: 'series' }> }) {
  const values = block.series.flatMap((series) =>
    series.points
      .filter((point) => point.value !== null)
      .map((point) => ({ time: Date.parse(point.time), value: Number(point.value) })),
  );
  if (values.length === 0) return <p className="text-muted-foreground">暂无序列数据</p>;
  const times = values.map((point) => point.time);
  const prices = values.map((point) => point.value);
  const first = Math.min(...times),
    last = Math.max(...times),
    low = Math.min(...prices),
    high = Math.max(...prices);
  const x = (time: number) => 60 + ((time - first) / (last - first || 1)) * 680;
  const y = (value: number) => 210 - ((value - low) / (high - low || 1)) * 170;
  const boundary = block.forecast_start ? Math.max(60, Math.min(740, x(Date.parse(block.forecast_start)))) : null;
  return (
    <div className="space-y-2">
      <svg viewBox="0 0 800 260" role="img" aria-label={block.title} className="w-full min-h-48 text-muted-foreground">
        {boundary !== null ? (
          <rect x={boundary} y="24" width={740 - boundary} height="196" fill="#ca8a04" opacity="0.1" />
        ) : null}
        {[low, (low + high) / 2, high].map((value, index) => (
          <g key={index}>
            <line x1="60" x2="740" y1={y(value)} y2={y(value)} stroke="currentColor" opacity="0.15" />
            <text x="54" y={y(value) + 4} textAnchor="end" fill="currentColor" fontSize="12">
              {value.toLocaleString('zh-CN', { maximumFractionDigits: 2 })}
            </text>
          </g>
        ))}
        {block.series.map((series, index) => {
          let penDown = false;
          const path = series.points
            .map((point) => {
              if (point.value === null) {
                penDown = false;
                return '';
              }
              const command = penDown ? 'L' : 'M';
              penDown = true;
              return `${command}${x(Date.parse(point.time))},${y(Number(point.value))}`;
            })
            .join(' ');
          return <path key={index} d={path} fill="none" stroke={colors[index % colors.length]} strokeWidth="2" />;
        })}
        {boundary !== null ? (
          <line x1={boundary} x2={boundary} y1="24" y2="220" stroke="#ca8a04" strokeDasharray="4 4" />
        ) : null}
        <text x="60" y="246" fill="currentColor" fontSize="12">
          {new Date(first).toLocaleString('zh-CN')}
        </text>
        <text x="740" y="246" textAnchor="end" fill="currentColor" fontSize="12">
          {new Date(last).toLocaleString('zh-CN')}
        </text>
      </svg>
      <ul className="flex flex-wrap gap-4 text-sm">
        {block.series.map((series, index) => (
          <li key={index} style={{ color: colors[index % colors.length] }}>
            {series.name}
            {series.unit ? `（${series.unit}）` : ''}
          </li>
        ))}
      </ul>
      {block.forecast_start ? (
        <p className="text-sm text-muted-foreground">
          预测区从 {timeLabel(block.forecast_start)} 开始；展示当次保存结果。
        </p>
      ) : null}
    </div>
  );
}

function SavedTimeline({ block }: { block: Extract<ResultBlock, { kind: 'timeline' }> }) {
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  return (
    <div>
      <button
        type="button"
        className="configuration-button min-h-10"
        aria-expanded={expanded}
        aria-controls={id}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? '收起' : '展开'}
        {block.title}
      </button>
      <ol id={id} hidden={!expanded} className="mt-4 space-y-5 border-l border-border pl-4">
        {block.entries.map((entry, index) => (
          <li key={index}>
            <div className="flex flex-wrap gap-2">
              <strong>{entry.actor}</strong>
              <time className="text-sm text-muted-foreground" dateTime={entry.time}>
                {timeLabel(entry.time)}
              </time>
            </div>
            <p className="mt-2 whitespace-pre-wrap break-words">{entry.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

function BlockContent({ block }: { block: ResultBlock }) {
  switch (block.kind) {
    case 'text':
      return <p className="whitespace-pre-wrap break-words">{block.body}</p>;
    case 'metrics':
      return (
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {block.metrics.map((metric, index) => (
            <div key={index}>
              <dt className="text-muted-foreground">{metric.key}</dt>
              <dd className="mt-1 font-medium">
                {scalar(metric.value)} {metric.unit}
              </dd>
              {metric.note ? <p className="text-sm text-muted-foreground">{metric.note}</p> : null}
            </div>
          ))}
        </dl>
      );
    case 'series':
      return <SavedSeries block={block} />;
    case 'table':
      return (
        <div
          role="region"
          aria-label={block.title}
          tabIndex={0}
          className="max-w-full overflow-x-auto focus-visible:outline focus-visible:outline-2"
        >
          <table className="w-full min-w-max border-collapse text-left">
            <thead>
              <tr>
                {block.columns.map((column) => (
                  <th key={column.key} className="border-b border-border p-3" scope="col">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, index) => (
                <tr key={index}>
                  {block.columns.map((column) => (
                    <td key={column.key} className="border-b border-border p-3">
                      {scalar(row.cells.find((cell) => cell.column_key === column.key)?.value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case 'timeline':
      return <SavedTimeline block={block} />;
  }
}

export function ResultBlocks({ blocks }: { blocks: ResultBlock[] }) {
  if (!blocks.length) return <p className="text-sm text-muted-foreground">本次没有展示数据。</p>;
  return (
    <div className="min-w-0 space-y-4 text-sm">
      {blocks.map((block, index) => (
        <section key={index} className="min-w-0 rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 font-semibold">{block.title}</h3>
          <BlockContent block={block} />
        </section>
      ))}
    </div>
  );
}
