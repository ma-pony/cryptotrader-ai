interface Props {
  value: string;
  onChange: (v: string) => void;
}

function describeCron(expr: string): string {
  const parts = expr.trim().split(/\s+/);
  if (parts.length !== 5) return '自定义表达式';
  const min = parts[0]!;
  const hour = parts[1]!;
  const dom = parts[2]!;
  const dow = parts[4]!;

  if (dom === '*' && dow === '*') {
    if (min === '0' && hour === '*') return '每小时整点';
    if (min !== '*' && hour !== '*') return `每天 ${hour}:${min.padStart(2, '0')}`;
    if (min !== '*' && hour === '*') return `每小时第 ${min} 分钟`;
  }
  if (dom === '*' && dow !== '*') {
    const dayNames: Record<string, string> = { '1': '周一', '2': '周二', '3': '周三', '4': '周四', '5': '周五', '6': '周六', '0': '周日', '7': '周日' };
    const dayLabel = dayNames[dow] ?? `周${dow}`;
    if (min !== '*' && hour !== '*') return `每${dayLabel} ${hour}:${min.padStart(2, '0')}`;
  }
  if (min === '*' && hour === '*' && dom === '*') return '每分钟';
  return '自定义表达式';
}

export const CronEditor = ({ value, onChange }: Props) => {
  const description = describeCron(value);

  return (
    <div className="space-y-1">
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="* * * * *"
        className="block h-8 w-full rounded-md border border-input bg-background px-2 font-mono text-sm"
        aria-label="Cron expression"
      />
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  );
};
