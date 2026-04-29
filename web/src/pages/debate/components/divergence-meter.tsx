interface Props {
  before: number;
  after: number;
  target: number;
}

export const DivergenceMeter = ({ before, after, target }: Props) => (
  <div className="flex flex-col gap-2 w-[280px]">
    <div className="flex justify-between text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
      <span>分歧度</span>
      <span>收敛目标 {target.toFixed(2)}</span>
    </div>
    <div className="relative h-2 rounded overflow-hidden bg-muted">
      <div
        className="absolute top-0 bottom-0 w-px bg-amber-500 opacity-60"
        style={{ left: `${target * 100}%` }}
      />
      <div
        className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-trade-long to-amber-400"
        style={{ width: `${after * 100}%` }}
      />
      <div
        className="absolute -top-[3px] -bottom-[3px] w-0.5 bg-foreground shadow-[0_0_8px_rgba(255,255,255,0.3)]"
        style={{ left: `${after * 100}%` }}
      />
    </div>
    <div className="flex justify-between text-[11px]">
      <span className="font-mono text-muted-foreground">开始 {before.toFixed(2)}</span>
      <span className="font-mono font-medium text-amber-400">→ 结束 {after.toFixed(2)}</span>
    </div>
  </div>
);
