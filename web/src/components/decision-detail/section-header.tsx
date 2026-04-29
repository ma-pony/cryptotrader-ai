import type { ReactNode } from 'react';

interface Props {
  id: string;
  index: number;
  title: string;
  eyebrow?: string;
  right?: ReactNode;
}

export const SectionHeader = ({ id, index, title, eyebrow, right }: Props) => (
  <div id={id} className="flex items-end gap-3 mb-2 scroll-mt-14">
    <div className="flex-1 min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
        {index} · {eyebrow ?? title}
      </div>
      <div className="text-[15px] font-semibold tracking-tight">{title}</div>
    </div>
    {right}
  </div>
);
