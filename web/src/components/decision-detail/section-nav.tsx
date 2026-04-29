import { useEffect, useRef, useState } from 'react';

import { cn } from '@/lib/cn';

export interface SectionDef {
  id: string;
  label: string;
}

interface Props {
  sections: SectionDef[];
  containerRef: React.RefObject<HTMLDivElement | null>;
}

// FE-I5: scroll fires 20-60×/sec during a flick; without debounce each event
// synchronously setState'd → N re-renders. 60ms debounce matches perceptual
// scroll latency while keeping the active-section highlight responsive.
const SCROLL_DEBOUNCE_MS = 60;

export const SectionNav = ({ sections, containerRef }: Props) => {
  const [active, setActive] = useState<string | undefined>(sections[0]?.id);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const compute = () => {
      const threshold = container.scrollTop + 80;
      let current: string | undefined = sections[0]?.id;
      for (const s of sections) {
        const el = container.querySelector<HTMLElement>(`#${s.id}`);
        if (el && el.offsetTop <= threshold) current = s.id;
      }
      setActive((prev) => (prev === current ? prev : current));
    };

    const handleScroll = () => {
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(compute, SCROLL_DEBOUNCE_MS);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    compute(); // run once on mount so initial active reflects scroll position
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (debounceRef.current !== null) clearTimeout(debounceRef.current);
    };
  }, [sections, containerRef]);

  const jumpTo = (id: string) => {
    const container = containerRef.current;
    const el = container?.querySelector<HTMLElement>(`#${id}`);
    if (!container || !el) return;
    container.scrollTo({ top: el.offsetTop - 12, behavior: 'smooth' });
  };

  return (
    <nav
      className="sticky top-0 z-10 flex gap-1 overflow-x-auto border-b border-border bg-card/95 backdrop-blur px-2 py-2"
      aria-label="decision sections"
    >
      {sections.map((s) => (
        <button
          key={s.id}
          onClick={() => jumpTo(s.id)}
          className={cn(
            'whitespace-nowrap rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors',
            active === s.id
              ? 'bg-amber-500/15 text-amber-500 border border-amber-500/40'
              : 'text-muted-foreground hover:bg-muted hover:text-foreground border border-transparent',
          )}
        >
          {s.label}
        </button>
      ))}
    </nav>
  );
};
