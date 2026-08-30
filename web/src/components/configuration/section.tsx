import { useId, type ReactNode } from 'react';

export function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <section className="configuration-section" aria-labelledby={id}>
      <header>
        <h2 id={id}>{title}</h2>
        {description ? <p className="configuration-help">{description}</p> : null}
      </header>
      <div className="configuration-grid">{children}</div>
    </section>
  );
}
export function AdvancedSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="configuration-advanced">
      <summary>{title}</summary>
      <div className="configuration-grid">{children}</div>
    </details>
  );
}
