import { Monitor } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

const MIN_WIDTH = 1024;

export const DesktopOnlyBanner = () => {
  const { t } = useTranslation();
  const [tooNarrow, setTooNarrow] = useState(false);

  useEffect(() => {
    const check = () => setTooNarrow(window.innerWidth < MIN_WIDTH);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  if (!tooNarrow) return null;
  return (
    <div
      role="status"
      className="border-b border-warning/30 bg-warning/10 px-4 py-2 text-center text-xs text-warning"
    >
      <Monitor className="mr-2 inline h-3.5 w-3.5" aria-hidden="true" />
      <span className="font-semibold">{t('desktop_only.title')}</span>
      <span className="ml-2">{t('desktop_only.body')}</span>
    </div>
  );
};
