import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { Button } from '@/components/ui/button';

const NotFoundPage = () => {
  const { t } = useTranslation();
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
      <p className="text-6xl font-bold text-muted-foreground">404</p>
      <p className="text-lg text-foreground">{t('errors.not_found')}</p>
      <Button asChild variant="outline">
        <Link to="/">{t('nav.dashboard')}</Link>
      </Button>
    </div>
  );
};

export default NotFoundPage;
