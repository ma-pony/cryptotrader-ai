import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BooleanField, NumberField } from '@/components/configuration/field';
import { Section } from '@/components/configuration/section';
import { useConfiguration } from '../configuration-context';
import { ConfigurationSaveBar } from '..';
import { AllocationPreview } from './allocation-preview';
import { BookForm, newBook } from './book-form';

export default function ExecutionBooksPage() {
  const runtime = useConfiguration();
  const { t } = useTranslation('configuration');
  const form = useRef<HTMLFormElement>(null);
  const [equity, setEquity] = useState<number | ''>(100000);
  const [exposure, setExposure] = useState<number | ''>(0.5);
  const execution = runtime.document?.execution;
  const connections = runtime.baseline?.execution.connections;
  if (!execution || !connections) return null;
  const changeBooks = (books: typeof execution.books) => runtime.update('execution', { ...execution, books });
  return (
    <form
      ref={form}
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        void runtime.save('books', form.current ?? undefined);
      }}
    >
      <header>
        <h1 className="text-xl font-semibold">{t('books')}</h1>
        <p className="configuration-help">{t('booksSubtitle')}</p>
      </header>
      {!execution.books.length ? <p className="configuration-help">{t('book.empty')}</p> : null}
      {execution.books.map((book, index) => (
        <BookForm
          key={runtime.bookKeys[index] ?? book.id}
          index={index}
          book={book}
          saved={runtime.baseline!.execution.books.some((item) => item.id === book.id)}
          connections={connections}
          errors={runtime.errors}
          onChange={(next) => changeBooks(execution.books.map((item, i) => (i === index ? next : item)))}
          onRemove={() => {
            runtime.setBookKeys((current) => current.filter((_, i) => i !== index));
            changeBooks(execution.books.filter((_, i) => i !== index));
          }}
        />
      ))}
      <button
        type="button"
        className="configuration-button"
        onClick={() => {
          runtime.setBookKeys((current) => [...current, crypto.randomUUID()]);
          changeBooks([...execution.books, newBook()]);
        }}
      >
        {t('addBook')}
      </button>
      <Section title={t('liveWrite.title')} description={t('liveWrite.warning')}>
        <BooleanField
          name="execution.live_order_execution_enabled"
          label={t('liveWrite.enable')}
          value={execution.live_order_execution_enabled}
          onChange={(live_order_execution_enabled) =>
            runtime.update('execution', { ...execution, live_order_execution_enabled })
          }
        />
      </Section>
      <Section title={t('allocationPreview')} description={t('book.exampleHelp')}>
        <div className="configuration-grid">
          <NumberField name="example.equity" label={t('sampleEquity')} value={equity} min={0} onChange={setEquity} />
          <NumberField
            name="example.exposure"
            label={t('targetExposure')}
            value={exposure}
            percent
            min={0}
            max={100}
            onChange={setExposure}
          />
        </div>
        {execution.books.map((book, index) => (
          <section key={runtime.bookKeys[index] ?? book.id}>
            <h3 className="text-sm font-semibold">{book.label || t('book.unnamed')}</h3>
            <AllocationPreview
              equity={equity}
              targetExposure={exposure}
              allocations={book.allocations
                .filter((item) => item.enabled)
                .map((item) => ({
                  connectionId: item.connection_id,
                  label:
                    connections.find((connection) => connection.id === item.connection_id)?.label ?? item.connection_id,
                  weight: item.weight === '' ? '' : item.weight * 100,
                }))}
            />
          </section>
        ))}
      </Section>
      <ConfigurationSaveBar section="books" form={form.current} />
    </form>
  );
}
