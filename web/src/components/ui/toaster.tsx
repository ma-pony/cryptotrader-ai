import { create } from 'zustand';

import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from './toast';

import type { ReactNode } from 'react';

export interface ToastInput {
  title?: ReactNode;
  description?: ReactNode;
  variant?: 'default' | 'destructive';
  durationMs?: number;
}

interface ToastEntry extends ToastInput {
  id: number;
}

interface ToastState {
  items: ToastEntry[];
  push: (input: ToastInput) => number;
  dismiss: (id: number) => void;
}

let counter = 0;
const nextId = () => ++counter;

export const useToastStore = create<ToastState>((set) => ({
  items: [],
  push: (input) => {
    const id = nextId();
    set((s) => ({ items: [...s.items, { ...input, id }] }));
    return id;
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((x) => x.id !== id) })),
}));

export const toast = (input: ToastInput) => useToastStore.getState().push(input);

export const Toaster = () => {
  const items = useToastStore((s) => s.items);
  const dismiss = useToastStore((s) => s.dismiss);
  return (
    <ToastProvider>
      {items.map((item) => (
        <Toast
          key={item.id}
          variant={item.variant ?? 'default'}
          duration={item.durationMs ?? 5000}
          onOpenChange={(open) => {
            if (!open) dismiss(item.id);
          }}
        >
          <div className="grid gap-1">
            {item.title ? <ToastTitle>{item.title}</ToastTitle> : null}
            {item.description ? <ToastDescription>{item.description}</ToastDescription> : null}
          </div>
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  );
};
