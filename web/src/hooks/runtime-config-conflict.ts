import { useSyncExternalStore } from 'react';

let conflicted = false;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((listener) => listener());
export const setRuntimeConfigConflict = () => { conflicted = true; notify(); };
export const clearRuntimeConfigConflict = () => { if (conflicted) { conflicted = false; notify(); } };
export const useRuntimeConfigConflict = () => useSyncExternalStore((listener) => { listeners.add(listener); return () => listeners.delete(listener); }, () => conflicted, () => false);
