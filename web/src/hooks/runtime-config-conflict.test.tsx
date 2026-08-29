import { renderHook } from '@testing-library/react';
import { act } from 'react';
import { expect, it } from 'vitest';
import { clearRuntimeConfigConflict, setRuntimeConfigConflict, useRuntimeConfigConflict } from './runtime-config-conflict';

it('shares conflict between independent runtime hook instances and clears only explicitly', () => {
  const first = renderHook(() => useRuntimeConfigConflict()); const second = renderHook(() => useRuntimeConfigConflict());
  act(setRuntimeConfigConflict); expect(first.result.current).toBe(true); expect(second.result.current).toBe(true);
  act(clearRuntimeConfigConflict); expect(first.result.current).toBe(false); expect(second.result.current).toBe(false);
});
