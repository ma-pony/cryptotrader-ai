import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ChatInput } from './chat-input';

describe('ChatInput', () => {
  it('keeps the text when the current run cannot accept a send', () => {
    const onSend = vi.fn(() => false);
    render(<ChatInput onSend={onSend} onStop={vi.fn()} status="idle" />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'follow up' } });

    fireEvent.click(screen.getByRole('button'));

    expect(onSend).toHaveBeenCalledWith('follow up');
    expect(input).toHaveValue('follow up');
  });
});
