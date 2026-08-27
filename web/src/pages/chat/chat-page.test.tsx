import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';

const mockSendMessage = vi.fn();
const mockStopStream = vi.fn();
const mockClearMessages = vi.fn();
const mockNavigate = vi.fn();
let mockCancelled = false;

vi.mock('react-router', async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  let locationState: unknown = null;
  return {
    ...actual,
    useParams: () => ({}),
    useNavigate: () => mockNavigate,
    useLocation: () => ({ state: locationState, pathname: '/chat', search: '', hash: '', key: 'default' }),
    __setLocationState: (s: unknown) => { locationState = s; },
  };
});

vi.mock('@/hooks/use-chat-messages', () => ({
  useChatMessages: () => ({
    messages: [],
    status: 'idle' as const,
    error: null,
    sendMessage: mockSendMessage,
    stopStream: mockStopStream,
    clearMessages: mockClearMessages,
  }),
}));

vi.mock('@/hooks/use-analysis-progress', () => ({
  useAnalysisProgress: () => ({
    progress: {
      cycleId: null,
      status: mockCancelled ? 'cancelled' : 'idle',
      components: {},
      agents: {},
      debateRound: 0,
      fusion: null,
      target: null,
      riskCheck: null,
      cancelled: mockCancelled,
      lastEventId: 0,
    },
    handleProgressEvent: vi.fn(),
    reset: vi.fn(),
    sendInterrupt: vi.fn(),
    sendSteer: vi.fn(),
  }),
}));

vi.mock('@/stores/use-chat-store', () => ({
  useChatStore: () => ({
    sessions: [],
    activeSessionId: null,
    setActiveSession: vi.fn(),
    removeSession: vi.fn(),
  }),
}));

vi.mock('./components/chat-input', () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock('./components/message-stream', () => ({
  MessageStream: () => <div data-testid="message-stream" />,
}));

vi.mock('./components/session-list', () => ({
  SessionList: () => <div data-testid="session-list" />,
}));

describe('ChatPage', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
    vi.clearAllMocks();
    mockCancelled = false;
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it('renders without chart context', async () => {
    const ChatPage = (await import('./index')).default;
    render(<ChatPage />);

    expect(screen.getByTestId('chat-input')).toBeInTheDocument();
    expect(screen.getByTestId('message-stream')).toBeInTheDocument();
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it('auto-sends analysis request when additionalContext is present', async () => {
    vi.useRealTimers();

    const { __setLocationState } = await import('react-router') as unknown as { __setLocationState: (s: unknown) => void };
    __setLocationState({
      additionalContext: {
        payloads: [{ symbol: 'BTC/USDT', timeframe: '1h', exchange: 'binance', dataUrl: null, description: 'test', capturedAt: '2026-01-01' }],
        model: '',
      },
    });

    const ChatPage = (await import('./index')).default;
    render(<ChatPage />);

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith(
        '请分析当前图表',
        expect.objectContaining({
          payloads: expect.arrayContaining([
            expect.objectContaining({ symbol: 'BTC/USDT', timeframe: '1h' }),
          ]),
        }),
      );
    });

    __setLocationState(null);
  });

  it('shows chart context badge', async () => {
    const { __setLocationState } = await import('react-router') as unknown as { __setLocationState: (s: unknown) => void };
    __setLocationState({
      additionalContext: {
        payloads: [{ symbol: 'ETH/USDT', timeframe: '4h', exchange: 'binance', dataUrl: null, description: '', capturedAt: '' }],
        model: '',
      },
    });

    const ChatPage = (await import('./index')).default;
    render(<ChatPage />);

    expect(screen.getByText(/ETH\/USDT/)).toBeInTheDocument();
    expect(screen.getByText(/4h/)).toBeInTheDocument();

    __setLocationState(null);
  });

  it('shows cancellation without constructing a partial verdict', async () => {
    mockCancelled = true;
    const ChatPage = (await import('./index')).default;
    render(<ChatPage />);

    expect(screen.getByText('本轮分析已取消')).toBeInTheDocument();
    expect(screen.queryByText('部分裁决')).not.toBeInTheDocument();
  });
});
