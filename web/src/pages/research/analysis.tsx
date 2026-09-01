import { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Button } from '@/components/ui/button';
import { TextField } from '@/components/configuration/field';
import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { useStartAnalysis } from '@/hooks/use-decisions';
import { ResearchNav } from './presentation';

export default function ResearchAnalysis() {
  const [pair, setPair] = useState('BTC/USDT:USDT');
  const readiness = useRuntimeStatus();
  const start = useStartAnalysis();
  const navigate = useNavigate();
  return (
    <main className="space-y-6 text-sm">
      <ResearchNav />
      <h1 className="text-xl font-semibold">仅分析</h1>
      <p>使用已保存的信号引擎配置。不读取交易账户，不创建审批或订单；模型请求可能产生费用。</p>
      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (!readiness.data?.analysis.ready || start.isPending) return;
          start.mutate(
            { pair: pair.trim(), expected_revision: readiness.data.saved_revision },
            {
              onSuccess: ({ decision_id }) => {
                void navigate(`/decisions/${decision_id}`);
              },
            },
          );
        }}
      >
        <TextField name="analysis-pair" label="分析交易对" value={pair} onChange={setPair} disabled={start.isPending} />
        <Button className="min-h-10" disabled={start.isPending || !readiness.data?.analysis.ready || !pair.trim()}>
          仅分析，不交易
        </Button>
      </form>
      {readiness.isError ? (
        <p role="alert">
          分析就绪状态读取失败。
          <Button className="min-h-10" onClick={() => void readiness.refetch()}>
            重新读取
          </Button>
        </p>
      ) : null}
      {readiness.data?.analysis.reasons.map((reason) => (
        <p key={reason.code}>{reason.message}</p>
      ))}
      {start.isError ? <p role="alert">分析未能启动，交易对已保留；请核对配置后重试。</p> : null}
      <Link className="inline-flex min-h-11 items-center text-primary" to="/decisions">
        查看原决策记录
      </Link>
    </main>
  );
}
