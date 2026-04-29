import { memo, useMemo } from 'react';

interface Props {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  strokeWidth?: number;
  fill?: string | undefined;
  showDot?: boolean;
}

// FE-I6: previously min/max/map/join ran on every render — when positions-table
// WS ticks at 200ms and N rows re-render with stable ``data`` props, we rebuilt
// all N paths every tick. useMemo + React.memo short-circuit both.
const SparklineInner = ({
  data,
  width = 80,
  height = 24,
  color = 'currentColor',
  strokeWidth = 1.5,
  fill,
  showDot = true,
}: Props) => {
  const geometry = useMemo(() => {
    if (!data || data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const stepX = width / (data.length - 1);
    const pts = data.map(
      (v, i): [number, number] => [i * stepX, height - ((v - min) / range) * height],
    );
    const path = 'M ' + pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' L ');
    const area = path + ` L ${width},${height} L 0,${height} Z`;
    const last = pts[pts.length - 1];
    return { path, area, last };
  }, [data, width, height]);

  if (geometry === null) return null;

  return (
    <svg width={width} height={height} className="overflow-visible block">
      {fill ? <path d={geometry.area} fill={fill} /> : null}
      <path
        d={geometry.path}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {showDot && geometry.last ? (
        <circle cx={geometry.last[0]} cy={geometry.last[1]} r={2} fill={color} />
      ) : null}
    </svg>
  );
};

export const Sparkline = memo(SparklineInner);
