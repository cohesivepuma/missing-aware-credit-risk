import { Metric, Prediction, Quality, percent } from './api';

const colors: Record<string, string> = { raw: '#b1bcc2', platt: '#187d69', isotonic: '#c3984a' };
const names: Record<string, string> = { raw: '原始概率', platt: 'Platt', isotonic: 'Isotonic' };

export function ReliabilityChart({ metrics }: { metrics: Metric[] }) {
  const width = 560, height = 284, left = 48, top = 17, right = 22, bottom = 44;
  const x = (v: number) => left + v * (width - left - right);
  const y = (v: number) => height - bottom - v * (height - top - bottom);
  return <div className="chart-block"><svg className="chart" role="img" aria-label="可靠性曲线：横轴为预测概率，纵轴为实际正类频率，越贴近对角线则校准越好" viewBox={`0 0 ${width} ${height}`}>
    {[0, .25, .5, .75, 1].map(tick => <g key={tick}><line x1={left} y1={y(tick)} x2={width - right} y2={y(tick)} stroke="#e8ecea"/><text x={left - 12} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text><text x={x(tick)} y={height - bottom + 22} textAnchor="middle">{tick.toFixed(2)}</text></g>)}
    <path d={`M${x(0)},${y(0)} L${x(1)},${y(1)}`} stroke="#c3ceca" strokeDasharray="5 5" fill="none"/>
    {metrics.map(metric => {
      const points = metric.reliability.mean_probability.flatMap((probability, i) => probability == null || metric.reliability.positive_frequency[i] == null ? [] : [{ x: probability, y: metric.reliability.positive_frequency[i]!, count: metric.reliability.counts[i] }]);
      return <g key={metric.calibration}><polyline points={points.map(point => `${x(point.x)},${y(point.y)}`).join(' ')} fill="none" stroke={colors[metric.calibration] || '#187d69'} strokeWidth="2.5" strokeLinejoin="round"/>{points.map((point, i) => <circle key={i} cx={x(point.x)} cy={y(point.y)} r="3.5" fill={colors[metric.calibration] || '#187d69'} stroke="#fff" strokeWidth="1.5"><title>{names[metric.calibration]} · 预测 {percent(point.x)} / 实际 {percent(point.y)} · {point.count} 条</title></circle>)}</g>;
    })}
    <text x={width / 2} y={height - 3} textAnchor="middle">预测正类概率</text><text transform={`translate(12 ${height / 2 - 8}) rotate(-90)`} textAnchor="middle">实际正类频率</text>
  </svg><div className="chart-legend">{metrics.map(metric => <span key={metric.calibration}><i style={{ background: colors[metric.calibration] }}/>{names[metric.calibration] || metric.calibration}</span>)}<span><i className="dashed"/>理想校准</span></div></div>;
}

export function MissingChart({ data }: { data: Quality[] }) {
  const items = [...data].sort((a, b) => b.train_missing_rate - a.train_missing_rate).slice(0, 8);
  const max = Math.max(.1, ...items.flatMap(item => [item.train_missing_rate, item.test_missing_rate]));
  return <div className="missing-chart"><div className="chart-legend left"><span><i style={{ background: '#187d69' }}/>训练集</span><span><i style={{ background: '#b6d5ca' }}/>测试集</span></div>{items.map(item => <div className="missing-row" key={item.name}><span className="truncate" title={item.name}>{item.name}</span><div className="missing-bars"><div style={{ width: `${item.train_missing_rate / max * 100}%` }}/><div style={{ width: `${item.test_missing_rate / max * 100}%` }}/></div><span>{percent(item.train_missing_rate)}</span></div>)}{!items.length && <p className="muted">暂无特征质量数据</p>}<p className="chart-footnote">按训练集缺失率排序，展示最多 8 个特征。</p></div>;
}

export function ProbabilityHistogram({ batch }: { batch: Prediction }) {
  const bins = batch.histogram?.counts || Array.from({ length: 10 }, () => 0);
  if (!batch.histogram) batch.preview.forEach(row => { bins[Math.min(9, Math.max(0, Math.floor(row.probability * 10)))] += 1; });
  const max = Math.max(1, ...bins);
  return <div className="histogram"><svg role="img" aria-label="正类概率分布直方图，横轴为正类概率区间，纵轴为记录数" viewBox="0 0 500 210">{[0, .5, 1].map(tick => <g key={tick}><line x1="30" x2="484" y1={170 - tick * 135} y2={170 - tick * 135} stroke="#e8ecea"/><text x="23" y={174 - tick * 135} textAnchor="end">{Math.round(max * tick)}</text></g>)}{bins.map((count, index) => <g key={index}><rect x={36 + index * 45} y={170 - count / max * 135} width="32" height={count / max * 135} rx="3" fill="#288975" opacity={.5 + index * .05}><title>{index * 10}–{(index + 1) * 10}%：{count} 条</title></rect><text x={52 + index * 45} y="190" textAnchor="middle">{index * 10}%</text></g>)}</svg><p className="chart-footnote">{batch.histogram ? `基于本次评分的全部 ${batch.rows.toLocaleString('zh-CN')} 条记录。` : `仅展示接口返回的 ${numberPreview(batch)} 条预览记录，不代表全批次分布。`}</p></div>;
}
const numberPreview = (batch: Prediction) => batch.preview.length.toLocaleString('zh-CN');
