export interface Auth { configured: boolean; authenticated: boolean; demo_mode?: boolean }
export interface Column { name: string; kind: 'numeric' | 'categorical'; missing_count: number; missing_rate: number; unique_count: number; sample_values: string[] }
export interface Dataset { id: string; name: string; rows: number; column_count: number; missing_rate: number; created_at: string; source: 'upload' | 'demo'; columns: Column[]; preview: Record<string, string | null>[]; warnings: string[] }
export interface RunConfig { dataset_id: string; name: string; target_column: string; positive_value: string; positive_label: string; id_column: string | null; feature_columns: string[]; categorical_columns: string[]; models: string[]; seed: number }
export interface Reliability { counts: number[]; mean_probability: (number | null)[]; positive_frequency: (number | null)[] }
export interface Metric { model: string; calibration: string; auc: number | null; brier: number | null; ece: number | null; f1: number | null; log_loss: number | null; reliability: Reliability }
export interface Quality { name: string; train_missing_rate: number; test_missing_rate: number }
export interface RunResult { rows: number; positive_label: string; split_sizes: { train: number; calibration: number; test: number }; class_counts: { train: number[]; calibration: number[]; test: number[] }; metrics: Metric[]; feature_quality: Quality[]; warnings: string[]; provenance: object }
export interface Run { id: string; name: string; dataset_id: string; dataset_name: string; status: 'queued' | 'running' | 'completed' | 'failed'; progress: number; message: string; error: string | null; created_at: string; completed_at: string | null; config: RunConfig; result: RunResult | null }
export interface Activity { id: string; action: string; detail: string; created_at: string }
export interface Dashboard { datasets_count: number; runs_count: number; completed_runs: number; prediction_batches: number; recent_runs: Run[]; recent_activity: Activity[] }
export interface Prediction { id: string; run_id: string; run_name: string; model: string; calibration: string; rows: number; mean_probability: number; missing_rate: number; created_at: string; positive_label: string; histogram?: { edges: number[]; counts: number[] }; preview: { record_id: string; probability: number; missing_count: number }[]; quality: { name: string; training_missing_rate: number; batch_missing_rate: number; unseen_rate: number }[]; warnings: string[] }

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, { credentials: 'include', ...options, headers: options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json', ...options.headers } : options.headers });
  if (!response.ok) {
    let message = `请求未完成（${response.status}）`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((item: { msg?: string }) => item.msg || '').join('；');
    } catch { /* retain status information if an upstream response is not JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const modelNames: Record<string, string> = { logistic: 'Logistic Regression', random_forest: 'Random Forest', lightgbm: 'LightGBM', mlp: 'MLP', mask_aware_mlp: 'MLP + 缺失掩码' };
export const calibrationNames: Record<string, string> = { raw: '原始概率', platt: 'Platt 校准', isotonic: 'Isotonic 校准' };
export const number = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('zh-CN').format(value);
export const percent = (value: number | null | undefined, digits = 1) => value == null ? '—' : `${(value * 100).toFixed(digits)}%`;
export const decimal = (value: number | null | undefined, digits = 3) => value == null ? '—' : value.toFixed(digits);
export const date = (value: string) => new Date(value).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : '请求失败，请稍后重试。';
