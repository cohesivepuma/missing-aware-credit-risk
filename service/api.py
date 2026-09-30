"""AWARE application API and production SPA host. Run a single worker."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import csv
import hashlib
import io
import ipaddress
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from service.http_limits import BodyLimitMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from service.security import Sessions, password_hash, verify_password
from service.storage import Store, new_id, now

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 20 * 1024 * 1024
LOGGER = logging.getLogger('aware')


class PasswordInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    password: str = Field(min_length=12, max_length=256)


class RunInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    name: str = Field(min_length=1, max_length=100)
    target_column: str = Field(min_length=1, max_length=200)
    positive_value: str = Field(min_length=1, max_length=200)
    positive_label: str = Field(min_length=1, max_length=120)
    id_column: str | None = None
    feature_columns: list[str] = Field(min_length=1, max_length=99)
    categorical_columns: list[str] = Field(default_factory=list, max_length=99)
    models: list[Literal['logistic', 'random_forest', 'lightgbm', 'mlp', 'mask_aware_mlp']] = Field(min_length=1, max_length=5)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator('name')
    @classmethod
    def meaningful_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('分析名称不能为空。')
        return value


def loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or '').is_loopback
    except ValueError:
        return host in {'localhost', 'testclient'}


def safe_csv(frame) -> bytes:
    """Neutralize spreadsheet formula execution in exported string identifiers."""
    copied = frame.copy()
    for column in copied.select_dtypes(include=['object', 'string']).columns:
        copied[column] = copied[column].map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value)
    return copied.to_csv(index=False).encode('utf-8-sig')


def create_app(storage_dir: Path | str | None = None, *, demo_mode: bool | None = None,
               seed_demo: bool = True) -> FastAPI:
    directory = Path(storage_dir or os.environ.get('AWARE_STORAGE_DIR', ROOT / 'var/workbench'))
    demo = os.environ.get('AWARE_DEMO_MODE', '0') == '1' if demo_mode is None else demo_mode
    store = Store(directory)
    sessions = Sessions(store, secure_cookie=os.environ.get('AWARE_SECURE_COOKIE') == '1')
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='aware-training')
    admission = Lock()
    port = os.environ.get('AWARE_PORT', '8000')
    allowed_origins = {f'http://127.0.0.1:{port}', f'http://localhost:{port}',
                       'http://127.0.0.1:5173', 'http://localhost:5173', 'http://testserver'}
    allowed_origins.update(filter(None, os.environ.get('AWARE_ALLOWED_ORIGINS', '').split(',')))

    def find(kind: str, identifier: str) -> dict:
        value = store.get(kind, identifier)
        if value is None:
            raise HTTPException(404, '记录不存在。')
        return value

    def public_run(value: dict) -> dict:
        return {k: v for k, v in value.items() if k != 'bundle_sha256'}

    def save_dataset(content: bytes, name: str, source: str) -> dict:
        from service.engine import read_csv_bytes, profile_frame
        frame = read_csv_bytes(content)
        summary = profile_frame(frame)
        identifier = new_id()
        path = store.path('datasets', identifier, '.csv')
        path.write_bytes(content)
        os.chmod(path, 0o600)
        value = {'id': identifier, 'name': name[:150], 'created_at': now(), 'source': source,
                 **summary, 'sha256': hashlib.sha256(content).hexdigest()}
        store.put('datasets', value)
        store.add_activity('dataset_imported', f'{name[:100]} · {summary["rows"]} 行')
        return value

    def read_dataset(identifier: str):
        from service.engine import read_csv_bytes
        dataset = find('datasets', identifier)
        content = store.path('datasets', identifier, '.csv').read_bytes()
        if hashlib.sha256(content).hexdigest() != dataset['sha256']:
            raise ValueError('数据文件校验失败，请重新导入。')
        return read_csv_bytes(content)

    def train_worker(identifier: str) -> None:
        from service.engine import train_analysis
        run = find('runs', identifier)
        store.update('runs', identifier, status='running', progress=3, message='检查数据与独立划分')
        try:
            def progress(value: int, message: str) -> None:
                store.update('runs', identifier, progress=max(3, min(98, int(value))), message=message[:200])
            result, bundle = train_analysis(read_dataset(run['dataset_id']), run['config'], progress)
            path = store.path('runs', identifier, '.joblib')
            temporary = path.with_suffix('.partial')
            joblib.dump(bundle, temporary, compress=3)
            os.chmod(temporary, 0o600)
            temporary.replace(path)
            store.update('runs', identifier, status='completed', progress=100,
                         message='模型训练与独立测试完成', completed_at=now(), result=result,
                         bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            store.add_activity('analysis_completed', run['name'])
        except Exception as error:
            LOGGER.exception('Analysis failed: %s', identifier)
            message = str(error)[:500] if isinstance(error, ValueError) else '计算未完成，请检查数据或查看本地服务日志后重新发起。'
            store.update('runs', identifier, status='failed', error=message, message=message, completed_at=now())
            store.add_activity('analysis_failed', run['name'])

    def queue_run(config: dict) -> dict:
        from service.engine import validate_training
        dataset = find('datasets', config['dataset_id'])
        with admission:
            active = [r for r in store.list('runs') if r['status'] in {'queued', 'running'}]
            if len(active) >= 4:
                raise HTTPException(429, '已有 4 个分析排队或运行，请等待完成。')
            validate_training(read_dataset(dataset['id']), config)
            run = {'id': new_id(), 'name': config['name'].strip(), 'dataset_id': dataset['id'],
                   'dataset_name': dataset['name'], 'status': 'queued', 'progress': 0,
                   'message': '等待计算', 'error': None, 'created_at': now(), 'completed_at': None,
                   'config': config, 'result': None}
            store.put('runs', run)
            store.add_activity('analysis_queued', run['name'])
            executor.submit(train_worker, run['id'])
            return run

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Stored training state cannot truthfully continue after a worker restart.
        for run in store.list('runs'):
            if run['status'] in {'queued', 'running'}:
                store.update('runs', run['id'], status='failed', error='服务曾中断，请重新发起此分析。',
                             message='服务中断', completed_at=now())
        if demo and seed_demo and not store.list('datasets'):
            from service.engine import demo_frame
            dataset = save_dataset(demo_frame().to_csv(index=False).encode(), '信用数据样例（模拟）.csv', 'demo')
            features = [c['name'] for c in dataset['columns'] if c['name'] != 'outcome']
            queue_run({'dataset_id': dataset['id'], 'name': '示例 · 缺失数据与概率校准',
                       'target_column': 'outcome', 'positive_value': '1', 'positive_label': '模拟逾期事件',
                       'id_column': None, 'feature_columns': features,
                       'categorical_columns': [c['name'] for c in dataset['columns'] if c['kind'] == 'categorical' and c['name'] in features],
                       'models': ['logistic', 'random_forest'], 'seed': 42})
        yield
        executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title='AWARE Workbench', version='1.0.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.executor = executor
    app.state.queue_run = queue_run
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver',
                                                           *filter(None, os.environ.get('AWARE_ALLOWED_HOSTS', '').split(','))])
    app.add_middleware(CORSMiddleware, allow_origins=sorted(allowed_origins), allow_credentials=True,
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @app.middleware('http')
    async def boundaries(request: Request, call_next):
        if demo and not loopback(request.client.host if request.client else None):
            return JSONResponse({'detail': '演示模式仅允许本机访问。'}, status_code=403)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if origin and origin not in allowed_origins:
                return JSONResponse({'detail': '请求来源不在允许列表。'}, status_code=403)
        public_paths = {'/api/health', '/api/auth/status', '/api/auth/setup', '/api/auth/login', '/api/auth/logout'}
        if request.url.path.startswith('/api/') and request.url.path not in public_paths and request.method != 'OPTIONS':
            if not demo and not sessions.authenticated(request):
                return JSONResponse({'detail': '请先登录。'}, status_code=401)
        length = request.headers.get('content-length')
        if length:
            try:
                if int(length) > MAX_UPLOAD + 65536:
                    return JSONResponse({'detail': '上传大小不能超过 20 MiB。'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail': '无效的请求大小。'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'DENY'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        issues = ['.'.join(str(x) for x in item['loc'][1:]) + ': ' + item['msg'] for item in error.errors()[:3]]
        return JSONResponse({'detail': '请检查输入：' + '；'.join(issues)}, status_code=422)

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError):
        return JSONResponse({'detail': str(error)[:600]}, status_code=422)

    def require_auth(request: Request) -> None:
        if not demo and not sessions.authenticated(request):
            raise HTTPException(401, '请先登录。')

    protected = [Depends(require_auth)]

    async def upload_bytes(file: UploadFile) -> bytes:
        try:
            content = await file.read(MAX_UPLOAD + 1)
        finally:
            await file.close()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, 'CSV 文件不能超过 20 MiB。')
        return content

    @app.get('/api/health')
    def health():
        return {'status': 'ok'}

    @app.get('/api/auth/status')
    def auth_status(request: Request):
        return {'configured': bool(store.setting('password')) or demo,
                'authenticated': demo or sessions.authenticated(request), 'demo_mode': demo}

    @app.post('/api/auth/setup')
    def setup(body: PasswordInput, request: Request, response: Response):
        if demo:
            raise HTTPException(409, '演示模式不需要配置账号。')
        if not loopback(request.client.host if request.client else None):
            raise HTTPException(403, '首次设置请在服务器本机完成。')
        if not store.set_initial_password(password_hash(body.password)):
            raise HTTPException(409, '账号已配置，请登录。')
        sessions.issue(response)
        store.add_activity('account_configured', '工作台账号已配置')
        return {'ok': True}

    @app.post('/api/auth/login')
    def login(body: PasswordInput, request: Request, response: Response):
        host = request.client.host if request.client else 'unknown'
        sessions.check_attempt(host)
        stored = store.setting('password')
        if stored is None or not verify_password(body.password, stored):
            sessions.failed_attempt(host)
            raise HTTPException(401, '密码错误或账号尚未配置。')
        sessions.issue(response)
        return {'ok': True}

    @app.post('/api/auth/logout')
    def logout(request: Request, response: Response):
        sessions.logout(request, response)
        return {'ok': True}

    @app.get('/api/dashboard', dependencies=protected)
    def dashboard():
        runs = store.list('runs')
        return {'datasets_count': len(store.list('datasets')), 'runs_count': len(runs),
                'completed_runs': sum(r['status'] == 'completed' for r in runs),
                'prediction_batches': len(store.list('predictions')),
                'recent_runs': [public_run(r) for r in runs[:6]], 'recent_activity': store.activities()}

    @app.get('/api/datasets', dependencies=protected)
    def datasets():
        return {'items': store.list('datasets')}

    @app.post('/api/datasets', dependencies=protected, status_code=201)
    async def upload_dataset(file: UploadFile = File(...)):
        filename = Path((file.filename or 'data.csv').replace('\\', '/')).name
        content = await upload_bytes(file)
        return await run_in_threadpool(save_dataset, content, filename, 'upload')

    @app.post('/api/datasets/demo', dependencies=protected, status_code=201)
    def demo_dataset():
        from service.engine import demo_frame
        return save_dataset(demo_frame().to_csv(index=False).encode(), '信用数据样例（模拟）.csv', 'demo')

    @app.get('/api/datasets/{identifier}', dependencies=protected)
    def dataset_detail(identifier: str):
        return find('datasets', identifier)

    @app.get('/api/datasets/{identifier}/download', dependencies=protected)
    def download_dataset(identifier: str):
        find('datasets', identifier)
        return FileResponse(store.path('datasets', identifier, '.csv'), media_type='text/csv', filename='dataset.csv')

    @app.get('/api/runs', dependencies=protected)
    def runs():
        return {'items': [public_run(r) for r in store.list('runs')]}

    @app.post('/api/runs', dependencies=protected, status_code=202)
    def create_run(body: RunInput):
        return public_run(queue_run(body.model_dump()))

    @app.get('/api/runs/{identifier}', dependencies=protected)
    def run_detail(identifier: str):
        return public_run(find('runs', identifier))

    @app.get('/api/runs/{identifier}/report', dependencies=protected)
    def report(identifier: str):
        run = find('runs', identifier)
        if run['status'] != 'completed':
            raise HTTPException(409, '分析尚未完成。')
        return JSONResponse(public_run(run), headers={'Content-Disposition': f'attachment; filename="aware-report-{identifier[:8]}.json"'})

    @app.get('/api/runs/{identifier}/template', dependencies=protected)
    def template(identifier: str):
        config = find('runs', identifier)['config']
        headers = ([config['id_column']] if config.get('id_column') else []) + config['feature_columns']
        stream = io.StringIO()
        csv.writer(stream).writerow(headers)
        return Response(stream.getvalue().encode('utf-8-sig'), media_type='text/csv',
                        headers={'Content-Disposition': 'attachment; filename="scoring-template.csv"'})

    @app.post('/api/runs/{identifier}/predict', dependencies=protected, status_code=201)
    async def predict(identifier: str, file: UploadFile = File(...), model: str = Form(...), calibration: str = Form(...)):
        # Parse upload asynchronously, then offload CPU prediction so polling stays responsive.
        from starlette.concurrency import run_in_threadpool
        from service.engine import read_csv_bytes, score_batch
        run = find('runs', identifier)
        if run['status'] != 'completed':
            raise HTTPException(409, '只有完成的分析可以用于评分。')
        if model not in run['config']['models'] or calibration not in {'raw', 'platt', 'isotonic'}:
            raise HTTPException(422, '请选择此分析实际拟合的模型与校准方式。')
        content = await upload_bytes(file)

        def work():
            path = store.path('runs', identifier, '.joblib')
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != run.get('bundle_sha256'):
                raise ValueError('模型文件校验失败，请重新训练。')
            # Only server-generated, hash-verified artifacts are deserialized.
            bundle = joblib.load(path)
            summary, scored = score_batch(bundle, read_csv_bytes(content), model, calibration)
            counts, edges = np.histogram(scored['probability'].to_numpy(), bins=np.linspace(0, 1, 11))
            batch = {**summary, 'id': new_id(), 'run_id': identifier, 'run_name': run['name'],
                     'model': model, 'calibration': calibration, 'created_at': now(),
                     'histogram': {'edges': edges.tolist(), 'counts': counts.tolist()}}
            output = store.path('predictions', batch['id'], '.csv')
            output.write_bytes(safe_csv(scored))
            os.chmod(output, 0o600)
            store.put('predictions', batch)
            store.add_activity('batch_scored', f'{run["name"]} · {batch["rows"]} 行')
            return batch
        return await run_in_threadpool(work)

    @app.get('/api/predictions', dependencies=protected)
    def predictions():
        return {'items': store.list('predictions')}

    @app.get('/api/predictions/{identifier}/download', dependencies=protected)
    def download_prediction(identifier: str):
        find('predictions', identifier)
        return FileResponse(store.path('predictions', identifier, '.csv'), media_type='text/csv', filename='aware-scores.csv')

    @app.get('/{requested_path:path}')
    def frontend(requested_path: str):
        if requested_path.startswith('api/'):
            raise HTTPException(404, 'API 不存在。')
        dist = ROOT / 'web/dist'
        path = (dist / requested_path).resolve()
        if not path.is_relative_to(dist.resolve()):
            raise HTTPException(404)
        if path.is_file():
            return FileResponse(path)
        if (dist / 'index.html').exists():
            return FileResponse(dist / 'index.html')
        return JSONResponse({'detail': '前端尚未构建，请运行 cd web && npm ci && npm run build。'}, status_code=503)

    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_UPLOAD + 65536)
    return app
