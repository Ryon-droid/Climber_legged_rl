"""Local training console: SSH status, a persistent remote terminal and videos.

Standard library only. Bind to loopback; SSH keys stay on this computer.
"""
import argparse
import atexit
import codecs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import pty
import re
import secrets
import shlex
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ANSI = re.compile(r'\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()][A-Z0-9]|\x1b[=>]')


class RemoteTerminal:
    def __init__(self, app):
        self.lock = threading.Lock()
        self.raw = ''
        master, slave = pty.openpty()
        self.master = master
        root = shlex.quote(app.config['remote_root'])
        venv = shlex.quote(str(PurePosixPath(app.config['remote_python']).parent))
        command = f'cd {root} && export PATH={venv}:"$PATH" && exec bash --noprofile --norc -i'
        try:
            self.process = subprocess.Popen(app.ssh_args(tty=True)+[command], stdin=slave,
                    stdout=slave, stderr=slave, start_new_session=True)
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            while True:
                part = os.read(self.master, 8192)
                if not part:
                    break
                with self.lock:
                    self.raw = (self.raw + decoder.decode(part))[-250000:]
        except OSError:
            pass

    def state(self):
        with self.lock:
            text = ANSI.sub('', self.raw).replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x07\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return dict(text=text, alive=self.process.poll() is None, exit_code=self.process.poll())

    def send(self, text):
        if self.process.poll() is not None:
            raise ValueError('终端已断开，请重新连接')
        os.write(self.master, text.encode())

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        try:
            os.close(self.master)
        except OSError:
            pass


class Console:
    def __init__(self, config):
        self.config = config
        self.token = secrets.token_urlsafe(32)
        self.cache = HERE/'cache'
        self.cache.mkdir(exist_ok=True)
        self.status_cache = {}
        self.status_lock = threading.Lock()
        self.terminal = None
        self.terminal_lock = threading.Lock()
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.media = {}
        for path in sorted(self.cache.glob('*')):
            if path.suffix not in ('.mp4', '.webm') or not re.fullmatch(r'[a-f0-9]{16}', path.stem):
                continue
            try:
                manifest = path.with_suffix('.job.json')
                if manifest.exists():
                    job = json.loads(manifest.read_text())
                else:
                    meta_path = path.with_suffix('.json')
                    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
                    job = dict(id=path.stem, kind='download', state='completed', started=path.stat().st_mtime,
                               finished=path.stat().st_mtime, options={}, metadata=metadata,
                               remote_path='outputs/training_console_previews/'+path.name)
                job.update(media_url='/media/'+path.stem, local_path=str(path))
                self.jobs[path.stem] = job
                self.media[path.stem] = path
            except (ValueError, OSError):
                continue
        self.jobs = dict(sorted(self.jobs.items(), key=lambda pair: pair[1].get('started', 0)))

    def ssh_args(self, tty=False):
        return ['ssh', '-i', self.config['ssh_key'], '-p', str(self.config.get('ssh_port', 22)),
                '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                '-o', 'ServerAliveInterval=15', '-o', 'LogLevel=ERROR',
                *(['-tt'] if tty else []), self.config['ssh_host']]

    def ssh(self, command, *, input_text=None, timeout=20):
        result = subprocess.run(self.ssh_args()+[command], input=input_text, capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-2500:] or result.stdout.strip()[-2500:] or 'SSH 连接失败')
        return result.stdout

    def status(self, run=None):
        if run and not re.fullmatch(r'[A-Za-z0-9_.-]+', run):
            raise ValueError('无效训练名称')
        key = run or ''
        with self.status_lock:
            prior = self.status_cache.get(key)
            if prior and time.monotonic()-prior[0] < 4:
                return prior[1]
            try:
                # Collector contains only reads, no training or configuration writes.
                source = (HERE/'remote_status.py').read_text()
                source = source.replace("ROOT = Path('/home/climber/Climber_legged_rl')",
                                        'ROOT = Path('+repr(self.config['remote_root'])+')')
                command = shlex.join(['python3', '-', *([run] if run else [])])
                result = json.loads(self.ssh(command, input_text=source))
                result['host_label'] = self.config['label']
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                result = dict(prior[1]) if prior else {}
                result.update(connected=False, error=str(error), host_label=self.config['label'])
            self.status_cache[key] = (time.monotonic(), result)
            return result

    def get_terminal(self, create=False):
        with self.terminal_lock:
            if create and (self.terminal is None or self.terminal.process.poll() is not None):
                if self.terminal:
                    self.terminal.close()
                self.terminal = RemoteTerminal(self)
            return self.terminal

    def start_job(self, kind, options):
        with self.jobs_lock:
            if any(job['state'] == 'running' for job in self.jobs.values()):
                raise ValueError('已有视频任务进行中，请等待完成')
            if kind == 'preview':
                run = options.get('run')
                status = self.status(run)
                if not status.get('connected') or not any(row['name'] == run for row in status.get('runs', [])):
                    raise ValueError('请先选择已连接的训练记录')
                mode = options.get('mode', 'standing')
                force = float(options.get('force', 0))
                seed = int(options.get('seed', 840000))
                if mode not in ('standing', 'folded') or not 0 <= force <= 150 or not 0 <= seed <= 2147483647:
                    raise ValueError('预览参数不正确')
                options = dict(run=run, mode=mode, force=force, seed=seed)
            elif kind == 'download':
                with self.status_lock:
                    videos = [video for _, state in self.status_cache.values() for video in state.get('videos', [])]
                if not any(video['path'] == options.get('path') and video['size_mb'] < 500 for video in videos):
                    raise ValueError('请从视频列表选择文件（单个小于 500 MB）')
            else:
                raise ValueError('未知视频任务')
            identity = secrets.token_hex(8)
            job = dict(id=identity, kind=kind, state='running', started=time.time(), options=options)
            self.jobs[identity] = job
        threading.Thread(target=self._video_job, args=(job,), daemon=True).start()
        return job.copy()

    def _fetch_file(self, remote, local):
        result = subprocess.run(['scp', '-i', self.config['ssh_key'], '-P', str(self.config.get('ssh_port', 22)),
                '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                self.config['ssh_host']+':'+remote, str(local)], capture_output=True, text=True, timeout=150)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-2500:] or '视频传输失败')

    def _video_job(self, job):
        try:
            root = PurePosixPath(self.config['remote_root'])
            metadata = None
            if job['kind'] == 'preview':
                options = job['options']
                relative = f'outputs/training_console_previews/{job["id"]}.mp4'
                remote = str(root/relative)
                command = shlex.join(['nice', '-n', '10', self.config['remote_python'],
                        str(root/'scripts/training_console_preview.py'), '--run', options['run'],
                        '--output', remote, '--mode', options['mode'], '--force', str(options['force']),
                        '--seed', str(options['seed'])])
                output = self.ssh(command, timeout=300)
                metadata = json.loads(output.strip().splitlines()[-1])
            else:
                relative = job['options']['path']
                path = PurePosixPath(relative)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'outputs':
                    raise ValueError('无效视频路径')
                remote = str(root/path)
                try:
                    metadata = json.loads(self.ssh(shlex.join(['cat', str(PurePosixPath(remote).with_suffix('.json'))])))
                    metadata.pop('trace', None)
                except (ValueError, RuntimeError):
                    pass
            suffix = PurePosixPath(remote).suffix
            local = self.cache/(job['id']+suffix)
            self._fetch_file(remote, local)
            self.media[job['id']] = local
            if metadata:
                (self.cache/(job['id']+'.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            with self.jobs_lock:
                job.update(state='completed', finished=time.time(), media_url='/media/'+job['id'],
                           metadata=metadata, remote_path=relative, local_path=str(local))
                local.with_suffix('.job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2))
            with self.status_lock:
                self.status_cache.clear()
        except Exception as error:
            with self.jobs_lock:
                job.update(state='failed', finished=time.time(), error=str(error))

    def close(self):
        if self.terminal:
            self.terminal.close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'ClimberConsole/1.0'

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith(('GET /api/status', 'GET /api/terminal', 'GET /api/jobs')):
            return
        super().log_message(fmt, *args)

    @property
    def app(self):
        return self.server.app

    def reply(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def trusted_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def authorized(self):
        return self.trusted_host() and secrets.compare_digest(self.headers.get('X-Climber-Token', ''), self.app.token)

    def do_GET(self):
        try:
            if not self.trusted_host():
                self.reply(dict(error='Invalid host'), 403)
                return
            url = urlparse(self.path)
            path = url.path
            if path.startswith('/api/'):
                if not self.authorized():
                    self.reply(dict(error='请从本机控制台打开'), 403)
                    return
                if path == '/api/status':
                    self.reply(self.app.status(parse_qs(url.query).get('run', [None])[0]))
                elif path == '/api/terminal':
                    terminal = self.app.get_terminal()
                    self.reply(terminal.state() if terminal else dict(text='', alive=False))
                elif path == '/api/jobs':
                    with self.app.jobs_lock:
                        jobs = [row.copy() for row in self.app.jobs.values()]
                    self.reply(dict(jobs=jobs))
                else:
                    self.reply(dict(error='未知接口'), 404)
                return
            if path.startswith('/media/'):
                identity = path.removeprefix('/media/')
                local = self.app.media.get(identity)
                if not local or not local.is_file():
                    self.reply(dict(error='视频不存在，请重新加载'), 404)
                    return
                self.send_file(local, allow_range=True)
                return
            files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if path not in files:
                self.reply(dict(error='Not found'), 404)
                return
            local = HERE/files[path]
            if path == '/':
                content = local.read_text().replace('__CLIMBER_TOKEN__', self.app.token).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Frame-Options', 'DENY')
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_file(local)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.reply(dict(error=str(error)), 400)
        except Exception as error:
            self.reply(dict(error=str(error)), 500)

    def send_file(self, local, allow_range=False):
        length = local.stat().st_size
        start, end = 0, length-1
        requested = self.headers.get('Range') if allow_range else None
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), end) if last else end
            else:
                start = max(0, length-int(last))
            if start >= length or start > end:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{length}')
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', mimetypes.guess_type(str(local))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if allow_range:
            self.send_header('Accept-Ranges', 'bytes')
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
        self.end_headers()
        with local.open('rb') as stream:
            stream.seek(start)
            remaining = end-start+1
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        try:
            if not self.authorized():
                self.reply(dict(error='本机访问验证失败，请刷新页面'), 403)
                return
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 20000:
                self.reply(dict(error='请求过长或为空'), 400)
                return
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/terminal/connect':
                self.reply(self.app.get_terminal(create=True).state())
            elif self.path == '/api/terminal/input':
                value = data.get('text', '')
                if not isinstance(value, str) or len(value) > 16000:
                    raise ValueError('指令过长')
                self.app.get_terminal(create=True).send(value)
                self.reply(dict(ok=True))
            elif self.path == '/api/preview':
                self.reply(self.app.start_job('preview', data), 202)
            elif self.path == '/api/videos/fetch':
                self.reply(self.app.start_job('download', data), 202)
            else:
                self.reply(dict(error='未知接口'), 404)
        except (ValueError, RuntimeError) as error:
            self.reply(dict(error=str(error)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.reply(dict(error=str(error)), 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    app = Console(config)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.app = app
    atexit.register(app.close)
    print(f'Climber 上位机：http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
