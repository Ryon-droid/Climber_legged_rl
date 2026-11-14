"""Start the local console and open it in the default browser."""
import argparse
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser

HERE = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765'


def available():
    try:
        with urllib.request.urlopen(URL, timeout=1) as response:
            return response.headers.get('Server', '').startswith('ClimberConsole/')
    except (OSError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not available():
        started = subprocess.run(['systemctl', '--user', 'start', 'climber-training-console.service'],
                                 capture_output=True, text=True)
        if started.returncode:
            raise RuntimeError('无法启动训练上位机：'+started.stderr)
        for _ in range(30):
            if available():
                break
            time.sleep(.2)
        else:
            raise RuntimeError('服务尚未响应，请检查 systemctl --user status climber-training-console')
    if not args.no_browser:
        webbrowser.open(URL)
    print(URL)


if __name__ == '__main__':
    main()
