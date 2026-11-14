"""Integration checks against the running local console; no training writes."""
import json
import re
import unittest
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8765'


class ConsoleContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with urllib.request.urlopen(BASE) as response:
            cls.html = response.read().decode()
        cls.token = re.search(r'name="climber-token" content="([^"]+)"', cls.html)[1]

    def request(self, path, extra=None):
        headers = {'X-Climber-Token': self.token, **(extra or {})}
        return urllib.request.urlopen(urllib.request.Request(BASE+path, headers=headers), timeout=30)

    def test_connected_status_contains_selected_checkpoint(self):
        with self.request('/api/status') as response:
            data = json.load(response)
        self.assertTrue(data['connected'])
        self.assertIn('push_', data['run'])
        self.assertIn('selected_summary', data)
        self.assertIn('runs', data)
        self.assertGreater(len(data['runs']), 0)
        selected = data.get('selected_checkpoint')
        if selected:
            self.assertTrue(selected.endswith('.zip'))

    def test_api_requires_request_token(self):
        for method in ('GET', 'POST'):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(BASE+'/api/terminal/connect',
                    data=b'{}' if method == 'POST' else None, method=method))
            self.assertEqual(error.exception.code, 403)

    def test_host_is_loopback_only(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/status', {'Host': 'other.example'})
        self.assertEqual(error.exception.code, 403)

    def test_unknown_run_cannot_read_arbitrary_path(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/status?run=..%2F..%2Fetc')
        self.assertEqual(error.exception.code, 400)

    def test_media_supports_seek_and_invalid_ranges(self):
        with self.request('/api/jobs') as response:
            jobs = json.load(response)['jobs']
        videos = [job for job in jobs if job['state'] == 'completed']
        self.assertTrue(videos, '先通过界面生成一段仿真视频')
        url = videos[-1]['media_url']
        with self.request(url, {'Range': 'bytes=0-31'}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(len(response.read()), 32)
            self.assertTrue(response.headers['Content-Range'].startswith('bytes 0-31/'))
            length = int(response.headers['Content-Range'].split('/')[-1])
        with self.request(url, {'Range':'bytes=-16'}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(len(response.read()), 16)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(url, {'Range': f'bytes={length+1}-'})
        self.assertEqual(error.exception.code, 416)


if __name__ == '__main__':
    unittest.main()
