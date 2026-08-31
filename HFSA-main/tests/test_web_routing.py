import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.routing import TaskInputError, TaskInterfaceUnavailableError, TaskRouter, get_task_config
from web_app import create_server


class _DummyAdapter:
    def __init__(self, config):
        self.config = config

    def predict(self, **inputs):
        return {"task": self.config.task_id, "received": sorted(inputs)}


class WebRoutingTest(unittest.TestCase):
    def test_three_task_input_contracts_remain_distinct(self):
        self.assertEqual([field.key for field in get_task_config("classification").inputs], ["image"])
        self.assertEqual(
            [field.key for field in get_task_config("counting").inputs],
            ["image", "target_class"],
        )
        self.assertEqual([field.key for field in get_task_config("refseg").inputs], ["image", "text"])

    def test_task_catalog_is_json_serializable_and_project_relative(self):
        router = TaskRouter()
        payload = json.loads(json.dumps(router.list_tasks(), ensure_ascii=False))
        self.assertEqual(len(payload), 3)
        for task in payload:
            self.assertFalse(Path(task["checkpoint"]).is_absolute())
            self.assertEqual(task["interface_status"], "pending")

    def test_router_validates_inputs_before_interface_lookup(self):
        router = TaskRouter()
        with self.assertRaisesRegex(TaskInputError, "要统计的目标"):
            router.predict("counting", {"image": "scene.png"})
        with self.assertRaises(TaskInterfaceUnavailableError):
            router.predict("classification", {"image": "scene.png"})

    def test_router_supports_late_adapter_registration(self):
        router = TaskRouter()
        router.register_adapter("classification", _DummyAdapter)
        result = router.predict("classification", {"image": "scene.png"})
        self.assertEqual(result, {"task": "classification", "received": ["image"]})
        self.assertEqual(router.list_tasks()[0]["interface_status"], "loaded")

    def test_local_web_server_exposes_page_and_task_api(self):
        server = create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            with urlopen(f"http://{host}:{port}/api/health", timeout=3) as response:
                self.assertEqual(json.load(response)["status"], "ok")
            with urlopen(f"http://{host}:{port}/api/tasks", timeout=3) as response:
                self.assertEqual(len(json.load(response)["tasks"]), 3)
            with urlopen(f"http://{host}:{port}/", timeout=3) as response:
                html = response.read().decode("utf-8")
                self.assertIn("HFSA 遥感智能分析", html)
                self.assertNotIn('class="steps"', html)
                self.assertRegex(html, r'<section class="workspace" id="workspace"[^>]*hidden>')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_predict_endpoint_reports_pending_interface(self):
        server = create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        request = Request(
            f"http://{host}:{port}/api/predict",
            data=json.dumps(
                {"task_id": "refseg", "inputs": {"image": "scene.png", "text": "the windmill"}}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 503)
            payload = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(payload["status"], "interface_pending")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
