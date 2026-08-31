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
from web_app import create_default_router, create_server


class _DummyAdapter:
    def __init__(self, config):
        self.config = config

    def predict(self, **inputs):
        return {"task": self.config.task_id, "received": sorted(inputs)}


class _FailingAdapter:
    def __init__(self, config):
        self.config = config

    def predict(self, **inputs):
        raise RuntimeError("synthetic inference failure")


class WebRoutingTest(unittest.TestCase):
    def test_three_task_input_contracts_remain_distinct(self):
        classification = get_task_config("classification")
        refseg = get_task_config("refseg")
        self.assertEqual(classification.title, "场景分类")
        self.assertEqual(refseg.title, "语义分割")
        self.assertEqual([field.key for field in classification.inputs], ["image"])
        self.assertEqual(
            [field.key for field in get_task_config("counting").inputs],
            ["image", "target_class"],
        )
        self.assertEqual([field.key for field in refseg.inputs], ["image", "text"])

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

    def test_default_router_exposes_only_refseg_as_ready(self):
        statuses = {
            task["task_id"]: task["interface_status"]
            for task in create_default_router().list_tasks()
        }
        self.assertEqual(
            statuses,
            {"classification": "pending", "counting": "pending", "refseg": "ready"},
        )

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
                self.assertIn("遥感图-文可解释轻量化多任务智能解译系统", html)
                self.assertNotIn(">HFSA<", html)
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
                {
                    "task_id": "counting",
                    "inputs": {"image": "scene.png", "target_class": "windmill"},
                }
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

    def test_predict_endpoint_returns_structured_inference_error(self):
        router = TaskRouter()
        router.register_adapter("classification", _FailingAdapter)
        server = create_server("127.0.0.1", 0, router=router)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        request = Request(
            f"http://{host}:{port}/api/predict",
            data=json.dumps(
                {"task_id": "classification", "inputs": {"image": "scene.png"}}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 500)
            payload = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(payload["status"], "inference_error")
            self.assertIn("synthetic inference failure", payload["message"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
