"""Synthetic assets only: no competition images, tasks, or gold labels."""

import hashlib
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_adapter import DemoModel  # noqa: E402
from server import APIServer, AssetStore  # noqa: E402


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        assets = self.package / "assets"
        assets.mkdir()
        self.asset_id = "a" * 32
        payload = b"synthetic test image bytes, never distributed"
        (assets / self.asset_id).write_bytes(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        self.manifest = {
            "format": "rsu-sealed-images-v1",
            "dataset_id": "rsu-dev100-v2",
            "assets": [{
                "asset_id": self.asset_id, "sha256": self.digest,
                "size": len(payload), "mime_type": "image/png",
            }],
        }
        (self.package / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.server = APIServer(("127.0.0.1", 0), AssetStore(self.package), DemoModel(), "x" * 32)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def call(self, path: str, data: dict | None = None, token: str = "x" * 32) -> tuple[int, dict]:
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(
            self.origin + path,
            data=body,
            headers={
                "Authorization": "Bearer " + token,
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def request_data(self) -> dict:
        return {
            "protocol_version": "2.0", "request_id": "test:1", "item_id": "TEST_1",
            "question": "Synthetic question", "answer_type": "short_text",
            "images": [{"asset_id": self.asset_id, "sha256": self.digest, "mime_type": "image/png"}],
            "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
        }

    def test_health_and_prediction(self) -> None:
        status, health = self.call("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(health, {
            "status": "ready", "protocol_version": "2.0", "dataset_id": "rsu-dev100-v2"
        })
        status, result = self.call("/v1/predict", self.request_data())
        self.assertEqual(status, 200)
        self.assertEqual(result, {
            "protocol_version": "2.0", "request_id": "test:1", "item_id": "TEST_1", "answer": "Yes"
        })

    def test_rejects_bad_token_and_bad_asset_hash(self) -> None:
        self.assertEqual(self.call("/healthz", token="bad")[0], 401)
        request = self.request_data()
        request["images"][0]["sha256"] = "0" * 64
        self.assertEqual(self.call("/v1/predict", request)[0], 422)

    def test_rejects_modified_local_asset(self) -> None:
        path = self.package / "assets" / self.asset_id
        path.write_bytes(b"X" * path.stat().st_size)
        status, result = self.call("/v1/predict", self.request_data())
        self.assertEqual((status, result["error"]), (422, "asset_sha256_mismatch"))

    def test_bbox_answer_is_pixel_list(self) -> None:
        request = self.request_data()
        request["answer_type"] = "bbox"
        request["response_constraint"] = {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4}
        request["image_width"], request["image_height"] = 640, 480
        status, result = self.call("/v1/predict", request)
        self.assertEqual(status, 200)
        self.assertEqual(result["answer"], [0, 0, 640, 480])


if __name__ == "__main__":
    unittest.main()
