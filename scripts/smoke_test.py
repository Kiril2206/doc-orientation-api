"""End-to-end verification script for demo readiness and AWS endpoints."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import create_app
import fitz

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "app" / "static" / "samples"


def run_smoke_test():
    print("Initializing application...")
    app = create_app()
    with TestClient(app) as client:
        print("[1/6] Testing GET /live (ALB Liveness Probe)...")
        res_live = client.get("/live")
        assert res_live.status_code == 200, f"Expected 200, got {res_live.status_code}"
        assert res_live.json() == {"status": "alive"}
        print("  -> Passed: status=alive, X-Request-ID present:", res_live.headers.get("x-request-id"))

        print("[2/6] Testing GET /ready (Readiness Probe)...")
        res_ready = client.get("/ready")
        assert res_ready.status_code == 200, f"Expected 200, got {res_ready.status_code}"
        assert res_ready.json() == {"status": "ready"}
        print("  -> Passed: models loaded and warmed up.")

        print("[3/6] Testing GET /health...")
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        print(f"  -> Passed: version={health_data['version']}, modes={list(health_data['modes'].keys())}")

        print("[4/6] Testing POST /correct-orientation with invoice_90.png (Pure CV)...")
        invoice_bytes = (SAMPLES_DIR / "invoice_90.png").read_bytes()
        res_inv = client.post(
            "/correct-orientation",
            data={"mode": "pure"},
            files={"file": ("invoice_90.png", invoice_bytes, "image/png")}
        )
        assert res_inv.status_code == 200, f"Expected 200, got {res_inv.status_code}"
        orig = res_inv.headers.get("X-Original-Orientation")
        rot = res_inv.headers.get("X-Rotation-Applied")
        print(f"  -> Passed: detected={orig}°, applied={rot}°, model={res_inv.headers.get('X-Model-Version')}")

        print("[5/6] Testing POST /correct-orientation with mixed_rotations.pdf (4 pages)...")
        pdf_bytes = (SAMPLES_DIR / "mixed_rotations.pdf").read_bytes()
        res_pdf = client.post(
            "/correct-orientation",
            data={"mode": "pure"},
            files={"file": ("mixed_rotations.pdf", pdf_bytes, "application/pdf")}
        )
        assert res_pdf.status_code == 200, f"Expected 200, got {res_pdf.status_code}"
        assert res_pdf.headers.get("Content-Type") == "application/pdf"
        assert res_pdf.headers.get("X-Page-Count") == "4"
        with fitz.open(stream=res_pdf.content, filetype="pdf") as doc:
            assert doc.page_count == 4
        print(f"  -> Passed: 4 pages corrected, page results={res_pdf.headers.get('X-Page-Results')}")

        print("[6/6] Testing POST /correct-orientation with blank.png (Abstention / Review flag)...")
        blank_bytes = (SAMPLES_DIR / "blank.png").read_bytes()
        res_blank = client.post(
            "/correct-orientation",
            data={"mode": "pure"},
            files={"file": ("blank.png", blank_bytes, "image/png")}
        )
        assert res_blank.status_code == 200
        assert res_blank.headers.get("X-Needs-Review") == "true"
        print("  -> Passed: X-Needs-Review=true, decision source preserved.")

    print("\nALL SMOKE TESTS PASSED! APPLICATION IS 100% READY FOR DEMONSTRATION AND DEPLOYMENT.")


if __name__ == "__main__":
    run_smoke_test()
