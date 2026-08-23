# -*- coding: utf-8 -*-
"""nh3-platform 独立仓库冒烟测试"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

def main():
    from fastapi.testclient import TestClient
    from server.main import app, scan_models, MODEL_REGISTRY

    reg = scan_models()
    print("models:", {k: v["steps"] for k, v in reg.items()})
    assert "output/ppo_agent" in reg, "默认模型缺失"

    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200 and "调度平台" in r.text
        print("index OK")

        r = client.get("/api/cases")
        cases = r.json()["cases"]
        print("cases:", [c["name"] for c in cases])

        csv = open("samples/sample_summer_sunny.csv", "rb").read()
        r = client.post("/api/predict", files={"file": ("s.csv", csv, "text/csv")})
        j = r.json()
        assert r.status_code == 200 and len(j["steps"]) == 96
        print(f"predict OK: model={j['model']} reward={j['summary']['reward_wan']}wan "
              f"util={j['summary']['utilization_pct']}%")

    print("\nSMOKE TEST PASS")
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
