# -*- coding: utf-8 -*-
"""轻量页面体检（无需浏览器）：工业终端 v5 关键要素与禁忌项"""
import urllib.request, urllib.error, json, re, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

html = urllib.request.urlopen("http://127.0.0.1:8000/", timeout=5).read().decode("utf-8")
print("HTTP OK, page KB:", len(html) // 1024)

checks = {
    "标题=调度控制系统": "氨电联产 AGC 调度控制系统" in html,
    "系统状态栏": 'class="sysbar"' in html,
    "单线图": 'class="sld"' in html,
    "AGC指令面板": 'id="agcLevel"' in html,
    "事件流": 'id="evtMini"' in html,
    "四页面齐全": all(p in html for p in
                 ["调度总览", "实时趋势", "典型日复盘", "事件记录"]),
    "无emoji": not re.findall(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]", html),
    "无设备字样": not re.search(r"计算设备|CUDA|cuda", html),
    "无上传控件": 'type="file"' not in html,
    "无模型下拉框": "modelSelect" not in html,
    "无侧边栏": ".sidebar" not in html,
}
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)

cases = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/cases", timeout=5).read().decode("utf-8"))
models = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/models", timeout=15).read().decode("utf-8"))
print("cases:", len(cases["cases"]), "| models:", models["n_models"])
if len(cases["cases"]) < 4: bad.append("案例不足4个")
if models["n_models"] < 1: bad.append("无已注册模型")
if "device" in models: bad.append("API泄露device字段")

print("\n结论:", "全部通过" if not bad else f"问题 {bad}")
sys.exit(0 if not bad else 1)
