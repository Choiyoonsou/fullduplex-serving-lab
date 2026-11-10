"""Experiment console: job queue + GPU scheduler + run browser over scripts/run_all.sh.

  python console/server.py                      # http://127.0.0.1:8800  (scripts/serve_console.sh)

Standard library only. Binds to 127.0.0.1 by default: anyone who can reach the
port can launch GPU jobs, so use an SSH tunnel rather than --host 0.0.0.0.

Jobs: validated against console/experiments.json, queued FIFO, started when
enough free GPU pairs exist (a GPU is free if no console job holds it and it
uses < FREE_MB of memory, so runs started from the shell are respected).
State persists in console/state/jobs.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
CONSOLE = ROOT / "console"
STATE = CONSOLE / "state" / "jobs.json"
FREE_MB = 1500
REG = json.loads((CONSOLE / "experiments.json").read_text())
EXPS = {e["id"]: e for e in REG["experiments"]}
MEMBERS = json.loads((ROOT / "members.json").read_text())["members"]
RATIOS = [0.2, 0.4, 0.6]
# mean frames per dialog (measured): per-variant forward cost ~ 0.75 s * (S/1000)^1.15 on a T4 pair
FRAMES = {"dev": {0.2: 1150, 0.4: 1600, 0.6: 2300}, "test": {0.2: 850, 0.4: 1150, 0.6: 1700}}

lock = threading.Lock()
jobs: list[dict] = []
procs: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------- utilities
def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def save_jobs() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=1))
    tmp.replace(STATE)


def load_jobs() -> None:
    global jobs
    if STATE.exists():
        jobs = json.loads(STATE.read_text())
        for j in jobs:   # processes do not survive a console restart
            if j["status"] == "running":
                j["status"], j["finished"] = "lost", now()
                j["error"] = "콘솔이 재시작되어 실행 상태를 추적하지 못했습니다. run 폴더를 확인하세요."


def gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return []
    res = []
    for line in out.strip().splitlines():
        i, name, used, total, util = [x.strip() for x in line.split(",")]
        res.append({"index": int(i), "name": name, "mem_used": int(used), "mem_total": int(total), "util": int(util)})
    return res


def gpu_pairs(gs: list[dict]) -> list[tuple[int, int]]:
    if not gs:
        return []
    if min(g["mem_total"] for g in gs) >= 22000:
        return [(g["index"], g["index"]) for g in gs]
    idx = [g["index"] for g in gs]
    return [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]


def held_gpus() -> dict[int, str]:
    held = {}
    for j in jobs:
        if j["status"] == "running":
            for a, b in j.get("pairs", []):
                held[a] = held[b] = j["id"]
    return held


def free_pairs(gs: list[dict]) -> list[tuple[int, int]]:
    held, used = held_gpus(), {g["index"]: g["mem_used"] for g in gs}
    return [(a, b) for a, b in gpu_pairs(gs)
            if a not in held and b not in held and used.get(a, 1e9) < FREE_MB and used.get(b, 1e9) < FREE_MB]


# ---------------------------------------------------------------- job spec
class SpecError(ValueError):
    pass


def validate(spec: dict) -> dict:
    """Turn a form submission into a safe, fully specified job (raises SpecError)."""
    exp = EXPS.get(spec.get("exp"))
    if not exp or exp.get("status") != "ready":
        raise SpecError("실행할 수 없는 실험 종류입니다.")
    member = spec.get("member")
    if member not in MEMBERS:
        raise SpecError("담당자를 선택하세요.")
    tag = str(spec.get("tag") or exp["id"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", tag):
        raise SpecError("태그는 영문, 숫자, -, _ 만 쓸 수 있습니다(최대 40자).")
    split = spec.get("split", "dev")
    if split not in ("dev", "test"):
        raise SpecError("split은 dev 또는 test입니다.")
    if split == "test" and spec.get("confirm_test") is not True:
        raise SpecError("test 셋은 최종 평가용입니다. 확인란을 체크해야 실행됩니다.")
    try:
        n = int(spec.get("n_dialogs", 100))
    except (TypeError, ValueError):
        raise SpecError("대화 수는 정수여야 합니다.") from None
    if not 1 <= n <= 100:
        raise SpecError("대화 수는 1~100입니다.")
    ratios = [r for r in RATIOS if r in [float(x) for x in spec.get("ratios", RATIOS)]]
    if not ratios:
        raise SpecError("침묵 비율을 하나 이상 고르세요.")
    n_pairs = int(spec.get("n_pairs", 0))
    if not 0 <= n_pairs <= 8:
        raise SpecError("GPU 쌍 수가 올바르지 않습니다.")
    args: list[str] = []
    params = {}
    for p in exp["params"]:
        v = (spec.get("params") or {}).get(p["key"], p["default"])
        if p["type"] == "multi":
            v = [o for o in p["options"] if o in v or str(o) in [str(x) for x in v]]
            if not v:
                raise SpecError(f"'{p['label']}'을(를) 하나 이상 고르세요.")
            args += [p["cli"], *map(str, v)]
        elif p["type"] == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise SpecError(f"'{p['label']}'은(는) 정수여야 합니다.") from None
            if not p["min"] <= v <= p["max"]:
                raise SpecError(f"'{p['label']}'은(는) {p['min']}~{p['max']} 사이여야 합니다.")
            args += [p["cli"], str(v)]
        elif p["type"] == "bool":
            v = bool(v)
            if v:
                args.append(p["cli"])
        params[p["key"]] = v
    note = str(spec.get("note") or "")[:300].replace("\n", " ")
    env = {"MEMBER": member, "TAG": tag, "NOTE": note, "SPLIT": split, "EXP": exp["id"],
           "N_DIALOGS": str(n), "RATIOS": " ".join(str(r) for r in ratios),
           "SKIP_VERIFY": "1" if spec.get("skip_verify", True) else "0"}
    if split == "test":
        env["CONFIRM_TEST"] = "1"
    return {"exp": exp["id"], "member": member, "tag": tag, "note": note, "split": split, "n_dialogs": n,
            "ratios": ratios, "n_pairs": n_pairs, "params": params, "args": args, "env": env}


def n_variants(job: dict) -> int:
    p = job["params"]
    if job["exp"] == "kv_policy":
        n = 2 + len(p["budgets"]) * len(p["kinds"])
        if p.get("int4"):
            n += 1 + len(p["budgets"]) * sum(k in p["kinds"] for k in ("B2", "P2"))
        return n
    if job["exp"] == "rvq":
        return 1 + len(p["rvq_ks"])
    return 1 + len(p["kv_bits"])


def estimate(job: dict, pairs: int) -> dict:
    var = n_variants(job)
    per_file = {r: 0.75 * (FRAMES[job["split"]][r] / 1000) ** 1.15 * var * 0.95 for r in job["ratios"]}
    nll = job["n_dialogs"] * sum(per_file.values()) / max(pairs, 1)
    prep = sum(1 for r in job["ratios"] for _ in range(job["n_dialogs"])
               if not (ROOT / f"data/prepared/r{r:.1f}").exists()) * 3
    overhead = 90 + (0 if job["env"]["SKIP_VERIFY"] == "1" else 300) + prep
    return {"variants": var, "forwards": var * job["n_dialogs"] * len(job["ratios"]),
            "seconds": int(nll + overhead), "pairs": pairs}


def command(job: dict, pairs: list | None = None) -> str:
    env = dict(job["env"])
    if pairs:
        env["GPU_PAIRS"] = " ".join(f"{a},{b}" for a, b in pairs)
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " scripts/run_all.sh " + \
        " ".join(shlex.quote(a) for a in job["args"])


# ---------------------------------------------------------------- scheduler
def start(job: dict, pairs: list) -> None:
    run_dir = ROOT / job["run_dir"]
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ | job["env"] | {"GPU_PAIRS": " ".join(f"{a},{b}" for a, b in pairs),
                                     "NEW_RUN_DIR": job["run_dir"], "HF_HOME": os.environ.get("HF_HOME", str(ROOT / "hf_cache"))}
    env.pop("RUN_DIR", None)
    log = open(run_dir / "logs" / "console.log", "a")
    p = subprocess.Popen(["bash", "scripts/run_all.sh", *job["args"]], cwd=ROOT, env=env, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    procs[job["id"]] = p
    job.update(status="running", started=now(), pairs=[list(x) for x in pairs], pid=p.pid,
               cmd=command(job, pairs))


def set_meta_status(run_dir: str, status: str) -> None:
    mp = ROOT / run_dir / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        if m.get("status") == "running":
            m["status"], m["finished"] = status, now()
            mp.write_text(json.dumps(m, ensure_ascii=False, indent=2))


def scheduler() -> None:
    while True:
        with lock:
            changed = False
            for j in jobs:
                p = procs.get(j["id"])
                if j["status"] == "running" and p and p.poll() is not None:
                    j["status"] = "done" if p.returncode == 0 else ("cancelled" if j.get("cancel") else "failed")
                    j["finished"], j["returncode"] = now(), p.returncode
                    if j["status"] != "done":
                        set_meta_status(j["run_dir"], j["status"])
                    procs.pop(j["id"], None)
                    changed = True
            queued = [j for j in jobs if j["status"] == "queued"]
            if queued:
                gs = gpus()
                free = free_pairs(gs)
                head = queued[0]   # FIFO: the head waits until it fits (no starvation)
                want = head["n_pairs"] or len(free)
                if free and len(free) >= max(want, 1):
                    start(head, free[:max(want, 1)])
                    changed = True
            if changed:
                save_jobs()
        time.sleep(3)


# ---------------------------------------------------------------- runs & progress
SHARD_RE = re.compile(r"^\[(\d+)/(\d+)\]")


def progress(run_dir: Path) -> dict:
    done = total = 0
    for f in sorted((run_dir / "logs").glob("nll_shard*.log")):
        last, shard_total = None, 0
        for line in f.read_text(errors="ignore").splitlines():
            m = SHARD_RE.match(line)
            if m:
                last = m
            else:
                mm = re.match(r"shard \d+/\d+: (\d+) files", line)
                if mm:
                    shard_total = int(mm.group(1))
        done += int(last.group(1)) if last else 0
        total += int(last.group(2)) if last else shard_total
    stage = ""
    cl = run_dir / "logs" / "console.log"
    if cl.exists():
        st = re.findall(r"^\[(\d)/4\] (.*)$", cl.read_text(errors="ignore"), re.M)
        if st:
            stage = f"{st[-1][0]}/4 {st[-1][1]}"
    return {"done": done, "total": total, "stage": stage}


def list_runs() -> list[dict]:
    out = []
    for mp in sorted(ROOT.glob("runs/*/*/meta.json")):
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        rd = mp.parent
        an = rd / "nll" / "analysis"
        out.append(m | {"run": str(rd.relative_to(ROOT)), "progress": progress(rd),
                        "has_analysis": an.exists(), "exp": m.get("exp", "kv_policy")})
    return sorted(out, key=lambda r: r.get("started") or "", reverse=True)


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def safe_run(path: str) -> Path:
    p = (ROOT / path).resolve()
    if not str(p).startswith(str((ROOT / "runs").resolve()) + os.sep) or not (p / "meta.json").exists():
        raise SpecError("run을 찾을 수 없습니다.")
    return p


def run_detail(path: str) -> dict:
    rd = safe_run(path)
    an = rd / "nll" / "analysis"
    md = lambda n: (an / n).read_text() if (an / n).exists() else None  # noqa: E731
    figs = [str(f.relative_to(ROOT)) for f in sorted(an.glob("*.png"))] if an.exists() else []
    return {"meta": json.loads((rd / "meta.json").read_text()), "run": path, "progress": progress(rd),
            "variants": read_csv(an / "variants.csv"), "table": read_csv(an / "table_all.csv"),
            "capacity": read_csv(an / "capacity.csv"), "hypotheses": read_csv(an / "hypotheses.csv"),
            "hypotheses_md": md("hypotheses.md"), "figures": figs}


def compare(paths: list[str], metric: str, ratio: float) -> dict:
    cols, rows = [], {}
    for path in paths:
        rd = safe_run(path)
        m = json.loads((rd / "meta.json").read_text())
        label = f"{m.get('name', m['member'])} · {m['tag']} ({m.get('split', 'dev')})"
        cols.append({"run": path, "label": label})
        for r in read_csv(rd / "nll" / "analysis" / "variants.csv"):
            if r["metric"] == metric and abs(float(r["ratio"]) - ratio) < 1e-6:
                rows.setdefault(r["policy"], {})[path] = {k: r[k] for k in ("mean_diff", "ci95_lo", "ci95_hi", "significant")}
    return {"cols": cols, "rows": [{"policy": k, "cells": v} for k, v in sorted(rows.items())]}


def tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()[-200_000:].decode(errors="ignore")
    data = re.sub(r"\r(?!\n)", "\n", data)   # tqdm progress lines
    return "\n".join(l for l in data.splitlines()[-n:] if "Warning" not in l and "warnings.warn" not in l)


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):  # quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n < 1_000_000 else {}

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/state":
                gs = gpus()
                held = held_gpus()
                for g in gs:
                    g["job"] = held.get(g["index"])
                with lock:
                    js = [dict(j) | ({"progress": progress(ROOT / j["run_dir"])} if j["status"] == "running" else {})
                          for j in reversed(jobs[-60:])]
                return self.send_json({"gpus": gs, "pairs": gpu_pairs(gs), "free_pairs": free_pairs(gs), "jobs": js,
                                       "members": MEMBERS, "time": now()})
            if u.path == "/api/experiments":
                return self.send_json(REG)
            if u.path == "/api/runs":
                return self.send_json(list_runs())
            if u.path == "/api/run":
                return self.send_json(run_detail(q.get("path", [""])[0]))
            if u.path == "/api/compare":
                return self.send_json(compare(q.get("runs", [""])[0].split(","), q.get("metric", ["text_nll_content"])[0],
                                              float(q.get("ratio", ["0.6"])[0])))
            if u.path == "/api/test_log":
                p = ROOT / "splits" / "test_log.jsonl"
                return self.send_json([json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else [])
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/log", u.path)
            if m:
                j = next((j for j in jobs if j["id"] == m.group(1)), None)
                if not j:
                    return self.send_json({"error": "없는 작업"}, 404)
                rd = ROOT / j["run_dir"] / "logs"
                n = min(int(q.get("n", ["200"])[0]), 2000)
                which = q.get("file", ["console"])[0]
                f = rd / ("console.log" if which == "console" else f"{re.sub(r'[^a-z0-9_]', '', which)}.log")
                return self.send_json({"text": tail(f, n), "files": sorted(x.stem for x in rd.glob("*.log"))})
        except SpecError as e:
            return self.send_json({"error": str(e)}, 400)
        # static: console at /, simulator at /sim/, analysis figures under /runs/
        if u.path in ("/", "/index.html"):
            self.path = "/console/index.html"
        elif u.path == "/sim":
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", "sim/")   # relative: works behind path-prefix proxies
            self.end_headers()
            return
        elif u.path.startswith("/sim/"):
            self.path = "/demo/" + u.path[len("/sim/"):]
        elif not (u.path.startswith("/console/") or u.path.startswith("/demo/")
                  or re.fullmatch(r"/runs/[^/]+/[^/]+/nll/analysis/[^/]+\.png", u.path)):
            return self.send_error(HTTPStatus.NOT_FOUND)
        if "/state/" in self.path:
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        try:
            if u.path == "/api/preview":
                job = validate(self.body())
                gs = gpus()
                pairs = job["n_pairs"] or max(len(free_pairs(gs)), 1)
                return self.send_json({"ok": True, "command": command(job), "estimate": estimate(job, min(pairs, max(len(gpu_pairs(gs)), 1)))})
            if u.path == "/api/jobs":
                job = validate(self.body())
                jid = uuid.uuid4().hex[:8]
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                job.update(id=jid, status="queued", created=now(),
                           run_dir=f"runs/{job['member']}/{stamp}_{job['split']}_{job['tag']}", command=command(job))
                with lock:
                    jobs.append(job)
                    save_jobs()
                return self.send_json(job, 201)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/cancel", u.path)
            if m:
                with lock:
                    j = next((j for j in jobs if j["id"] == m.group(1)), None)
                    if not j:
                        return self.send_json({"error": "없는 작업"}, 404)
                    if j["status"] == "queued":
                        j.update(status="cancelled", finished=now())
                    elif j["status"] == "running" and j["id"] in procs:
                        j["cancel"] = True
                        try:
                            os.killpg(procs[j["id"]].pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    save_jobs()
                return self.send_json(j)
        except SpecError as e:
            return self.send_json({"error": str(e)}, 400)
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({"error": f"잘못된 요청: {e}"}, 400)
        return self.send_json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    load_jobs()
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"experiment console: http://{a.host}:{a.port}   (simulator: /sim/)")
    if a.host not in ("127.0.0.1", "localhost"):
        print("[warn] reachable from the network: anyone with access can start GPU jobs")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
