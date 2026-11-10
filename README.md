# fullduplex-serving-lab

Full-duplex 음성 모델(Moshi) 서빙 실험실: 침묵 기반 KV cache 축소 연구, 실험 콘솔과 정책 시뮬레이터.

연구계획서 "Full-duplex 음성 모델 서빙: 침묵 기반 KV cache 축소 연구"의 **Phase 1(teacher-forced NLL)** 을 구현한 코드이며, Phase 3(실제 서빙 측정)에 쓸 스트리밍 KV cache도 포함합니다.

## 빠른 시작

### 0. 요구 사항
- Linux, NVIDIA GPU. moshiko fp16(~15.4GB)이 들어갈 메모리가 필요합니다: **24GB GPU 1장**(3090/4090/A100 등) 또는 **16GB GPU 2장**(T4 등, 모델을 반씩 나눠 올림).
- 디스크 약 25GB (모델 16GB + 데이터 150개 대화 약 3GB + 인코딩 결과)
- Python 3.10 이상. [uv](https://docs.astral.sh/uv/)가 있으면 사용하고, 없으면 `python3 -m venv`로 설치합니다.
- (선택) `ffmpeg`: 데모용 오디오를 새로 만들 때만 필요합니다.

### 1. 환경 설정 + 모델/데이터 다운로드

```bash
git clone https://github.com/Choiyoonsou/fullduplex-serving-lab.git && cd fullduplex-serving-lab
./setup.sh                     # .venv 생성, torch(CUDA 12.6) 설치, 모델과 DailyTalk 대화 150개 다운로드
```

- 모델(`kyutai/moshiko-pytorch-bf16`)과 데이터(`kyutai/DailyTalkContiguous`, CC-BY-SA 4.0)는 Hugging Face에서 자동으로 받습니다. 로그인은 필요 없습니다.
- 익명 다운로드는 속도 제한(HTTP 429)에 걸릴 수 있습니다. 스크립트가 자동으로 기다렸다가 다시 시도하고, `HF_TOKEN=hf_xxx ./setup.sh`처럼 토큰을 주면 빨라집니다.
- CUDA 버전이 다르면 `TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./setup.sh`처럼 지정합니다.
- **데이터와 모델은 git에 없습니다.** `./setup.sh`가 받거나, setup을 건너뛰어도 `scripts/run_all.sh`가 처음 실행될 때 없는 것만 자동으로 받습니다. 이미 있으면 네트워크 없이 그대로 진행합니다.
- 환경만 만들려면 `./setup.sh --no-download`, 데이터만 다시 받으려면 `.venv/bin/python scripts/00_download.py --skip_model`을 실행합니다.

### 2. 설치 확인 (CPU, 몇 초)

```bash
PYTHONPATH=. .venv/bin/python tests/test_policies.py
```

### 3. 실험 실행

실행 한 번은 `runs/<사람>/<날짜시각>_<태그>/` 폴더 하나에 저장됩니다. 사람 id와 기본 GPU는 [members.json](members.json)에 있습니다.

```bash
MEMBER=yschoi TAG=baseline scripts/run_all.sh                    # 정책 30개 전체 그리드 (dev)
MEMBER=yschoi TAG=final SPLIT=test CONFIRM_TEST=1 scripts/run_all.sh   # 최종 test 평가 (방법 확정 후 1회)
MEMBER=jmkim TAG=hang2 NOTE="침묵 판정 여유 2프레임" scripts/run_all.sh --hang 2
MEMBER=sgkwon TAG=quick N_DIALOGS=20 SKIP_VERIFY=1 scripts/run_all.sh --budgets 125 250
RUN_DIR=runs/jmkim/20261111-0930_hang2 scripts/run_all.sh        # 중간에 끊긴 run 이어서 실행
```

데이터 준비 → 정확성 검증 → NLL 그리드 → 분석 순서로 실행합니다. 6×T4에서 GPU 6장을 모두 쓰면 약 1.5시간, 한 사람 몫(GPU 2장)이면 약 4시간이 걸립니다. 스크립트 뒤에 붙인 인자는 `03_run_nll.py`로 그대로 전달됩니다(`--hang`, `--budgets`, `--n_sink`, `--int4`).

### 4. 데모 실행

```bash
scripts/serve_demo.sh                       # http://localhost:8000
PORT=9000 BIND=0.0.0.0 scripts/serve_demo.sh  # 다른 기기에서 접속할 때
```

원격 서버라면 `ssh -L 8000:localhost:8000 <서버>`로 포트를 연결한 뒤 브라우저에서 엽니다. 데모 데이터(`demo/data.json`, `demo/audio/`)는 저장소에 들어 있어서, **GPU나 모델 없이도 바로 실행됩니다.** `index.html`을 파일로 바로 열면 data.json을 불러오지 못하므로 반드시 서버로 띄워야 합니다.

**GitHub Pages 배포:** 저장소 Settings → Pages → Source를 "GitHub Actions"로 바꾸면, `main`에 push할 때마다 `.github/workflows/pages.yml`이 `demo/`를 배포합니다.

**내 실험 결과로 데모 갱신:** `uv pip install --python .venv/bin/python -r requirements-demo.txt`로 Whisper를 설치한 뒤 `.venv/bin/python scripts/05_export_demo.py --res runs/<id>/<dev run>/nll --test_res runs/<id>/<test run>/nll --ids 746 1624 ...`를 실행합니다. ffmpeg가 필요하고, 사용자 쪽 전사는 CPU에서 대화당 몇 초면 끝납니다.

**데모 구성:** 핵심 결과(가설별 dev/test 판정), 정책 시뮬레이터(두 정책 비교, KV 내용, 프레임별 ΔNLL), 대화 패널(재생에 맞춘 단어 하이라이트, 정책이 잊은 말 표시), 전체 결과 그래프, 3090 세션 수 환산표.

### 5. 실험 콘솔 (웹에서 실험 실행·관리)

```bash
scripts/serve_console.sh          # http://localhost:8800   (원격: ssh -L 8800:localhost:8800 <서버>)
```

- **실험 탭:**
  - 계획서 후보 14개를 그룹별로 보여줍니다. A는 지금 실행 가능, B는 서빙 하네스 필요, C는 모델·데이터 필요입니다.
  - 실행 가능한 실험은 폼에서 파라미터, 담당자, dev/test, 대화 수, 침묵 비율, GPU 쌍을 고릅니다. **예상 소요 시간과 실제 실행될 명령어**가 미리 보입니다.
  - test를 고르면 확인란과 이전 test 평가 기록이 나옵니다.
- **작업 큐 탭:** FIFO 큐에서 빈 GPU 쌍이 생기면 자동으로 시작합니다. 콘솔 밖(터미널)에서 쓰는 GPU는 메모리 사용량으로 감지해서 피합니다. 진행률, 남은 시간, 단계, 로그(worker별), 취소(두 번 눌러 확인)를 지원합니다.
- **결과 탭:** 모든 run(콘솔과 터미널 실행 모두)을 담당자, 실험, split으로 거릅니다. run을 열면 설정, 커밋, 기준 대비 ΔNLL 표(유의성, CI), 가설 검정, 그래프가 나옵니다. 여러 run을 체크하면 지표별로 나란히 비교합니다.
- 콘솔은 결국 `scripts/run_all.sh`를 실행하므로, 터미널에서 실행한 것과 결과가 같고 같은 `runs/` 구조에 저장됩니다.
- 보안: 기본으로 127.0.0.1에만 열립니다. 포트에 닿는 사람은 누구나 GPU 작업을 실행할 수 있으므로, 외부에 열지 말고 SSH 터널을 쓰세요. 입력은 서버에서 `console/experiments.json` 스키마로 검증한 뒤 명령으로 만듭니다.

| 실행 가능한 실험 (`EXP`) | 계획서 후보 | 바꾸는 것 | 보는 것 |
|---|---|---|---|
| `kv_policy` | KV cache 관리 | KV 정책 × 메모리 | ΔNLL, 3090 세션 수, 가설 H1~H4 |
| `rvq` | RVQ 단계 조절 | 입력에 남길 Moshi codebook 수 k | 빠진 codebook 때문에 생기는 이해 손실 |
| `kv_quant` | 양자화와 대화 타이밍 | KV 비트 수 (8/6/5/4/3/2) | 턴 시작, EPAD(말 시작 결정), 듣는 중 침묵 유지 NLL |

모든 실험에는 턴테이킹 지표 세 가지가 함께 기록됩니다. **턴 시작**은 Moshi 턴의 첫 단어, **EPAD**는 "다음 프레임에 말을 시작한다"는 결정, **듣는 중**은 사용자만 말할 때 PAD를 유지하는 것(끼어들지 않기)입니다. 결과는 `nll/analysis/variants.md`와 `fig_variants.png`에 저장됩니다.

새 실험 종류를 추가하려면 `console/experiments.json`에 항목을 넣고(params의 `cli`가 `03_run_nll.py` 인자), `03_run_nll.py --exp`에 분기를 추가하면 됩니다. B 그룹은 서빙 하네스가 생기면 같은 방식으로 붙입니다.

### 단계별 실행 (참고)

```bash
export HF_HOME=$PWD/hf_cache; PY=.venv/bin/python
$PY scripts/01_prep_data.py --n 100 --ratios 0.2 0.4 0.6          # Mimi 인코딩 (1회만)
$PY scripts/02_verify.py --data data/prepared/r0.6 --gpus 0 1 --bf16_gpus 2 3   # 정확성 검증 (bf16 비교는 선택)
$PY scripts/03_run_nll.py --gpus 0 1 --shard 0 --nshards 3 --out runs/<id>/<run>/nll   # worker 하나 = GPU 쌍 하나
$PY scripts/04_analyze.py --res runs/<id>/<run>/nll                         # 표, 그림, 동시 세션 수 환산
$PY scripts/06_hypotheses.py --res runs/<id>/<run>/nll                      # 사전 등록 가설 검정
```

저장소에는 100개 대화 실험의 요약(`runs/yschoi/20261110-0510_baseline/nll/`)과 분석 결과가 들어 있습니다. 따라서 `04_analyze.py`는 GPU 없이 바로 돌려볼 수 있습니다. 프레임별 결과(npz, 84MB)는 용량 때문에 제외했습니다.

## 평가 절차 (dev/test 분리와 사전 등록 가설 검정)

**데이터 분할** ([splits/](splits/), 2026-11-10 고정)

| split | 대화 | 길이 | 용도 |
|---|---|---|---|
| `dev` | 가장 긴 100개 | 69~103초 | 튜닝과 탐색. P1W window, hang, sink 개수는 모두 여기서 정했습니다. |
| `test` | 그다음으로 긴 100개 | 57~69초 | 최종 수치 전용. 방법을 확정한 뒤 한 번만 평가합니다. |

- 모든 run은 기본적으로 `SPLIT=dev`입니다. 새 아이디어는 dev에서 비교합니다.
- `SPLIT=test`는 `CONFIRM_TEST=1`이 있어야 실행되고, 실행할 때마다 [splits/test_log.jsonl](splits/test_log.jsonl)에 기록이 남습니다(git으로 공유). test 결과를 보고 설정을 바꾼 뒤 다시 test를 돌리면, 그 결과는 더 이상 held-out이 아닙니다. 로그로 이를 확인할 수 있습니다.
- test 대화는 dev보다 평균 18% 짧습니다(길이가 긴 순서로 나눴기 때문). 같은 메모리 budget에서 효과가 약간 작게 나올 수 있으므로 보고서에 한계로 적습니다.

**가설 검정** (`scripts/06_hypotheses.py`, run_all.sh 마지막 단계에서 자동 실행)

- 분석 단위는 대화입니다. 프레임은 서로 독립이 아니므로 프레임 단위 검정은 하지 않습니다.
- 대화 단위 paired sign-flip permutation 검정을 쓰고, 가설군 × 지표마다 Holm 보정을 합니다(α=0.05).
- 효과 크기로 평균 차이, 95% bootstrap CI, Cohen's dz, 차이가 음수인 대화 비율을 함께 보고합니다.
- 비교 대상과 검정 방향은 코드에 고정되어 있습니다. test 결과를 본 뒤에는 수정하지 않습니다.

| 가설 | 검정 |
|---|---|
| H1 메모리를 줄이면 NLL이 오른다 | B1@C − full > 0 (단측), C = 125/250/500/1000 |
| H2 같은 메모리에서 침묵 인식이 B1보다 낫다 | P1W@C − B1@C (양측), 보조로 P1S. 텍스트와 cb0를 따로 판정 |
| H3 그 이득은 침묵 비율에 비례한다 | gain(60%) − gain(20%) > 0 (단측, 같은 대화끼리 짝지음) |
| H4a int8은 품질을 바꾸지 않는다 | full_int8 − full의 90% CI가 ±0.01 nats 안 (동등성, TOST) |
| H4b int8과 결합하면 더 낫다 | P2 − B2, P2 − P1W (같은 메모리, 양측) |
| H5 서빙 동시 세션 수 | 3090 서빙 측정이 필요하며 여기서는 검정하지 않습니다 |

결과는 run 폴더의 `nll/analysis/hypotheses.md`(표)와 `hypotheses.csv`에 저장됩니다.

## git에 올라가는 것과 아닌 것

| 올라감 (~10MB) | 올라가지 않음 (`.gitignore`, 처음 실행할 때 생성되거나 다운로드) |
|---|---|
| 코드, 설정, `splits/`, `members.json` | `.venv/`, `hf_cache/`(모델 16GB) |
| run별 `meta.json`, `ids.txt`, `nll/summary.jsonl.gz`, `nll/analysis/` | `data/`(오디오, Mimi 토큰 4GB) |
| `demo/`(data.json, 오디오 8MB) | run별 `logs/`, 프레임별 `*.npz`, 실행 중 shard 파일 |

run이 끝나면 worker별 결과가 `summary.jsonl.gz` 하나로 합쳐집니다. 원본보다 8배 작아서 run이 계속 쌓여도 저장소가 크게 늘지 않습니다. 데모 오디오는 데이터지만, GPU 실험 없이 데모와 GitHub Pages가 바로 동작하도록 예외로 포함했습니다.

## 팀 실험 관리 (권숭구 · 김정민 · 최윤수)

```
runs/
  yschoi/20261110-0510_baseline/      # 실행 1회 = 폴더 1개
    meta.json        # 누가, 언제, 어떤 커밋과 인자로 돌렸는지, 메모, 상태(running/done/failed)
    ids.txt          # 사용한 대화 목록
    logs/            # prep, verify, worker별 로그
    nll/summary_shard*.jsonl, nll/analysis/   # 결과 (npz는 git 제외)
  jmkim/...
  sgkwon/...
```

- **GPU 분배:** `members.json`에 사람별 기본 GPU 쌍을 나눠 두었습니다(권숭구 0,1 / 김정민 2,3 / 최윤수 4,5). 셋이 동시에 돌려도 겹치지 않습니다. 남는 GPU를 빌릴 때는 `GPU_PAIRS="0,1 2,3"`처럼 직접 지정합니다.
- **공유 데이터:** Mimi 인코딩 결과(`data/prepared`)는 항상 같은 결과가 나오므로 모두가 같이 씁니다. 동시에 실행해도 파일이 깨지지 않도록 저장합니다.
- **기록:** meta.json에 git 커밋과 "커밋 안 된 코드 변경 여부(git_dirty)"가 남습니다. 결과를 비교하거나 공유할 때는 코드를 커밋한 뒤 돌리는 것을 권장합니다.
- **조회와 비교:**
  ```bash
  .venv/bin/python scripts/runs.py                          # 전체 run 목록
  .venv/bin/python scripts/runs.py --member jmkim           # 한 사람의 run
  .venv/bin/python scripts/runs.py compare                  # 사람별 최신 run의 ΔNLL 비교
  .venv/bin/python scripts/runs.py compare runs/a/... runs/b/... --ratio 0.4 --metric d_cb0_nll
  ```
- **Git 운영 권장:**
  - 코드(`fdkv/`, `scripts/`)를 바꿀 때는 각자 브랜치(`sgkwon/...`)에서 작업하고 PR로 합칩니다.
  - 실험 결과(`runs/<본인>/`)는 본인 폴더에만 생기므로, main에 바로 커밋해도 서로 충돌하지 않습니다.
  - 의미 있는 run에는 `NOTE`로 무엇을 바꿨는지 적어 두세요. `runs.py` 목록에 표시됩니다.
- **새 인원 추가:** `members.json`에 한 줄 추가하면 됩니다.

## 구성

| 파일 | 역할 |
|---|---|
| `fdkv/policies.py` | 정책 정의, 정책별 **attention 마스크**, 스트리밍용 `PolicyKVCache`(RingKVCache 대체, int8 저장 포함) |
| `fdkv/model.py` | moshiko를 GPU 2개에 나눠 fp16으로 로드, temporal attention 패치, teacher-forced forward, 프레임별 NLL |
| `fdkv/data.py` | DailyTalkContiguous 재구성(침묵 비율 조절), Mimi 인코딩, 텍스트 스트림(moshi-finetune Interleaver와 동일 규칙) |
| `setup.sh`, `scripts/00_download.py` | 환경 설정, 모델·데이터 다운로드 |
| `splits/`, `scripts/06_hypotheses.py` | dev/test 분할(고정), 가설 검정 |
| `console/`, `scripts/serve_console.sh` | 실험 콘솔 (작업 큐, GPU 스케줄러, 결과 브라우저) |
| `scripts/07_analyze_variants.py` | 모든 실험 공통 분석 (기준 대비 ΔNLL, 턴테이킹 지표) |
| `members.json`, `scripts/runs.py` | 실험 인원과 GPU 분배, run 목록과 비교 |
| `scripts/0*.py`, `scripts/run_all.sh` | 준비 → 검증 → 그리드 → 분석 → 데모 데이터 내보내기 |
| `demo/` | 정적 데모 페이지 (`scripts/serve_demo.sh`) |

## 효율화 핵심: 스트리밍 S스텝을 병렬 forward 1회로

Moshi는 key를 **RoPE를 적용한 뒤(절대 위치)** 저장합니다. 따라서 step s의 출력은 "그 시점에 cache에 남아 있는 과거 프레임 집합"만으로 정해집니다. ring buffer, sink, 침묵 쓰기 생략은 모두 **causal 마스크의 희소 패턴**이므로, 정책 하나를 마스크를 넣은 teacher-forced forward 한 번으로 정확하게 재현할 수 있습니다.

- 검증 결과(`02_verify.py`): moshi 실제 스트리밍(RingKVCache)과 비교한 max|ΔNLL|이 fp16 노이즈 수준(0.03)이고, 평균 logit 차이는 0.003입니다.
- 대화 1개 × 정책 30개 = 약 25~65초. 100개 대화 × 침묵 3조건 × 30정책이 6×T4에서 약 1.5시간입니다.
- 마스크가 같은 정책(budget ≥ 대화 길이 등)은 다시 계산하지 않고 결과를 공유합니다. 같은 대화의 정책들은 batch로 묶어서 계산합니다.
- 프레임별 NLL을 npz로 저장하므로, 구간별·위치별 재분석은 GPU 없이 할 수 있습니다.
- 대화마다 Mimi 인코딩은 한 번만 하고 모든 정책에서 재사용합니다.

## 정책 그리드 (`make_policies`)

budget은 세션당 KV 메모리이며 bf16 프레임 수로 표시합니다. 1프레임 = 0.5MB이고, int8 정책은 같은 메모리에 슬롯이 2배입니다.

| 기호 | 내용 | 비고 |
|---|---|---|
| full | context 3000 | 기준. `full_int8`은 H4(양자화만 적용) 확인용 |
| B0 | ring buffer | 계획서 원안 |
| B1 | sink 4프레임 + 최근 window | StreamingLLM |
| P1 | 침묵 프레임 KV 쓰기 생략 | 계획서 원안 |
| **P1S** | P1 + sink | 침묵 생략 효과만 따로 보는 B1 대조군 |
| **P1W** | sink + 최근 budget/4 프레임은 모두 유지 + 그보다 오래된 구간은 발화 프레임만 | 제안 수정안 |
| B2 | B1 + KV int8 | |
| P2 | P1W + KV int8 | |

침묵 판정은 양 채널 에너지(80ms, -45dBFS) 기준이고, 앞뒤로 1프레임(`--hang`)씩 늘립니다. 입력 step s에는 프레임 s-1의 semantic 토큰과 프레임 s-2의 acoustic 토큰이 들어가므로(Moshi delay), 두 프레임 중 하나라도 발화면 그 step을 유지합니다.

## 계획서와 달라진 점과 이유

1. **GPU가 T4(16GB)라서** 가중치(15.4GB)를 GPU 2개에 나눠 올리고 fp16으로 실행합니다. T4에서는 bf16 연산이 9배 느립니다. fp16과 bf16의 NLL 차이는 0.0005 미만으로 확인했습니다. 80ms 실시간 서빙 측정(H5)은 T4로 하면 의미가 없으므로 **3090에서 진행**해야 합니다.
2. **데이터는 `kyutai/DailyTalkContiguous`를 사용합니다.** Kyutai가 moshi-finetune용으로 만든 버전으로, 화자별 스테레오 채널과 단어 타임스탬프가 있고 라이선스는 CC-BY-SA 4.0입니다. 원본 DailyTalk를 받아 2채널로 직접 합치고 정렬할 필요가 없습니다. 단어 정렬은 Moshi 화자 쪽에만 있지만, 학습 때도 `keep_main_only=True`로 Moshi 쪽 단어만 썼으므로 문제가 없습니다.
3. **침묵 판정을 "Moshi 텍스트 PAD" 대신 에너지로 합니다.** Moshi 텍스트 스트림은 단어 *도중에도* PAD를 넣기 때문에, PAD만으로 침묵을 판정하면 발화 구간 대부분이 침묵으로 잡힙니다.
4. **침묵 비율은 화자가 바뀌는 간격만 늘려서 조절합니다.** 같은 화자 턴 안의 문장 간격은 그대로 둡니다. 원본 대화의 양쪽 침묵 비율은 약 23%입니다.
5. **sink 발견 → P1S, P1W를 추가했습니다** (아래 참고).

## 본 결과 (100개 대화 × 침묵 3조건, `runs/yschoi/20261110-0510_baseline/nll/`)

ΔNLL은 full context 대비 증가량이며, 괄호는 대화 단위 bootstrap 95% CI입니다. budget 125는 세션당 KV 62MB입니다.

| 침묵 | 정책 (budget 125) | Δ text content | Δ cb0 |
|---|---|---|---|
| 20% | B1 / P1W / B2 / P2 | +0.115 / +0.115 / +0.035 / +0.036 | +0.054 / +0.088 / +0.020 / +0.059 |
| 60% | B1 / P1W / B2 / P2 | +0.294 / +0.129 / +0.116 / **+0.042** | +0.061 / +0.214 / **+0.027** / +0.112 |

- **H3 지지(텍스트):** 침묵 생략의 텍스트 이득은 침묵 비율에 비례해 커집니다. 침묵 20%에서는 차이가 없고, 60%에서는 B1 대비 Δ가 절반 이하로 줄어듭니다.
- **H2는 텍스트에서만 성립합니다:** 음성 cb0는 침묵 생략이 일관되게 손해입니다. P1W로 줄일 수는 있지만 B1보다 여전히 나쁩니다.
- **H4 지지:** int8은 사실상 손실이 없습니다(full_int8 Δ ≈ 0). 같은 메모리에서 B2가 두 지표 모두 B1보다 낫습니다.
- B0와 P1(sink 없음)은 budget 500 이하에서 무너집니다.
- 세션 수 환산(3090, KV 6GB 기준, 두 지표 모두 상한 CI ≤ 0.05): full 4, B1 24~49, **B2 48~96**, P2 48. 이 값은 *메모리만* 기준으로 한 환산이라, 실제 상한은 연산(80ms) 한계에서 먼저 걸릴 가능성이 큽니다. 3090에서 확인해야 합니다.

## 예비 결과 (3개 대화)

- **sink가 없으면 Moshi가 무너집니다.** budget 125(1분 대화의 약 10초)에서 B0의 content NLL은 +6.0이고, sink 4프레임만 남긴 B1은 +0.09입니다. 계획서의 P1과 B0 비교는 사실상 sink 유무를 비교하게 됩니다.
- **P1S(침묵 생략 + sink)는 텍스트 NLL에서 B1보다 낫지만**(침묵 60%, budget 125: +0.12 vs +0.25), **음성 cb0 NLL이 budget과 관계없이 약 +1.0 나빠집니다.** 최근 침묵 프레임까지 버리기 때문으로 보입니다.
- **P1W**는 최근 window를 유지해서 cb0 손실을 +0.003~0.18 수준으로 되돌립니다. 그래서 침묵 생략은 "오래된 기억"에만 적용하는 것이 맞아 보입니다.
- **int8(B2)은 NLL을 거의 바꾸지 않습니다**(full_int8의 Δ ≈ 0). 같은 메모리에서 int8+2배 슬롯이 가장 효과가 큽니다.

## 다음 단계

- **Recall probe (Phase 2)**: DailyTalk 대화는 최대 약 100초로 짧아서 context 1500~3000 조건을 볼 수 없습니다. 계획대로 LLM 대본과 TTS로 5~10분 대화를 만든 뒤 `01_prep_data.py`와 같은 형식(`codes`, `speech`)으로 저장하면 `03_run_nll.py`를 그대로 쓸 수 있습니다. 심어둔 정보 토큰 위치의 NLL만 따로 뽑으면 생성 평가 전에 값싼 recall 지표를 얻을 수 있습니다.
- **서빙 측정 (Phase 3, 3090)**: `PolicyKVCache`를 temporal layer마다 `state.kv_cache`로 바꿔 끼우고 `mha.context=None`으로 설정합니다(`02_verify.py`의 `streaming_text_logits` 참고). 세션별 keep 값은 `keep_ref.keep[B]`로 매 스텝 넣어줍니다. 아직 CUDA Graph 호환성(dtype·크기 고정은 되어 있음)과 int8 복원 비용은 측정하지 않았습니다.

## 라이선스와 출처

- 데이터: [kyutai/DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) (원본 [DailyTalk](https://github.com/keonlee9420/DailyTalk), CC-BY-SA 4.0). `demo/audio/`는 이 데이터의 턴 간격을 다시 배치한 파생물이므로 같은 라이선스를 따릅니다.
- 모델: [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (사용 조건은 모델 카드를 따릅니다). 저장소에는 가중치가 들어 있지 않습니다.
