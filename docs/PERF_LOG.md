# FastSegmentator 高速化ログ（`total` タスク）

> 2026-11-18: repository reorganised. `accel/` in the entries below now lives in `flashseg/` (batched.py, fastpost.py, static_compile.py) and `benchmarks/` (everything else); upstream root scripts moved to `fastsegmentator/`.

すべての変更について「何を変えたか → 速度の変化 → 結果の一致度」を記録する。
基準：FastSegmentator c8040ff `totalseg_infer.py --task total`（無変更）＝「original」。

方針の変更（2026-11-18）：最初の依頼では「前処理・後処理は原則変更しない」だったが、
ユーザーの指示で前処理・後処理の変更も可とした（変更はすべてここに記録する）。

## 環境
| 名前 | GPU | torch | 備考 |
|---|---|---|---|
| dev | 小型の開発用 GPU（機種名は非公開） | 2.14 | 予備実験用 |
| GH200 | GH200 120GB (miyabi-g) | 2.11.0+cu128 | TensorRT 10.12 (cu12, aarch64) |
| T4 | Kaggle T4 16GB | Kaggle 既定 | 実行待ち |

## 記録

### 0. 事前調査（コード変更なし）
- sliding window はバッチサイズ 1、1 症例ずつ逐次処理。5 モデル（291–295）は部位別モデル（5-fold ではない）。
- autocast fp16、step 0.8、mirror なし、cudnn deterministic。
- TensorRT なし。torch.compile は環境変数 `nnUNet_compile=1` で有効化可能（既定は無効）。

### 1. バッチ推論（`accel/batched.py`）
複数 CT のパッチを 1 つのキューに入れ、B 個ずつ forward。OOM 時はバッチを半分にして再試行。
- dev GPU / BTCV 12 例：original 285 CT/h → b1 309 / b2 310（+9%）/ b8 290 / b32 187。大きいバッチは逆効果（1 パッチで GPU が飽和）。
- GH200 / BTCV 12 例：original 946 CT/h → b2 1164（+23%）/ b4 1040 / b8 1045 / b32 799。forward 単体は B=8 で 1.65 倍。
- 一致度：b1 はビット単位で一致。b≥2 は Dice ≥0.999（fp16 カーネルの違い。BTCV_0003 の右腎嚢胞が 0→4 ボクセル）。
- b1 が original より約 8% 速い理由は未検証（`empty_cache()` の多用が原因との仮説）。

### 2. TensorRT（`accel/trt_backend.py`）
- dev GPU：初回 800 ms/patch（eager の 13 倍遅い）。原因は InstanceNorm（TRT が 128³ の 1 段 reduction に低並列のカーネルを選ぶ、約 90 ms/層）。
  等価な 2 段 reduction に書き換えて 105 ms/patch まで改善したが、eager 62 ms には及ばず。この GPU 向けの最適なカーネルが無い → 打ち切り。
- GH200（patch 単体、ms/patch、B=1/8）：eager 13.4/8.1、TRT fp16 10.7/9.2、TRT fp32 14.4/12.5、ORT CUDA 16.4/15.2。
- GH200 e2e（BTCV 12 例）：TRT fp16 単独 1134 CT/h（1.20 倍）、TRT fp16+b8 1235 CT/h（1.31 倍）、Dice 0.9987。エンジンのビルドは 125 s/モデル（fp16）、キャッシュ再利用可。
- バッチ 16 以上は TRT FP16 のビルドに失敗（テンソルが 2^31 要素を超える）→ max batch 8、それより大きいバッチは分割して実行。

### 3. torch.compile（patch 単体、GH200）
- miyabi では Inductor が `nvc` を使って失敗 → `CC=gcc CXX=g++` で解決。
- default：B=1 6.8 ms（2.0 倍）、B=8 5.5 ms。max-autotune：B=1 6.5 ms、B=8 5.4 ms（2.5 倍）。compile 時間 61 s / 151 s。argmax 一致 99.99995%。
- e2e（`nnUNet_compile=1`、TS test 89 例）：実行中。

### 4. パイプライン化（`batched.run_pipeline`）
読み込みスレッドで次のグループを先読み、書き出しスレッドで逆リサンプリングと書き出し、メインスレッドは GPU の forward だけ。
- dev GPU / BTCV 4 例：original 326 → pipeline_b1 360 CT/h（+10%）、ビット単位で一致。
- GH200 / TS test：実行中。

### 5. 前処理・後処理の最適化
（以下に追記）

#### 5a+5b. logit→seg を fp16 のまま GPU で処理／ラベルの統合を GPU の LUT で（`accel/fastpost.py`）
- プロファイル（dev GPU、TS test 10 例、original）：forward 81.2%、logit→seg 8.7%（1.15 s/例）、merge_part 2.6%、get_fdata 1.3%、cucim 1.3%、preprocess 1.2%、入力の再読込（sitk.ReadImage）1.1%、書き出し 0.4%。
- logit→seg が重い理由：同じ形なのでリサンプリングは元々スキップされているが、(i) logit を fp16→fp32 に丸ごとコピー、(ii) 毎回 gc.collect()+empty_cache()、(iii) crop の復元が CPU。
- 変更：fp16 のまま max と閾値の判定（fp16→fp32 は誤差なしなので結果は同じ）、crop の復元は GPU、gc/empty_cache なし。ラベルの統合は部位ごとの LUT を使い GPU で 1 回の gather。
- dev GPU / TS test 8 例：original 286 → pipeline_b1 319（1.11 倍）→ **pipeline_b1+fastpost 356 CT/h（1.24 倍）**。8/8 例でビット単位一致。

#### 5c. 書き出しで入力を読み直さない／float64 を経由しない（`batched.write_case_fast`、fastpost 版で使用）
- original は書き出しのときに `sitk.ReadImage(入力)` を `CopyInformation` のためだけに呼ぶ（入力 CT を丸ごと再展開）。
  さらに **方向行列が厳密に正規直交でない NIfTI では ITK がエラーを出す**ため、TS test 89 例中 16 例（s0235, s0236, s0295, s0311, s0561, s0684, s0735, s0802, s0829, s1174, s1276, s1406, s1407, s1409, s1417, s1419）で original は必ず失敗する。
- 変更：入力の nibabel ヘッダと affine で uint8 のまま書き出す。
- 同等性：同じセグメンテーションを旧・新の書き出し方で書き、8/8 例で voxel と affine が一致。
- 16 例：fastpost 版ではすべて出力できた（形・affine は入力と一致。s14xx の 5 例は頭部 CT でラベルが 5 種類なのは妥当）。
- 速度：dev GPU で pipefast_b1 356 → 360 CT/h（誤差の範囲。書き出しスレッドで既に隠れていた）。

### 参考：original 自身の実行ごとのぶれ
- cudnn deterministic を指定していても、**プロセスをまたぐと original もビット単位では再現しない**。
- original 同士（BTCV 2 例）：違うボクセル 248/例（6.6e-6）、Dice 0.99996、最も悪いラベル 0.9976。
- 新しい版同士（TS 8 例）：139/例（6.1e-6）、Dice 0.99997、最も悪いラベル 0.9994。
- → バッチ版などの差（Dice ≥0.9999）はこのぶれと同程度。「同じプロセス内でビット単位一致」と「プロセス間のぶれの範囲内」を分けて報告する。

### GT に対する Dice（`accel/eval_gt.py`、TotalSegmentator v2.0.1 test の正解マスク、`total` の 117 ラベル）
- 正解マスクは Zenodo の zip から 1 症例につき 1 回の範囲リクエストで取得（`fetch_ts_test_gt.py`。個別に取りに行くと 429 で弾かれる）。
- dev GPU / TS test 8 例：original 0.9476（症例ごとの平均の平均）、pipeline_b1 0.9476（完全に同じ）、pipefast_b1 0.9476（+0.000002）。
- original が失敗する 16 例（fastpost 版のみ）：0.9452 → 正しくセグメンテーションできている。

### GH200 / TS test 73 例（SimpleITK で読める症例。CT/h、Dice は GT に対する症例ごとの平均の平均）
| 条件 | compile なし | `nnUNet_compile=1` |
|---|---|---|
| original | 1104（3.26 s/CT）GT 0.93765 | 1182 GT 0.93765 |
| batched_b2 | 1237 | 951 |
| batched_b4 | 1293 | 1062 |
| pipeline_b1 | 1293 | 1243 |
| pipeline_b2 | 1518 | 1275 |
| pipeline_b4 | **1670（1.51 倍）** | 1287 |
- 全条件で GT Dice 0.93765〜0.93766（original との症例・ラベルごとの差は平均 1e-6、最大 0.026）。
- original のレイテンシ：中央値 3.1 s（p10 1.4、p90 5.6、最大 10.1 s）。compile あり：中央値 2.9 s（最大 19.2 s）、最初のコンパイル込みで 32 s。
- **既存の `nnUNet_compile`（torch.compile の既定設定）は、バッチ・パイプラインと組み合わせると逆に遅くなる。** 端数のバッチで形が変わると動的な形として再コンパイルされるためと推定 → 対策：`dynamic=False` と端数のバッチの 0 埋め。

### 6. 形を固定した torch.compile（`accel/static_compile.py`）
- `torch.compile(dynamic=False)` とし、端数のバッチは 0 で埋めて常に同じ形にする（InstanceNorm はサンプルごとの統計なので、0 埋めした行は本物の出力に影響しない）。
- dev GPU / TS test 8 例：pipeline＋fastpost＋static compile（default、b2）**492 CT/h（original 286 の 1.72 倍）**、コンパイル 24 s。GT Dice 0.94763（original と同じ、ラベルごとの差は平均 -1.6e-8）。

### 7. GH200 / TS test 73 例：fastpost＋compile とバッチ×グループの探索（CT/h、括弧は original 1104 に対する倍率）
| 構成 | b1 | b2 | b4 | b8 | b16 | b32 |
|---|---|---|---|---|---|---|
| pipeline＋fastpost（compile なし） | 1786（1.62） | 2117 | 2283（2.07） | | | |
| ＋既存の `nnUNet_compile` | 3317（3.00） | 2281 | 3038 | | | |
| ＋static compile、グループ 4 | 3328 | **3532（3.20）** | 2576 | 2683 | 2625 | 2407 |
| ＋static compile、グループ 8 | 3314 | **3542（3.21）** | 2589 | 2733 | 2699 | 2549 |
- GT Dice：全条件で 0.93765±0.00002（original 0.93765）。症例・ラベルごとの差は平均 |≤1e-5|、最大 0.033（小さい構造での数ボクセルの入れ替わり）。
- コンパイル時間（static）：b1 23 s、b2 28 s、b4 以上 9〜12 s。ピークメモリ：b2 11 GB、b32 43〜46 GB。OOM なし。
- **最速は static compile＋b2（約 3540 CT/h、1.02 s/CT、3.2 倍）**。b4 以上で遅くなる理由は未解明（パッチ単体では compile の B=4 のほうが速かった）。端数のバッチの 0 埋めによる無駄や、カーネル選択の違いを疑っている。
- 既存の `nnUNet_compile` も b1 なら同じ速さ（形が常に同じため）。b2 以上では再コンパイルの問題で落ちる。

### 8. 安い手の検証（dev GPU、パッチ単体、ms/パッチ/モデル。`accel/micro_knobs.py`）
| 手 | 結果 |
|---|---|
| cuDNN benchmark=True（deterministic 解除） | eager 63.9→63.4、compile 47.7→47.0（効果なし） |
| channels_last_3d | eager 63.9→176.7（2.8 倍遅い）、compile は変化なし |
| 5 モデルを grouped conv で 1 つにまとめる（`accel/fused.py`） | 別々 46.7（compile）→ まとめる 91.0（約 2 倍遅い）。出力は一致（argmax 一致 ≥99.9993%） |
- **compile の効果は B=4 で消える**（B=1/2 は 47 ms、B=4 は 63.5 ms で eager と同じ）→ GH200 で b4 以上が遅かった理由の候補。
- GH200 での同じ測定：実行待ち。

### 9. 外部評価：BTCV 30 例（GH200、正解は BTCV の 13 臓器。`accel/btcv_label_map.json` で `total` のラベルを対応づけ）
| | original | 最終設定（pipeline＋fastpost＋static compile、b2、g4） |
|---|---|---|
| スループット | 975 CT/h（3.69 s/CT） | **3124 CT/h（1.15 s/CT、3.20 倍）** |
| レイテンシの中央値（p10〜p90、最大） | 3.57 s（2.72〜5.22、6.69） | **1.61 s**（1.15〜2.28、2.79） |
| ピークメモリ | 7.4 GB | 10.2 GB |
| BTCV Dice（13 臓器の平均） | 0.8446 | 0.8446（症例・臓器ごとの差は最大 0.0012） |
- コンパイルは 33 s（推論時間とは別）。

### 10. 公式 TotalSegmentator（GH200、TS test、CLI で 1 例ずつ）
- 1 例あたり 50〜68 s（モデルの読み込みを毎回含む、公式の使い方）。debug-g の 30 分では 73 例が終わらないため、続きのジョブを 2 本つないでいる。

## 11. GH200 knobs / pipeline profile / official TotalSegmentator (2026-11-18)
- micro_knobs (ms per patch, all 5 models; results_gh200/micro_knobs_gh200.json):
  eager B1 67.0 / B2 104.7 / B4 176.7; compile B1 34.0 / B2 64.7 / **B4 176.9 (compile gain vanishes at B4, same as dev GPU)**.
  cudnn.benchmark: <=1% effect. channels_last: eager 1.6x slower, compile unchanged (B4 1.8x slower).
  fused (grouped conv): eager B1 54.9 (faster than separate eager 66.8), but compile B1 41.2 vs separate compile 33.7 -> fused loses once compiled. Not adopted.
- ts_prof_sc_b2 (pipefast_sc_b2, g4, TS-73): 3515 CT/h, GPU thread waited for prep only 0.73 s of 74.8 s, writer tail 0.05 s
  -> GPU-bound; further CPU pre/post work will not raise throughput on GH200.
- Official TotalSegmentator (8806051, GH200, 73 TS test): 54.95 s/case mean (median 52.9) = 65.5 CT/h; GT Dice case-mean 0.93765 (median 0.94625, label-mean 0.93081)
  = same accuracy as FastSegmentator original/final (0.93765). Final is ~54x faster in throughput.

## 12. Why is batch >=4 slower? (2026-11-18)
- probe_batch.py (fresh process per mode x B, GH200; results_gh200/probe_batch_{a,b}.jsonl), ms/patch/model:
  | B | 1 | 2 | 3 | 4 | 6 | 8 |
  |---|---|---|---|---|---|---|
  | eager | 13.40 | 10.38 | 9.29 | 8.71 | 8.42 | 8.10 |
  | static compile | 6.81 | 6.28 | 5.87 | **5.80** | 5.66 | 5.55 |
  -> the network itself does NOT slow down at B4; per-patch cost falls monotonically. The §11 / micro_knobs "B4 compile = eager"
  was an artefact of reusing one compiled module across B=1,2,4 in one process (third shape not compiled). cudnn.benchmark: no effect.
  Compile warm-up: 40 s with cudnn deterministic vs 14 s with benchmark (same speed after).
- Kernel profile (B2/B4): conv implicit-GEMM is NHWC; cudnn inserts nchw<->nhwc transposes around every conv
  (~29% of forward time at B2 under compile). Candidate for later (channels_last didn't remove them under compile).
- So the e2e b4+ slowdown (2400–2730 vs 3530 CT/h) must come from the pipeline around the forward. Next: FS_TIMERS=1 stage timers
  (stack / forward / gaussian accumulate, synced) for pipefast_sc b2/b4/b8 -> job 4139766.
- FS_TIMERS run (job 4139766, b2 -> b4 -> b8 in ONE process, TS-73):
  | B | CT/h | forward s | accum s | stack s | compile warm-up s |
  |---|---|---|---|---|---|
  | 2 | 3455 | 63.1 | 3.45 | 0.32 | 31.8 |
  | 4 | **3690** | 58.3 | 3.36 | 0.23 | 26.0 |
  | 8 | 2672 | **85.7** | 3.27 | 0.18 | **9.3** |
  b4 is now FASTER than b2. b8's forward = 8.7 ms/patch/model ≈ eager (8.1), and its "compile" warm-up took only 9 s
  -> b8 was never compiled. Cause: dynamo cache_size_limit=8 per code object; 5 models x 2 batch sizes already = 10 graphs of the
  same PlainConvUNet.forward -> torch silently falls back to eager. Earlier b4–b32 numbers (2400–2730) in multi-batch runs were the same artefact.
  Padding waste is small (b8: 10280 slots for 9870 patches, 4%). Gaussian accumulate 3.3 s, stack 0.2–0.3 s: not the issue.
- Fix: static_compile.py raises cache_size_limit to 64 (accumulated 512). Re-run b2/4/8/16 with TORCH_LOGS=recompiles -> job 4141106.
- After fix (job 4141106, b2 -> b4 -> b8 -> b16 in one process, TS-73; recompile log shows the expected new-shape compiles, no eager fallback):
  | B | CT/h | compile warm-up s | peak GB |
  |---|---|---|---|
  | 2 | 3541 | 31.6 | 11.4 |
  | 4 | 3736 | 25.9 | 12.3 |
  | 8 | **3782** | 26.2 | 15.8 |
  | 16 | 3647 | 26.9 | 22.8 |
  -> batch scaling restored: b8 best (+7% over b2, 3.43x original 1104). b4 almost the same with less memory.

## 13. Final-config checks (2026-11-18)
- Accuracy b8 (GH200 TS-73, eval_gt_sc_b8.json): Dice case-mean 0.937659 vs official TS 0.937649; paired mean diff 1e-6, abs max 0.026 (fp16 order).
- dev GPU TS-8 with cache-limit fix + cudnn.benchmark (bench2_sc_b248_dev): b2 494 / b4 483 / b8 479 CT/h (warm-up 11/26/31 s, peak 9.7/10.8/14.3 GB)
  -> on dev GPU batch size does not matter (compute-bound); b2 stays best there.

## 14. T4 three-way (Kaggle, BTCV 12, 2026-11-18)
| | per CT | CT/h | latency median | peak GB | BTCV Dice |
|---|---|---|---|---|---|
| official TotalSegmentator CLI (8806051) | 135.9 s | 26.5 | – | – | 0.81903 |
| FastSegmentator original | 25.6 s | 140.9 | 25.6 s | 7.4 | 0.81902 |
| pipeline + fastpost b1 / b2 (no compile) | 21.7 / 21.2 s | 165.7 / 169.5 | – | 7.8 | – |
| pipeline + fastpost + static compile + cudnn.benchmark b1 / **b2** | 17.8 / **17.5 s** | 202.5 / **205.9** | 18.6 / 18.7 s | 7.8 | 0.81902 |
-> final = 1.46x FastSegmentator, 7.8x official CLI on T4. Compile warm-up 45 s, model load 5 s (excluded).
   Dice paired vs official: mean -1e-5, abs max 0.024. (kaggle/out_3way, kaggle/eval_t4_3way.json)

## 15. FastSegmentator with its own nnUNet_compile=1 (fair baseline, 2026-11-18)
- GH200 TS-73 (from §6 runs): 1182 CT/h (3.05 s/CT), Dice 0.93765 -> FlashSegmentator b8 = 3.20x.
- T4 BTCV-12 (Kaggle fastseg-t4-fscompile): 163.6 CT/h (22.0 s/CT), latency median 21.7 s, first case incl. compile 66.7 s, peak 7.3 GB,
  BTCV Dice 0.81902 -> FlashSegmentator b2 (205.9) = 1.26x.

## 16. Equivalence tests (TOST, paired Dice, delta = 0.01, 90% CI; benchmarks/equivalence.py, 2026-11-18)
| dataset (GPU) | method vs reference | cases | case-mean diff [90% CI] | labels equivalent | widest label CI bound | (case,label) pairs: abs max / p99 |
|---|---|---|---|---|---|---|
| TS-73 (GH200) | FastSegmentator vs official TS | 73 | +1.3e-6 [-1.7e-6, +4.4e-6] | 117/117 | 0.0006 | 0.0058 / 0.0003 |
| TS-73 (GH200) | FlashSegmentator b8 vs official TS | 73 | +9.7e-6 [-2.8e-6, +2.2e-5] | 117/117 | 0.0021 (label 93) | 0.026 / 0.0004 |
| BTCV-30 (GH200) | FlashSegmentator b2 vs FastSegmentator | 30 | +5.4e-6 [-1.4e-6, +1.2e-5] | 13/13 | 0.0001 | – |
| BTCV-12 (T4) | FastSegmentator vs official TS | 12 | -8.8e-6 [-6.5e-4, +6.4e-4] | 13/13 | 0.0062 | – |
| BTCV-12 (T4) | FlashSegmentator b2 vs official TS | 12 | -1.0e-5 [-6.6e-4, +6.4e-4] | 13/13 | 0.0062 | – |
All case-level TOST p < 1e-11. Every label is equivalent even at delta = 0.0025 on TS-73 (widest bound 0.0021).
Files: results_gh200/equiv_ts73.json, equiv_btcv30.json, kaggle/equiv_t4.json.

## 17a. Layout transposes / cudnn.benchmark / BTCV-30 at b8 (GH200, 2026-11-18)
- probe_layout (ms/patch/model, B4 / B8): compile 5.83 / 5.58; + channels_last 5.83 / 5.62; + inductor force_layout_optimization 5.81 / 5.58; both 5.83 / 5.61.
  cudnn nchw<->nhwc transposes stay at ~33% of forward (B8: 73.5 of 223 ms) in every compiled variant; only eager+channels_last removes them,
  but eager is 2.7x slower overall. -> not removable with stock PyTorch/cuDNN; not adopted (would need an NDHWC-native conv path).
- cudnn.benchmark in the final pipeline: TS-73 b8 3769 CT/h (vs 3782 without) -> speed unchanged; kept for shorter warm-up on fresh shapes.
- BTCV-30 FlashSegmentator b8 + cudnn.benchmark: 3333 CT/h (b2 was 3124; FastSegmentator 975 -> 3.42x), latency median 1.63 s, warm-up 15 s.

## 18. Official TotalSegmentator with concurrent CLI processes (user-level `xargs -P`, not an official feature; 2026-11-18)
- GH200, first 24 TS-test CTs: NP=1 65.5 (73 CTs) / NP=2 124.0 / NP=4 224.2 / NP=8 397.0 CT/h; GPU util mean 12/15/33 %, peak GPU mem 10.4/13.8/24.2 GB; no failures.
  NP=16/32 (48 CTs) pending. FlashSegmentator b8 (3782) = 9.5x NP=8.
- T4 (Kaggle, 4 vCPU, 31 GB RAM, single GPU), BTCV-12: NP=1 26.5 / NP=2 39.0 / NP=4 42.0 and 44.3 (two runs) CT/h; GPU util ~19–23 %, peak GPU mem 7.6 / 11.5–14.4 GB of 16; no failures.
  CPU-bound on small machines. FlashSegmentator b2 (206) = 4.7–4.9x NP=4.
- GPU energy per CT on GH200 (power.draw only, idle 67 W; throughput window): official NP=1 2.02 Wh, FastSegmentator 0.28 Wh, FlashSegmentator b8 0.16 Wh (602 W mean).
