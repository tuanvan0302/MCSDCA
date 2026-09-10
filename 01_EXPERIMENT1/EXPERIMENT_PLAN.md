# Kịch bản thử nghiệm 1 — Dùng MCSDCA làm optimizer cho LeWM

> ⚠️ **ĐÃ LỖI THỜI (2026-09).** Toàn bộ phần dò tham số (tuning) đã bị gỡ bỏ.
> Giờ chỉ còn **một** cách chạy, mọi tham số nằm trong `configs/experiment1.yaml`
> (= giá trị 2 paper, KHÔNG tune):
>
> ```text
> python src/run_experiment1.py                    # quét data.fractions × seed từ YAML
> python src/run_experiment1.py --only-fraction 0.04
> python src/run_experiment1.py --set train.budget=400 --set 'seed=[3072]'   # smoke
> ```
>
> Đã xoá: `src/sweep.py`, `src/run_ablation.py`, `src/smoke_test_predictor.py`,
> `configs/sweep/`, `configs/ablation.yaml`, `winners.json`, các cờ `--data` /
> `--profile` / `--reuse-winners`. Phần bên dưới giữ lại để tham khảo lịch sử.

---

> File này thay cho `EXPERIMENT_PLAN.txt` cũ. Bản `.txt` nói về việc train model
> LeWM gốc bằng `le-wm/train.py` — đó là việc khác, không nằm trong thử nghiệm này.

3 script gốc:

- `src/run_pusht_predictor_experiment.py` — chạy 1 lần train.
- `src/sweep.py` — chạy nhiều lần train theo lưới tham số rồi tự xếp hạng.
- `src/run_ablation.py` — chạy vòng đánh giá cuối theo nhiều mức dữ liệu.
- `src/run_experiment1.py` — **bọc cả 3 cái trên vào một lệnh** cho một mức dữ liệu.

Config nằm ở `configs/sweep/*.yaml` và `configs/ablation.yaml`.

---

## 1. Mục tiêu

Train lại toàn bộ 5 khối của LeWM (`encoder`, `projector`, `action_encoder`,
`predictor`, `pred_proj`) từ đầu, với hàm mục tiêu của LeWM là
`sai số dự đoán (MSE) + hệ số · SIGReg`.

So sánh 3 thuật toán cập nhật trọng số (optimizer):

| Optimizer | Là gì |
|---|---|
| `AdamW` | thuật toán chuẩn, dùng làm mốc so sánh (có tinh chỉnh tham số) |
| `MCSDCA-odLD` | MCSDCA bản overdamped Langevin |
| `MCSDCA-udLD` | MCSDCA bản underdamped Langevin |

**Câu hỏi cần trả lời:** khi cho cùng một lượng tính toán như nhau, MCSDCA có cho
kết quả dự đoán (rollout) tốt hơn hoặc ổn định hơn AdamW không? Và lợi thế đó có
**rõ hơn khi dữ liệu ít** không?

Để trả lời, ta làm **thí nghiệm giảm dần dữ liệu**: giữ nguyên lượng tính toán,
chỉ đổi tỉ lệ dữ liệu dùng để train ở các mức `1%`, `10%`, `50%`, `100%`
(và `0.01%` để test luồng trước).

---

## 2. Những thứ phải giữ giống nhau khi so sánh

Muốn so sánh công bằng, mọi optimizer phải chạy trong cùng điều kiện:

1. **Đo theo số lần `backward()`, không đo theo epoch hay theo giờ.**
   Mỗi optimizer được cấp cùng một "ngân sách" số lần lan truyền ngược
   (`backprop_budget`). MCSDCA dừng khi dùng hết ngân sách này.
2. **Cùng điểm xuất phát.** Trọng số ban đầu được lưu 1 lần, rồi nạp lại y hệt
   trước mỗi optimizer.
3. **Cùng thứ tự dữ liệu.** Cùng seed, cùng luồng minibatch. MCSDCA lấy minibatch
   mới cho mỗi bước Langevin bên trong (đúng như thuật toán trong bài báo).
4. **Cùng cấu hình train** (`profile: small`): cùng batch size, cùng độ chính xác
   số (`fp32`), cùng `sigreg_num_proj`, cùng scheduler.
5. **Bên nào cũng được tinh chỉnh tham số.** Cả AdamW lẫn MCSDCA đều phải dò tham
   số trước khi vào so sánh cuối. Không lấy "MCSDCA đã chỉnh" đấu với
   "AdamW để mặc định".
6. **Ngân sách đánh giá cố định ở mọi mức dữ liệu.** Mức dữ liệu là biến duy nhất
   thay đổi; mức dữ liệu nhỏ hơn thì đơn giản là đi qua dữ liệu nhiều vòng hơn.

---

## 3. Các tham số cần dò

### 3.1 MCSDCA-odLD

| Tham số (dòng lệnh / tên trong yaml) | Ý nghĩa ngắn gọn | Các giá trị thử |
|---|---|---|
| `--mcsdca-eta` / `mcsdca_eta` | độ lớn bước Langevin (bản overdamped) | `3e-3, 1e-2, 3e-2` |
| *(tỉ lệ)* `mcsdca_epsilon_ratio` | `r = epsilon/eta`; code tự đặt `epsilon = r · eta` | `1e-6, 1e-2, 1e0` |
| `--mcsdca-beta0` / `mcsdca_beta0` | trọng số trộn của DCA vòng ngoài | `0.5, 0.9, 0.99` |

**Vì sao dò `epsilon` theo tỉ lệ `epsilon/eta`:** độ lệch chuẩn nhiễu mỗi bước
odLD là `sqrt(2·eta·epsilon)`, còn bước gradient là `eta·|g|`, nên tỉ số
nhiễu/tín hiệu chỉ phụ thuộc `epsilon/eta`. Dò theo tỉ lệ giữ nguyên *chế độ*
Langevin khi `eta` đổi: `r=1e-6` ≈ tất định (regime "viscosity-vanishing" của
paper), `r=1e-2` = Langevin nhẹ, `r=1e0` = khám phá mạnh (nhiễu ~ bước gradient).

`eta` và `beta0` ảnh hưởng lẫn nhau (tốc độ học hiệu dụng
`LR_eff ≈ beta0 · (n_k/2) · eta`). Vì vậy **loại trước 2 góc phân kỳ**
`eta=3e-2 × beta0 ∈ {0.9, 0.99}` (LR_eff ≈ 0.08–0.09, nổ với model train from
scratch) — khai báo ở khối `exclude:` trong `stage1_odld.yaml`. Dải `eta` cũng
dịch lên so với paper vì `1e-3` bị stall (theo `00_MCSDCA_paper_experiments`).

`n_k` và `gamma_k` **không dò** — chúng tự cập nhật trong lúc train theo lịch
tăng của paper: `n_k = b̄ + ⌊(k+1)^λ⌋` (`langevin_steps=5` = `b̄`,
`langevin_steps_power=0.1` = `λ`) và `gamma_k = gamma_0 · (k+1)^β`
(`gamma_power=0.1` = `β`). Các tham số cố định khác: `max_langevin_steps=8`,
`burn_in=2`, `local_entropy_time (t)=1e4`, `max_grad_norm=1.0`.

### 3.2 MCSDCA-udLD

| Tham số | Ý nghĩa | Các giá trị thử |
|---|---|---|
| `--mcsdca-delta` / `mcsdca_delta` | độ lớn bước Langevin (bản underdamped) | `0.03, 0.1, 0.3` |

`epsilon` và `beta0` **lấy luôn từ kết quả tốt nhất của odLD**, không dò lại.
udLD nhận `epsilon` tuyệt đối (không phải tỉ lệ), nên tính
`epsilon = mcsdca_epsilon_ratio · mcsdca_eta` từ winner odLD. Stage 0 đã thử sẵn
`delta=0.3` để bắt phân kỳ.

### 3.3 AdamW (mốc so sánh)

| Tham số | Các giá trị thử |
|---|---|
| `--lr` / `lr` | `2e-5, 5e-5, 1e-4` |
| `--weight-decay` / `weight_decay` | `0.0, 1e-3, 1e-2` |

Thêm `1e-2` để AdamW cũng được điều chuẩn tường minh một cách công bằng (giả
thuyết là MCSDCA điều chuẩn *ngầm* tốt hơn, nên không để AdamW thiệt về mặt này).

---

## 4. Cách chọn cấu hình tốt nhất

`src/sweep.py` (hàm `select_best`) tự làm — và `run_experiment1.py` gọi lại đúng
hàm này.

**Bước 1 — loại bỏ các lần chạy hỏng:**

- `status` không phải `ok` (train bị lỗi hoặc phân kỳ)
- `val_mse` vô hạn hoặc `≥ 10` (số nổ)
- `pred_latent_variance < 1e-5` (latent bị co về một điểm — "collapse")
- `|latent_norm_drift| > target_latent_norm` (độ lớn latent khi rollout trôi quá xa)

**Bước 2 — xếp hạng phần còn lại (theo *trung bình các seed*):**

Các dòng chỉ khác nhau ở `seed` được gộp thành **một cấu hình**; cấu hình bị
loại nếu *bất kỳ* seed nào `status != ok`, hoặc nếu *trung bình* các seed không
qua các cổng ở Bước 1.

1. Ưu tiên: `rollout_mse_5` trung bình nhỏ nhất (sai số rollout tầm nhìn 5 bước, val).
2. Nếu bằng nhau: `val_mse` trung bình nhỏ nhất (sai số dự đoán 1 bước).

---

## 5. Cách nhanh — chạy 1 lệnh (`run_experiment1.py`)

```bash
python src/run_experiment1.py --data <phần_trăm>
```

`--data` là **phần trăm dữ liệu train PushT** dùng cho lần chạy này. Giá trị `< 1`
tự động bật **chế độ test luồng** (lưới nhỏ xíu, ngân sách nhỏ, 1 seed) — chỉ để
kiểm tra pipeline chạy trơn từ đầu đến cuối.

Một lệnh sẽ chạy tuần tự 5 phần cho mức dữ liệu đó:

| Phần | Làm gì |
|---|---|
| 1. sanity | odLD + udLD chạy vài bước, kiểm tra số không nổ |
| 2. dò odLD | lưới `(eta × epsilon/eta × beta0)` trừ 2 góc phân kỳ = 21 điểm, **× 2 seed** (`--tune-seeds`), xếp hạng theo trung bình seed |
| 3. dò udLD | lưới `delta` (1 seed), lấy `epsilon`/`beta0` từ winner odLD |
| 4. dò AdamW | lưới `(lr × weight_decay)` = 9 điểm (1 seed), tự xếp hạng |
| 5. đánh giá | chạy cả 3 optimizer với cấu hình winner, nhiều seed, **cùng một ngân sách cố định** |

### Trình tự khuyến nghị

```bash
# 1) test luồng: vài phút, chỉ để chắc pipeline không lỗi
python src/run_experiment1.py --data 0.01

# 2) dò tham số một lần ở mức 10%, rồi TÁI SỬ DỤNG cho các mức khác
python src/run_experiment1.py --data 10

# 3) các mức còn lại: bỏ qua bước dò, dùng lại winner của mức 10%
python src/run_experiment1.py --data 1   --reuse-winners outputs/experiment1/data10/winners.json
python src/run_experiment1.py --data 50  --reuse-winners outputs/experiment1/data10/winners.json
python src/run_experiment1.py --data 100 --reuse-winners outputs/experiment1/data10/winners.json
```

> Vì sao tái sử dụng winner? Để **mức dữ liệu là biến duy nhất thay đổi**. Nếu dò
> lại tham số ở từng mức, ta không biết chênh lệch kết quả là do optimizer hay do
> tham số khác nhau. Nếu vẫn muốn dò riêng cho từng mức thì bỏ `--reuse-winners`.

### Kết quả sinh ra

Trong `outputs/experiment1/data<X>/`:

- `winners.json` — cấu hình tốt nhất của 3 optimizer ở mức dữ liệu đó
- `sweep_odld/`, `sweep_udld/`, `sweep_adamw/` — chi tiết từng lần dò + `sweep_summary.csv`
- `comparison.csv` — mỗi dòng là `(optimizer × seed)` với đầy đủ metric
- `comparison_agg.csv` — trung bình và độ lệch chuẩn trên các seed

Chạy xong cả 5 mức thì gộp các file `comparison_agg.csv` lại để vẽ đồ thị
"kết quả theo tỉ lệ dữ liệu".

### Các tuỳ chọn hay dùng

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--data` | (bắt buộc) | phần trăm dữ liệu PushT |
| `--seeds` | `3072,3073,3074` | các seed cho phần đánh giá |
| `--tune-seeds` | `3072,3073` | seed cho lưới odLD (xếp hạng theo trung bình); test luồng ép về 1 |
| `--reuse-winners PATH` | — | bỏ qua bước dò, nạp `winners.json` có sẵn |
| `--only {all,tune,eval}` | `all` | chỉ chạy phần dò, hoặc chỉ chạy phần đánh giá |
| `--tune-budget` | `3000` (test luồng: 60) | số `backward()` cho mỗi điểm lưới khi dò |
| `--eval-budget` | `8000` (test luồng: 120) | ngân sách cố định cho phần đánh giá, **giống nhau ở mọi mức dữ liệu** |
| `--precision` | `fp32` | đổi `bf16` để chạy nhanh ~1.8× (đánh đổi: thêm nhiễu làm tròn) |
| `--batch-size` | `64` | |
| `--device` | `cuda` | |
| `--dry-run` | — | chỉ in kế hoạch rồi thoát |

---

## 6. Cách thủ công — nhiều bước (khi cần kiểm soát kỹ)

> Chạy lần lượt. Sau mỗi bước: `--select-only` để xem xếp hạng → chép tay cấu hình
> tốt nhất vào file yaml của bước sau → mới chạy tiếp.

| Bước | Để làm gì | profile | data | ngân sách | seed | số điểm lưới | File config |
|---|---|---|---|---|---|---|---|
| **0 — Kiểm tra nhanh** | odLD **và** udLD chạy, số không nổ (gồm `eta=3e-2`, `delta=0.3`), `sampler_loss` giảm | `smoke` | 5% | 120 | 1 | 2×2×1×2 = 8 (×2 opt) | `stage0_sanity.yaml` |
| **1 — Dò thô odLD** | quét `eta` / `epsilon/eta` / `beta0`, trừ 2 góc phân kỳ | `small` | 10% | 3000 | 1 | 3×3×3 − 6 = 21 | `stage1_odld.yaml` |
| **2 — Dò tinh odLD** | thu hẹp quanh **3 cấu hình** tốt nhất của bước 1 | `small` | 10% | 5000 | **2** | ~2×2×2 × 2 seed | `stage2_odld.yaml` (sửa lưới) |
| **3 — Xác nhận odLD** | chạy lại cấu hình tốt nhất với 3 seed | `small` | 10% | 8000 | 3 | 1 | `stage3_confirm_odld.yaml` |
| **4 — Dò udLD** | quét `delta`, lấy `epsilon`/`beta0` từ odLD | `small` | 10% | 5000 | 1 → 3 | 3 | `stage4_udld.yaml` |
| **B — Dò AdamW** | tìm `lr` / `weight_decay` | `small` | 10% | 5000 | 1 → 3 | 3×3 = 9 | `baseline_adamw.yaml` |
| **Cuối — Ablation** | vòng đánh giá chính thức nhiều mức dữ liệu | `small` | 10/25/50/100% | **8000 cố định** | 3 | cấu hình đã chốt | `configs/ablation.yaml` |

`select_best` gộp các dòng chỉ khác `seed` và xếp hạng theo **trung bình seed**,
nên bước 2 chỉ cần thêm trục `seed: [3072, 3073]` vào lưới.

Lệnh mẫu cho mỗi bước:

```bash
python src/sweep.py configs/sweep/stage1_odld.yaml
python src/sweep.py configs/sweep/stage1_odld.yaml --select-only            # xem xếp hạng
python src/sweep.py configs/sweep/stage1_odld.yaml --select-only --optimizer MCSDCA-odLD
```

Kết quả bước trước chuyển sang bước sau:

```
Bước 1 (dò thô odLD)   -- 3 cấu hình đầu --> lưới bước 2
Bước 2 (dò tinh odLD)  -- cấu hình tốt nhất --> lưới bước 3 (giá trị đơn)
Bước 3 (xác nhận odLD) -- cấu hình chốt --> configs/ablation.yaml, mục MCSDCA-odLD
                                       \--> bước 4: epsilon = ratio*eta, beta0
Bước 4 (udLD)          -- cấu hình chốt --> configs/ablation.yaml, mục MCSDCA-udLD
Bước B (AdamW)         -- cấu hình chốt --> configs/ablation.yaml, mục AdamW
```

Rồi chạy vòng đánh giá:

```bash
python src/run_ablation.py configs/ablation.yaml --dry-run   # kiểm tra config
python src/run_ablation.py configs/ablation.yaml             # chạy thật
```

Sinh `outputs/ablation/<thời_gian>/ablation.csv` — mỗi dòng là một tổ hợp
`(optimizer × tỉ lệ dữ liệu × seed)`.

---

## 7. Đọc kết quả

Từ `comparison_agg.csv` (cách nhanh) hoặc `ablation.csv` (cách thủ công), vẽ:

1. **`rollout_mse_5` theo tỉ lệ dữ liệu** — 3 đường cho 3 optimizer. Kỳ vọng:
   khoảng cách giữa MCSDCA và AdamW *rộng ra* khi tỉ lệ dữ liệu giảm.
2. **`train_val_gap` theo tỉ lệ dữ liệu** — xem MCSDCA có ít overfit hơn khi dữ
   liệu ít không.
3. **`rollout_mse_1` / `3` / `5`** — sai số cộng dồn theo tầm nhìn.
4. **Bảng độ ổn định** — đếm số lần chạy bị loại (`status` lỗi, collapse, drift)
   theo từng optimizer.

Cập nhật số vào `01_EXPERIMENT1/MCSDCA_Predictor_Optimizer_Report.md`.

---

## 8. Cấu hình mặc định (đã chỉnh cho máy ảo thuê)

**Máy:** Windows 10 · i7-12700KF (12 nhân / 20 luồng) · 1× RTX 3090 (24 GB VRAM) ·
56 GB RAM · SSD NVMe 1600 GB.

Đã cập nhật sẵn trong code / config:

| Chỗ | Trước | Giờ | Lý do |
|---|---|---|---|
| profile `small`: `batch_size` | 32 | **64** | 24 GB VRAM thừa sức, ít step hơn / epoch |
| profile `small`: `sigreg_num_proj` | 256 | **512** | ước lượng SIGReg ổn định hơn |
| profile `small`: `eval_batch_size` / `eval_train_batches` / `val_batches` | 4 / 8 / 8 | **8 / 16 / 16** | đánh giá bớt nhiễu → xếp hạng đáng tin hơn |
| các file `configs/sweep/*.yaml`: `device` | `auto` | **`cuda`** | ép dùng GPU |
| `stage1..4`, `baseline`: `backprop_budget` | 14000 / (trống) | **3000 / 5000 / 8000** | đủ để xếp hạng, không cần hội tụ; ~1–2 h mỗi sweep |
| `configs/ablation.yaml`: `budget` | 5000 | **8000** | ngân sách cố định cho vòng đánh giá |
| `run_experiment1.py`: `--precision` | — | **`fp32`** mặc định | tránh nhiễu bf16 làm lẫn vào so sánh; có thể bật `bf16` để chạy nhanh |
| lưới odLD: `mcsdca_eta` | `1e-3, 3e-3, 1e-2` | **`3e-3, 1e-2, 3e-2`** | `1e-3` stall (theo `00_MCSDCA_paper_experiments`) |
| lưới odLD: nhiễu | `mcsdca_epsilon` tuyệt đối `1e-8, 1e-4, 1e-2` | **`mcsdca_epsilon_ratio` `1e-6, 1e-2, 1e0`** (`epsilon = ratio·eta`) | tỉ số nhiễu/tín hiệu chỉ phụ thuộc `epsilon/eta` |
| lưới odLD: `exclude` | — | **bỏ `eta=3e-2 × beta0∈{0.9,0.99}`** | `LR_eff ≈ 0.08–0.09` → phân kỳ, khỏi tốn slot |
| lưới odLD: seed | 1 | **2** (`--tune-seeds`, xếp hạng theo trung bình) | tránh chọn theo may seed trong lưới 21 điểm |
| `stage0_sanity.yaml` | chỉ odLD | **odLD + udLD**, gồm `eta=3e-2`, `delta=0.3` | bắt phân kỳ udLD ngay ở sanity |
| lưới AdamW: `weight_decay` | `0.0, 1e-3` | **`0.0, 1e-3, 1e-2`** | AdamW cũng được điều chuẩn tường minh công bằng |
| `select_best` | xếp hạng theo 1 dòng | **gộp seed, xếp hạng theo trung bình**, loại nếu bất kỳ seed nào hỏng | chọn cấu hình ổn định giữa các seed |

**Bắt buộc:** dùng `.venv\Scripts\python.exe` (bản `torch` có CUDA). Không dùng
`python` hệ thống nếu đó là bản chỉ chạy CPU.

Nên đặt trước khi chạy:

```bash
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
```

**Thời gian ước tính trên RTX 3090** (batch 64, fp32, profile `small`):

| Lệnh | Thời gian |
|---|---|
| `--data 0.01` (test luồng) | ~3–8 phút |
| `--data 10` (dò đầy đủ + đánh giá 3 seed) | ~5–7 giờ (lưới odLD ×2 seed) |
| `--data 1 / 50` (`--reuse-winners`, chỉ đánh giá) | ~40–90 phút |
| `--data 100` (`--reuse-winners`, chỉ đánh giá) | ~1.5–3 giờ |
| Cả 5 mức | ~1–1.5 ngày |

Muốn nhanh hơn ở bước dò: `--tune-seeds 3072` (về 1 seed như cũ) hoặc giảm
`--tune-budget`.

Nếu cần nhanh hơn: thêm `--precision bf16` (nhanh ~1.8×) hoặc giảm `--eval-budget`
(nhớ giữ **cùng** giá trị cho mọi mức dữ liệu).

---

## 9. Danh sách việc cần làm

- [ ] Đã kích hoạt `.venv`, kiểm tra `torch.cuda.is_available()` trả về `True`
- [ ] Đặt `PYTORCH_CUDA_ALLOC_CONF` như trên
- [ ] `python src/run_experiment1.py --data 0.01` chạy hết, không lỗi
- [ ] `python src/run_experiment1.py --data 10` → có `outputs/experiment1/data10/winners.json`
- [ ] Xem lại `winners.json` mức 10% cho hợp lý (không phải giá trị ở rìa lưới)
- [ ] `--data 1 / 50 / 100` với `--reuse-winners` của mức 10%
- [ ] Gộp 5 file `comparison_agg.csv` → vẽ đồ thị theo tỉ lệ dữ liệu
- [ ] Cập nhật `MCSDCA_Predictor_Optimizer_Report.md`
