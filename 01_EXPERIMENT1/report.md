# Báo cáo về quy trình thực nghiệm tối ưu LeWM: thay AdamW bằng thuật toán MCSDCA

---

## Tóm tắt

LeWM (Latent World Model) là một dạng mô hình thế giới (world model) xây dựng dựa trên kiến trúc JEPA (Joint-Embedding Predictive Architecture), thay vì tái tạo lại điểm ảnh, nó học và dự đoán trạng thái tiếp theo trực tiếp trong không gian vector tiềm ẩn. Cụ thể, LeWM học đồng thời một encoder ánh xạ quan sát sang một không gian tiềm ẩn và một predictor học quy luật vận động để dự đoán trạng thái tiếp theo của thế giới thực trong không gian tiềm ẩn đó. Mô hình được huấn luyện tự giám sát thông qua hàm mất mát kết hợp giữa mục tiêu dự đoán vector đặc trưng kế tiếp và áp dụng một kĩ thuật điều chuẩn SIGReg để tránh tình trạng sụp đổ biểu diễn (representation collapse) - các vector dự đoán bị suy biến thành hằng số.

Thí nghiệm này nhằm tối ưu hóa quá trình huấn luyện mô hình thế giới LeWM bằng cách thay thế bộ tối ưu AdamW bằng thuật toán MCSDCA (Markov Chain Stochastic Difference-of-Convex Algorithm), qua đó nâng cao chất lượng dự đoán rollout tiềm ẩn phục vụ khâu lập kế hoạch bằng CEM (Cross-Entropy Method).

Cụ thể, toàn bộ mô hình LeWM được huấn luyện lại từ đầu với hàm mục tiêu giữ nguyên. Việc lựa chọn MCSDCA xuất phát từ sự tương thích chặt chẽ về mặt lý thuyết với bản chất của bài toán: hàm mất mát phi lồi theo tham số mạng, gradient có tính ngẫu nhiên từ mini-batch và dữ liệu quỹ đạo mang tính phụ thuộc chuỗi Markov (vi phạm giả thiết độc lập i.i.d).

Khác với các thuật toán tối ưu bậc một sử dụng rộng rãi như Adam, SGD, MCSDCA giải bài toán quy hoạch DC ngẫu nhiên có tính đến bất định nội sinh, đồng thời gián tiếp tối ưu một phiên bản local-entropy làm mịn của hàm mất mát, theo nguyên tắc cơ chế này giúp hướng các tham số mô hình hội tụ vào vùng cực tiểu phẳng, mang lại khả năng tổng quát hóa ổn định hơn.

**Giả thuyết cần kiểm chứng:**
- Hiệu năng và độ ổn định của quá trình huấn luyện: MCSDCA có thể thay thế hiệu quả AdamW, giúp hạ thấp sai số dự đoán biểu diễn kế tiếp và duy trì sự ổn định của biểu diễn tiềm ẩn, đặc biệt trong điều kiện dữ liệu quỹ đạo bị giới hạn.
- Chất lượng lập kế hoạch: Việc cải thiện độ chính xác của quá trình dự đoán chuỗi dài trong không gian tiềm ẩn sẽ trực tiếp gia tăng hiệu quả tối ưu hóa quỹ đạo hành động của thuật toán CEM ở khâu sau.

**Giả thuyết 1**

Thực nghiệm hiện đã chạy các bộ thực nghiệm trên PushT, so sánh AdamW / MCSDCA-odLD / MCSDCA-udLD trên cùng trục số lần lan truyền ngược:

**Thực nghiệm lần 1:**

| Bộ | Cấu hình | Dữ liệu | Ngân sách backprop | Seed |
|---|---|---:|---:|---:|
| *Set 1* | có tinh chỉnh siêu tham số cả hai phía ở 1% dữ liệu | 10% | 8 000 | 3 |
| *Set 2* | không tinh chỉnh, sử dụng cấu hình theo paper | 4% | 40 000 | 1 |

**Kết quả chính:** ở cả hai mức dữ liệu, *AdamW cho sai số dự đoán thấp hơn MCSDCA rõ rệt* — one-step MSE thấp hơn ~20 lần, rollout 5 bước thấp hơn ~6–8 lần. Qua tiến hành điều tra thì MCSDCA cho kết quả kém do sụp đổ biểu diễn trong quá trình training ở cả 2 thuật toán tối ưu đang xét, do nhiều nguyên nhân.

- Đối với set 1, việc thay đổi các tham số huấn luyện cho MCSDCA làm tăng một lượng lớn nhiễu trong quá trình cập nhật, việc dữ liệu quá ít cũng là nguyên nhân khi tinh chỉnh tham số khiến thuật toán MCSDCA hoạt động không ổn định.
- Đối với set 2, cấu hình MCSDCA đặc biệt là số bước Langevin lớn, burn-in nhiều, làm cho thuật toán MCSDCA cập nhật ít hơn 20 lần nếu so với AdamW khi ở cùng một budget; việc sử dụng norm trong quá trình tính toán cũng có thể là nguyên nhân dẫn đến sự sụp đổ biểu diễn nhẹ trong quá trình huấn luyện MCSDCA.

**Thực nghiệm lần 2:**

| Bộ | Cấu hình | Dữ liệu | Ngân sách backprop | Seed |
|---|---|---:|---:|---:|
| *Set 1* | không tinh chỉnh | 4% | 40 epoch ~ | 1 |
| *Set 2* | Thay đổi số bước Langevin, số bước cập nhật ×3 | 4% | 40 epoch ~ | 1 |

**Kết quả chính:** Tôi đang cập nhật, kết quả đã tốt hơn nhiều, không còn sụp đổ biểu diễn.

---

## I. Bối cảnh, phương pháp đề xuất, quá trình thực nghiệm

### 1.1. Bài toán

Toàn bộ pipeline gồm ba khâu: *huấn luyện world model (encoder + predictor + phụ trợ) → rollout tưởng tượng trong latent → CEM tối ưu chuỗi hành động*. Thí nghiệm này chỉ động tới khâu đầu.

```text
observation o_t --[encoder]--> latent z_t
latent z_t + action a_t --[predictor]--> latent dự đoán ẑ_{t+1}
rollout ẑ_{t+1}, ẑ_{t+2}, ... --[CEM / MPC]--> chuỗi action
```

Với một cửa sổ trajectory, LeWM mã hóa toàn bộ khung thành embedding rồi dự đoán khung kế bằng teacher forcing:

$$z_t = \text{encoder}_\theta(o_t),\qquad\hat z_{t+1} = \text{predictor}_\phi(z_{1:t}, a_{1:t}),\qquad z_{t+1} = \text{encoder}_\theta(o_{t+1}).$$

Hàm mục tiêu huấn luyện (`training_objective` trong `src/run_pusht_predictor_experiment.py`, khớp `le-wm/train.py::lejepa_forward`):

$$L = L_\text{pred} + \lambda\,\text{SIGReg}(Z),\qquad L_\text{pred} = \big\lVert \hat z_{t+1} - z_{t+1}\big\rVert^2,\qquad \lambda = 0.09,$$

trong đó $Z$ là toàn bộ embedding của cửa sổ (mọi khung, cả context lẫn target). *Điểm quan trọng:* target $z_{t+1}$ do chính encoder đang huấn luyện sinh ra và *không stop-gradient*. Do đó $L_\text{pred}$ một mình có thể bị cực tiểu hóa bằng nghiệm tầm thường — encoder co mọi embedding về một điểm để $\hat z \equiv z$. LeWM chống điều này chỉ bằng SIGReg (ép $Z$ về phân phối Gaussian isotropic; không dùng stop-gradient, không dùng EMA teacher).

Nói cách khác, đây là bài toán học biểu diễn + học dynamics đồng thời, và điểm cân bằng chống sụp đổ hoàn toàn nằm ở SIGReg — một điều kiện dễ vỡ khi đổi optimizer. Ta vẫn báo cáo các chỉ số phía predictor (sai số dự đoán 1 bước, rollout latent) vì đó là thứ khâu planning tiêu thụ trực tiếp; nhưng thứ đang được tối ưu là cả world model, và một encoder sụp đổ sẽ làm mọi chỉ số này vô nghĩa dù loss huấn luyện nhỏ.

### 1.2. Giả thuyết: vì sao MCSDCA có thể thay AdamW

Muốn áp dụng MCSDCA, hàm mục tiêu cần phải thỏa các điều kiện sau — và trong bài toán đang xét tất cả đều thỏa mãn. Ở đây $x$ là *toàn bộ trọng số của 5 module LeWM* (không chỉ predictor), $f(x)$ là loss huấn luyện ở §1.1.

| Câu hỏi | Trả lời cho bài toán huấn luyện LeWM (toàn mô hình) |
|---|---|
| Hàm mục tiêu không lồi? | *Có.* $f(x)$ là hợp thành phi tuyến của ViT encoder, transformer predictor tự hồi quy, các MLP `projector`/`pred_proj` có BatchNorm, và SIGReg (random projection + sin/cos). Target không stop-gradient nên $f$ còn có cả nghiệm sụp đổ tầm thường lẫn nghiệm hữu ích — landscape nhiều cực tiểu chất lượng rất khác nhau. |
| Gradient ngẫu nhiên? | *Có.* Huấn luyện bằng mini-batch (batch 64 ở small, 128 ở paper) ⇒ $\nabla \hat f_B(x)$ là stochastic gradient. |
| Dữ liệu Markov / không i.i.d.? | *Có.* Trajectory $\tau=(o_0,a_0,\dots,o_T)$ với $o_{t+1}\sim P(o_{t+1}\mid o_t,a_t)$. Hai transition $X_t=(o_t,a_t,o_{t+1})$ và $X_{t+1}$ chia sẻ $o_{t+1}$ nên phụ thuộc nhau; các cửa sổ trích từ cùng episode chồng lấn khung. Lập luận y hệt trong không gian latent. Đây là điểm khớp quan trọng nhất: MCSDCA được thiết kế riêng cho tình huống không lấy được mẫu i.i.d. |
| Đưa được về dạng DC? | *Có* (qua PDE / local-entropy — mục II.3–II.4). Tính không trơn thì chưa kết luận được: pred_loss (MSE) và SIGReg hiện dùng toàn phép khả vi, nên luận điểm mạnh không nằm ở "loss không trơn" mà ở ba điểm trên. |

**Kỳ vọng nghiên cứu:** MCSDCA không chỉ là "AdamW đổi learning rate". Nó tối ưu bản local-entropy-smoothed của loss, dùng chuỗi Markov ước lượng phần subgradient khó tính, nên về nguyên tắc ưu tiên vùng nghiệm phẳng — từ đó có thể giảm lỗi tích lũy trong rollout dài. Toàn bộ báo cáo này là để kiểm chứng kỳ vọng đó.

### 1.3. Phương pháp đề xuất

- *Huấn luyện joint cả 5 module từ đầu* (encoder, projector, action_encoder, predictor, pred_proj) — `select_full_model_parameters` bật grad cho cả năm, optimizer (AdamW hoặc MCSDCA) cập nhật toàn bộ tập tham số này. Không đóng băng encoder, không train predictor-only. Bản mô tả "MCSDCA làm optimizer riêng cho predictor" trong các doc cũ (`note.txt`, phần II.4 của `MCSDCA_Predictor_Optimizer_Report.md`) *không còn đúng* với code hiện tại.
- **So sánh trên trục backprop_calls, không theo epoch hay wall-clock.** Một outer step MCSDCA thực hiện nhiều lượt `backward()` (mỗi bước Langevin nội chain_length − 1 lượt), nên đo theo epoch là không công bằng. Mọi optimizer nhận *cùng ngân sách backprop*, *cùng trọng số khởi tạo* (lưu 1 lần theo seed, `restore_state` nạp lại y hệt trước mỗi optimizer), *cùng luồng mini-batch* (cùng seed). MCSDCA lấy mini-batch mới cho từng bước Langevin, đúng như Algorithm 2/3 của paper. Khác biệt phụ: AdamW chạy kèm lịch learning-rate kiểu LeWM (1% warmup + cosine, `make_lewm_lr_scheduler`); MCSDCA không có lịch LR — "lịch" của nó là chuỗi Markov dài dần và γ_k tăng dần.
- *Cả hai phía đều được tinh chỉnh* (Set 1) để không so "MCSDCA đã chỉnh" với "AdamW mặc định". Set 2 thì ngược lại — cố tình không chỉnh, lấy thẳng cấu hình từ paper gốc của mỗi bên, để có một mốc tái lập trung thành.

### 1.4. Cấu hình chung của Thực nghiệm 1 (Set 1 / Set 2)

*Đây là cấu hình cho Thực nghiệm 1 — bộ đã chạy đầy đủ và phân tích chi tiết ở Phần III (§3.2). Thực nghiệm 2 (thay đổi số bước Langevin, tăng số bước cập nhật) có cấu hình riêng, trình bày tại §3.3.*

| | Set 1 — có tinh chỉnh | Set 2 — theo paper |
|---|---|---|
| Mục tiêu | so sánh công bằng sau khi cả hai phía đã tối ưu siêu tham số | tái lập trung thành + quan sát hành vi hội tụ dài hạn |
| Script | `src/run_experiment1.py --data 10` | `src/run_experiment1.py --data 4 --profile paper` |
| Dữ liệu PushT | 10% (178 282 / 24 228 cửa sổ train / val) | 4% |
| Ngân sách backprop | 8 000 (≈ 2.87 epoch tương đương) | 40 000 (≈ 40 epoch, đúng con số paper MCSDCA dùng) |
| Seed | 3 (3072–3074) | 1 |
| Batch / precision / SIGReg proj | 64 / bf16 / 512 (profile small) | 128 / bf16 / 1024 (profile paper, khớp `le-wm/config/train/lewm.yaml`) |
| AdamW | winner từ lưới lr × weight_decay | ghim: lr=5e-5, weight_decay=1e-3 (config LeWM) |
| MCSDCA | winner từ lưới (mục 3.2.1) | `MCSDCAConfig.paper()`: n_k = 20 + ⌊(k+1)^0.1⌋, discard 10, ε=1e-8, bước Langevin 1e-3, γ_0=1e-9 |

Set 1 còn có sẵn hai mức phụ 1% và 10% để dựng đường "kết quả theo tỉ lệ dữ liệu" (mức 0.01% là chế độ test luồng, bỏ khỏi phân tích).

### 1.5. Khó khăn và các quyết định thiết kế đã gặp

| Vấn đề | Xử lý |
|---|---|
| Góc lưới eta = 3e-2 × beta0 ∈ {0.9, 0.99} cho LR_eff ≈ β0·(n_k/2)·eta ≈ 0.08–0.09 ⇒ nổ khi train from scratch | loại trước hai góc này khỏi lưới odLD (exclude trong `stage1_odld.yaml`) |
| eta = 1e-3 (giá trị paper) bị stall, sai số không giảm | dịch dải eta lên {3e-3, 1e-2, 3e-2} cho Set 1 |
| Nhiễu Langevin √(2·eta·ε) phụ thuộc eta, khó dò ε tuyệt đối | dò ε theo *tỉ lệ* ε/eta ∈ {1e-6, 1e-2, 1e0} để cố định chế độ nhiễu/tín hiệu |
| Một outer step MCSDCA = nhiều `backward()` | so sánh theo backprop_calls, không theo epoch |
| MCSDCA đắt: cùng ngân sách backprop thì đi được ít outer step hơn | chấp nhận; đây là một phần của phép so sánh công bằng |
| Set 2 mới chạy 1 seed | kết quả Set 2 chỉ mang tính định hướng, chưa có sai số chuẩn |
| Kết quả `outputs/` sinh trên máy Windows (`D:\DCA\MCSDCA`) | số trong báo cáo lấy từ output đã lưu trong hai notebook; hình hội tụ regenerate bằng notebook (mục III) |

### 1.6. Nhận định sơ bộ

MCSDCA gắn được vào pipeline LeWM và chạy ổn định về mặt số học (không nổ, status = ok ở mọi seed), nhưng trong ngân sách hiện tại *chưa vượt AdamW ở bất kỳ chỉ số dự đoán nào* và đang kéo latent về trạng thái sụp đổ. Chi tiết ở phần III.

---

## II. Cơ sở lý thuyết

### 2.1. Từ bài toán tối ưu đến quy hoạch DC

Xét $\min_x F(x)$. Trong deep learning $F$ thường không lồi, nhiều cực tiểu cục bộ, bề mặt loss gồ ghề. *Quy hoạch DC* (Difference of Convex) phân tách

$$F(x) = G(x) - H(x),\qquad G,\,H \text{ lồi}.$$

Thuật toán kinh điển *DCA* giải bằng chuỗi bài toán con lồi: tại vòng $k$, lấy $y_k \in \partial H(x_k)$, tuyến tính hóa $H(x)\approx H(x_k)+\langle x-x_k,y_k\rangle$, rồi

$$x_{k+1}\in\arg\min_x\{G(x)-\langle x,y_k\rangle\}.$$

DCA hướng tới *điểm tới hạn* (critical point), không đảm bảo tối ưu toàn cục.

### 2.2. MCSDCA: DCA + ước lượng subgradient bằng chuỗi Markov

MCSDCA giải lớp bài toán DC *ngẫu nhiên có bất định nội sinh*: subgradient của $H$ không tính trực tiếp được mà có dạng kỳ vọng

$$\mathbb{E}_{P(\xi\mid x)}[v(x,\xi)] \in \partial H(x),$$

trong đó phân phối lấy mẫu $P(\xi\mid x)$ *phụ thuộc chính biến quyết định* $x$ (đây là bất định nội sinh). Không có nguồn mẫu i.i.d. cố định ⇒ dùng chuỗi Markov có phân phối cân bằng $P(\xi\mid x_k)$.

Tại outer iteration $k$: chạy chuỗi $\xi_0^k\to\cdots\to\xi_{n_k-1}^k$, bỏ $b$ mẫu burn-in, ước lượng

$$y_k = \frac{1}{n_k-b}\sum_{i=b}^{n_k-1} v(x_k,\xi_i^k)$$

(một ước lượng có thể chệch vì các mẫu Markov phụ thuộc nhau — điểm mạnh của paper là vẫn chứng minh được hội tụ), rồi giải bài toán con proximal

$$x_{k+1}\in\arg\min_x\Big\{G(x)-\langle x,y_k\rangle+\tfrac{\gamma_k}{2}\lVert x-x_k\rVert_A^2\Big\},\qquad \gamma_k>0,\; A\succeq I.$$

Ngắn gọn: *MCSDCA = DCA + stochastic approximation + Markov sampling*.

### 2.3. PDE regularization và local entropy

Thay vì cực tiểu trực tiếp loss thô $f(x)$ (rất gồ ghề), ta tối ưu bản làm mịn của nó sinh bởi phương trình Hamilton–Jacobi

$$u_t + \tfrac12\langle\nabla u,\mathcal H^{-1}\nabla u\rangle = 0,\qquad u(x,0)=f(x),$$

với $\mathcal H$ đối xứng xác định dương. Theo công thức *Hopf–Lax*, nghiệm viscosity là inf-convolution

$$u(x,t) = \inf_{x'}\Big\{f(x') + \tfrac{1}{2t}\langle x'-x,\mathcal H(x'-x)\rangle\Big\}.$$

Diễn giải: chọn $x'$ có loss thấp nhưng bị phạt nếu quá xa $x$ ⇒ đỉnh sắc bị san, thung lũng cực tiểu được mở rộng, ưu tiên vùng loss thấp rộng. Để tránh tính không trơn của $\inf$, dùng xấp xỉ log-sum-exp (local entropy):

$$\tilde u(x,t) := -\varepsilon\log\int_{\mathbb R^n}\exp\!\Big(-\tfrac{1}{\varepsilon}\big[f(x')+\tfrac{1}{2t}\langle x'-x,\mathcal H(x'-x)\rangle\big]\Big)\,dx',$$

và $\tilde u(x,t)\to u(x,t)$ khi $\varepsilon\to 0^+$. $\varepsilon$ kiểm soát mức làm trơn.

### 2.4. Phân tách DC của local entropy

Khai triển $\langle x'-x,\mathcal H(x'-x)\rangle = x'^\top\mathcal Hx' - 2x^\top\mathcal Hx' + x^\top\mathcal Hx$ cho

$$G(x) = \tfrac{1}{2t}\,x^\top\mathcal H x\quad\text{(bậc hai, lồi, trơn)},$$

$$H(x) = \varepsilon\log\int_{\mathbb R^n}\exp\!\Big(-\tfrac{1}{\varepsilon}\big[f(x')+\tfrac{1}{2t}x'^\top\mathcal Hx'-\tfrac1t x^\top\mathcal Hx'\big]\Big)dx'\quad\text{(lồi nhờ cấu trúc log-sum-exp của các hàm affine theo }x).$$

Gradient của $H$:

$$\nabla H(x) = \tfrac1t\,\mathbb{E}_{p(x'\mid x)}[\mathcal H x'],\qquad p(x'\mid x)\propto\exp\!\Big(-\tfrac{1}{\varepsilon}\big[f(x')+\tfrac{1}{2t}x'^\top\mathcal Hx'-\tfrac1t x^\top\mathcal Hx'\big]\Big).$$

Với $\mathcal H = I$:

$$G(x)=\tfrac{1}{2t}\lVert x\rVert^2,\qquad\nabla H(x)=\tfrac1t\,\mathbb{E}_{p(x'\mid x)}[x'],\qquad p(x'\mid x)\propto \exp\!\Big(-\tfrac{1}{\varepsilon}\big[f(x')+\tfrac{1}{2t}\lVert x'-x\rVert^2\big]\Big).$$

Nghẽn duy nhất: tính kỳ vọng theo $p(x'\mid x)$ trên không gian tham số cực lớn. Đây là chỗ Langevin dynamics vào cuộc.

### 2.5. Langevin dynamics để lấy mẫu

Cần lấy mẫu từ $\pi(x)\propto\exp(-\beta U(x))$ mà không biết hằng số chuẩn hóa $Z$. Overdamped Langevin (liên tục):

$$dX_t = -\nabla U(X_t)\,dt + \sqrt{2\beta^{-1}}\,dB_t,$$

có phân phối bất biến $\pi$. Rời rạc Euler–Maruyama:

$$X_{i+1} = X_i - \eta\nabla U(X_i) + \sqrt{2\eta\beta^{-1}}\;\mathcal N(0,I).$$

Chọn $\beta = 1/\varepsilon$ và, trong bài toán local entropy với $\mathcal H=I$,

$$U(x') = f(x') + \tfrac{1}{2t}\lVert x'-x^k\rVert^2,\qquad\nabla U(x') = \nabla f(x') + \tfrac1t (x'-x^k).$$

Thay $\nabla f$ bằng stochastic gradient trên mini-batch $\widetilde\nabla f$, được bước *odLD*:

$$x_{i+1}^k = x_i^k - \eta\Big(\widetilde\nabla f(x_i^k) + \tfrac1t(x_i^k - x^k)\Big)+ \sqrt{2\eta\varepsilon}\;\mathcal N(0,I).$$

Ba thành phần: $\widetilde\nabla f$ kéo về vùng loss thấp; $\tfrac1t(x_i^k-x^k)$ giữ mẫu gần nghiệm hiện tại; nhiễu Gaussian giúp khám phá lân cận.

**udLD** thêm biến vận tốc $v$ (rời rạc theo Cheng et al. 2018):

$$dV_t = -\mu V_t\,dt - \nabla U(X_t)\,dt + \sqrt{2\mu\beta^{-1}}\,dB_t,\qquad dX_t = V_t\,dt,$$

quán tính giúp chuỗi đi xa theo hướng hợp lý thay vì random-walk zigzag, thường hội tụ tới phân phối đích nhanh hơn; đổi lại phải lưu thêm $v$ và công thức phức tạp hơn. Bước lấy `ud_delta` là kích thước bước rời rạc.

### 2.6. Giả mã và bước cập nhật DCA dạng đóng

**Algorithm 1 — MCSDCA tổng quát:** với $x_0$, dãy $\gamma_k>0$, độ dài chuỗi $n_k$, burn-in $b$, $A\succeq I$; mỗi vòng: (1) chạy chuỗi Markov cân bằng $P(\xi\mid x_k)$, (2) $y_k = \frac{1}{n_k-b}\sum_{i=b}^{n_k-1} v(x_k,\xi_i^k)$, (3) giải bài toán con proximal để ra $x_{k+1}$.

**Algorithm 2 — odLD / Algorithm 3 — udLD:** khởi tạo $x_0^k = x^k$ (udLD thêm $v_0^k=0$); chạy $n_k$ bước Langevin lấy mini-batch mới mỗi bước; bỏ $b$ mẫu đầu; lấy $y_k = \text{average}(x_b^k,\dots,x_{n_k-1}^k)$.

Với local entropy $\mathcal H = I$, bài toán con có *nghiệm đóng* (không cần solver):

$$x_{k+1} = \arg\min_x\Big\{\tfrac{1}{2t}\lVert x\rVert^2 - \tfrac1t\langle x,y_k\rangle+ \tfrac{\gamma_k}{2}\lVert x - x^k\rVert^2\Big\}= \underbrace{\frac{t\gamma_k}{1+t\gamma_k}}_{\alpha_k}\,x^k+ \underbrace{\frac{1}{1+t\gamma_k}}_{\beta_k}\,y_k.$$

Tức $x_{k+1}$ chỉ là *trộn tuyến tính* giữa nghiệm hiện tại $x^k$ và trung bình Markov $y_k$. Tham số `beta0` chính là $\beta_0$ (tỉ lệ của $y_k$ tại $k=0$); đặt `beta0` tương đương đặt $\gamma_0 = (1/\beta_0 - 1)/t$. odLD và udLD *chung* bước ngoài này, chỉ khác cách sinh chuỗi Markov bên trong.

### 2.7. Kết quả hội tụ cần hiểu đúng

- *Không tiệm cận:* sau hữu hạn $T$ bước, thuật toán đạt điểm $\epsilon$-critical hoặc nearly $\epsilon$-critical (tồn tại $\bar x$ với $\lVert\bar x - x^*\rVert = O(\epsilon)$ và $\text{dist}(\partial G(\bar x),\partial H(\bar x))\le\epsilon$) theo kỳ vọng, với tốc độ xác định, dưới giả thiết về $n_k$, $\gamma_k$, sai số bài toán con, tính $L$-smooth.
- *Tiệm cận:* $\sum_k \lVert x^{k+1}-x^k\rVert_A^2 < \infty$ hầu chắc chắn ⇒ $\lVert x^{k+1}-x^k\rVert_A \to 0$; nếu $\{x^k\},\{y^k\}$ bị chặn thì mọi điểm tụ là critical point của $F$. *Không* phải nghiệm tối ưu toàn cục.
- Kết quả còn đúng cho *chuỗi Markov không đồng nhất theo thời gian* (luật chuyển thay đổi theo bước) dưới giả thiết ergodic phù hợp — trường hợp xuất hiện tự nhiên với cơ chế lấy mẫu kiểu diffusion.

### 2.8. Kiến trúc LeWM (các tham số được huấn luyện)

Cấu hình tham chiếu (`le-wm/config/train/`), embed_dim = 192, history_size = 3, num_preds = 1 (dự đoán một bước):

| Module | Chi tiết |
|---|---|
| encoder | ViT-HF size *tiny*, patch_size 14, image_size 224, *không pretrained* (train from scratch) |
| predictor | ARPredictor — transformer nhân quả tự hồi quy: num_frames 3, dim 192, depth 6, heads 16, mlp_dim 2048, dim_head 64, dropout 0.1 |
| action_encoder | Embedder, input_dim gán lúc chạy = 5 × dim(action) (frameskip 5 ghép 5 action thô), emb_dim 192 |
| projector, pred_proj | MLP 192 → 2048 → 192 kèm BatchNorm1d — bù cho LayerNorm cuối encoder/predictor (LayerNorm phá cơ chế chống sụp đổ) |
| SIGReg | bộ điều chuẩn Gaussian, weight 0.09, knots 17, num_proj 512 (small) / 1024 (paper) |

Cả 5 module được huấn luyện *joint from scratch* trên pred_mse + 0.09 · SIGReg; target embedding *không* stop-gradient (xem §1.1). Optimizer gốc: AdamW(lr = 5e-5, weight_decay = 1e-3), bf16, gradient_clip 1.0, lịch LR 1% warmup + cosine. Dữ liệu PushT: bộ demo chuyên gia dạng HDF5 (`pusht_expert_train.h5`, pixel nén Blosc), frameskip 5 (ghép 5 action thô mỗi bước), chia train/val theo episode tỉ lệ 0.9; cửa sổ train dài 4 bước (history 3 + pred 1), cửa sổ đánh giá dài 8 bước (history 3 + rollout 5). `--data X` chọn X% cửa sổ train; đánh giá luôn dùng 15% cửa sổ val.

### 2.9. Tham số MCSDCA

Phần này chỉ liệt kê *tên tham số và ý nghĩa* của chúng trong thuật toán (không gắn với thực nghiệm cụ thể nào). Giá trị cài đặt thực tế của từng tham số cho mỗi lần chạy được trình bày tại đầu mục thực nghiệm tương ứng ở Phần III (§3.2.1 cho Thực nghiệm 1, §3.3.1 cho Thực nghiệm 2).

| Tham số (code) | Ký hiệu | Ý nghĩa |
|---|---|---|
| od_eta | $\eta$ | Bước học (step size) của Langevin overdamped (odLD) — điều khiển tốc độ mẫu $x'$ di chuyển theo hướng giảm năng lượng $U(x')$ ở mỗi bước lấy mẫu. |
| ud_delta | $\delta$ | Bước rời rạc của Langevin underdamped (udLD) — vai trò tương tự $\eta$ nhưng cho biến thể có quán tính (kèm biến vận tốc $v$), thường cho phép bước dài hơn mà chuỗi vẫn ổn định. |
| epsilon | $\varepsilon$ | "Nhiệt độ" làm mịn local-entropy; điều khiển cường độ nhiễu Gaussian trong bước Langevin ($\sqrt{2\eta\varepsilon}$). $\varepsilon$ lớn ⇒ chuỗi khám phá rộng quanh $x^k$; $\varepsilon$ nhỏ ⇒ chuỗi gần như chỉ là gradient descent ngắn, ít khám phá. |
| local_entropy_time | $t$ | Thời gian giả trong PDE Hamilton–Jacobi; kiểm soát mức phạt khoảng cách $\lVert x'-x^k\rVert$ trong hàm năng lượng $U$. $t$ lớn ⇒ ràng buộc yếu, mẫu $x'$ được tự do rời xa $x^k$ hơn (làm mịn mạnh hơn). |
| beta0 | $\beta_0$ | Tỉ trọng của trung bình Markov $y_k$ trong bước trộn tuyến tính DCA tại $k=0$ (nghiệm đóng $x_{k+1} = \alpha_k x^k + \beta_k y_k$, với $\beta_k = 1/(1+t\gamma_k)$). Đặt `beta0` tương đương đặt $\gamma_0 = (1/\beta_0-1)/t$ — cách tham số hóa trực quan hơn gamma khi cần dò lưới. |
| burn_in | $b$ | Số mẫu đầu của chuỗi Markov bị loại bỏ trước khi lấy trung bình $y_k$, vì các mẫu này chưa kịp hội tụ về phân phối cân bằng $P(\xi\mid x_k)$. |
| langevin_steps, langevin_steps_power | $\bar b$, $\lambda$ | Tham số lịch độ dài chuỗi Markov $n_k = \bar b + \lfloor(k+1)^\lambda\rfloor$ — chuỗi dài dần theo outer step $k$, giúp ước lượng $y_k$ chính xác hơn khi thuật toán tiến gần nghiệm. |
| gamma, gamma_power | $\gamma_0$, $\beta$ | Lịch hệ số proximal $\gamma_k = \gamma_0\cdot(k+1)^\beta$ trong bài toán con DCA. $\gamma_k$ lớn ⇒ bước cập nhật $x_{k+1}$ bị giữ gần $x^k$ hơn (thận trọng hơn); $\gamma_k$ nhỏ ⇒ bước cập nhật thiên về $y_k$ nhiều hơn. |
| max_grad_norm | — | Ngưỡng gradient clipping áp dụng trong mỗi bước Langevin, tránh bước cập nhật quá lớn khi $\nabla f$ đột biến. |

---

## III. Thực nghiệm

### 3.1. Thiết lập chung

- *Dữ liệu:* PushT (`pusht_expert_train.h5`), chia train/val theo episode tỉ lệ 0.9, seed 3072.
- *So sánh công bằng:* cùng trọng số khởi tạo (lưu một lần theo seed, nạp lại y hệt trước mỗi optimizer), cùng luồng mini-batch, đọc kết quả tại **cùng backprop_calls**. Một outer step MCSDCA ⇒ chain_length − 1 lượt `backward()`; vòng lặp dừng khi backprop_calls ≥ budget nên MCSDCA vượt nhẹ (8001 so với 8000; 40011 so với 40000).
- *Chỉ số* (hàm `evaluate` / `rollout_stats`): train_mse / val_mse = MSE dự đoán một bước, teacher-forced, chỉ phần prediction (không gồm SIGReg); rollout_mse_1/3/5 = MSE khi predictor tự hồi quy 1/3/5 bước trên chính output của nó; train_val_gap = val − train; latent_norm_drift = trung bình‖$\hat z$‖ − trung bình‖$z$‖ ở tầm nhìn 5 (âm lớn = latent *do predictor xuất ra* có norm nhỏ hơn embedding encoder); pred_latent_variance = phương sai của *latent rollout của predictor* theo (batch, thời gian) ở tầm nhìn 5 (≈ 0 = predictor xuất ra gần như hằng số). Lưu ý: pred_latent_variance đo đầu ra predictor, *không* đo phương sai của embedding encoder; target_latent_norm (norm của embedding encoder) cũng được ghi trong metrics.csv nhưng không hiển thị trong notebook — cần nó để nói về trạng thái encoder.
- *Xử lý BatchNorm khi chạy MCSDCA:* `train_mcsdca` lưu `model.buffers()` trước mỗi outer step, khôi phục sau khi chạy xong chuỗi Markov (bỏ thống kê BN của tham số bị nhiễu), rồi chạy một forward no_grad tại $x_{k+1}$ để cập nhật running-stats *một lần mỗi outer step*. AdamW cập nhật running-stats **mỗi backward()** ⇒ MCSDCA cập nhật BN ít hơn ~5–7 lần.
- Báo cáo cả *MSE* (đối chiếu trực tiếp loss huấn luyện) lẫn *RMSE = √MSE* (cùng đơn vị latent; theo `results_report.ipynb`, norm latent đích cỡ 2–14). Thứ hạng optimizer không đổi giữa hai thang.

### 3.2. Thực nghiệm 1: so sánh AdamW / MCSDCA-odLD / MCSDCA-udLD trên PushT (Set 1 tinh chỉnh vs Set 2 theo paper)

**Kết quả — Set 1 (10%, tinh chỉnh, 3 seed):**

> *Ghi chú đối chiếu: tên cột dưới đây bị mất trong file gốc và đã được khôi phục bằng cách đối chiếu số liệu với `01_EXPERIMENT1/results_report.ipynb` / phần văn bản diễn giải ngay sau bảng — không tự bịa số liệu.*

| optimizer | val MSE | train MSE | train–val gap | rollout MSE@1 | rollout MSE@3 | rollout MSE@5 | latent_norm_drift | pred_latent_var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *AdamW* | *0.0301 ± 0.0020* | 0.0268 | +0.0033 | *0.0287* | *0.0694* | *0.1210* | −0.55 | *~0.80* |
| MCSDCA-odLD | 0.6022 ± 0.0104 | 0.5845 | +0.0178 | 0.5955 | 0.6629 | 0.6797 | −8.3 | ~0.026 |
| MCSDCA-udLD | 0.6816 ± 0.0627 | 0.6636 | +0.0179 | 0.6878 | 0.7380 | 0.7361 | −7.9 | ~0.025 |

(Thang RMSE: val AdamW ≈ 0.173, odLD ≈ 0.776, udLD ≈ 0.826.)

**Kết quả theo tỉ lệ dữ liệu** (val MSE / rollout MSE@5, 3 seed mỗi mức):

| dữ liệu | AdamW | MCSDCA-odLD | MCSDCA-udLD |
|---:|---:|---:|---:|
| 1% | 0.0364 / 0.1317 | 0.6180 / 0.6936 | 0.7141 / 0.7607 |
| 10% | 0.0301 / 0.1210 | 0.6022 / 0.6797 | 0.6816 / 0.7361 |

Khoảng cách AdamW ↔ MCSDCA *không* thu hẹp khi giảm dữ liệu từ 10% xuống 1% — trái với kỳ vọng "MCSDCA lợi hơn khi dữ liệu ít".

**Hình hội tụ:** chạy `src/MCSDCA_Predictor_Optimizer_Walkthrough.ipynb` — ô cell05 (one-step MSE train/val + rollout MSE@5 + latent_norm_drift theo backprop_calls) và cell06 (ba chỉ số theo tỉ lệ dữ liệu). Diễn giải từ vệt sampler (3 eval cuối mỗi seed, backprop 6405 → 8001): sampler_loss của cả odLD và udLD dao động quanh 1.0–1.2 *không giảm rõ*; train_mse / val_mse bật lên bật xuống trong khoảng 0.5–0.8 ⇒ chưa có xu thế hội tụ xuống. markov_chain_length ≈ 7 (trần 8), retained_samples = 5, tức ~6 lượt `backward()` mỗi outer step ⇒ ~1 500 outer step trong 8 000 backprop (khớp outer_step 1504 ở cuối vệt).

**Kết quả — Set 2 (4%, theo paper, 1 seed):**

> *Ghi chú đối chiếu: tên cột dưới đây cũng được khôi phục theo cùng cách nêu trên (đối chiếu với bảng plateau và `val_MSE (ref)` ngay dưới nó, nơi 0.116² ≈ 0.01346 v.v. khớp khít).*

| optimizer | val RMSE | train RMSE | train–val gap | rollout RMSE@1 | rollout RMSE@3 | rollout RMSE@5 | val MSE (ref) |
|---|---:|---:|---:|---:|---:|---:|---:|
| *AdamW* | *0.1160* | 0.0933 | +4.8e-3 | *0.1198* | *0.1674* | *0.2082* | *0.01346* |
| MCSDCA-udLD | 0.5557 | 0.5942 | −4.4e-2 | 0.5709 | 0.5835 | 0.5920 | 0.30875 |
| MCSDCA-odLD | 0.7675 | 0.7619 | +8.6e-3 | 0.7741 | 0.8317 | 0.8499 | 0.58904 |

**Chẩn đoán plateau** (xét 30% cuối ngân sách; ngưỡng tương đối 5%):

| optimizer | split | RMSE cuối | rel_drop đoạn cuối | kết luận |
|---|---|---:|---:|---|
| AdamW | train / val | 0.093 / 0.116 | +11.4% / +8.0% | *chưa hội tụ (còn giảm)* |
| MCSDCA-odLD | train / val | 0.762 / 0.767 | +16.8% / +16.9% | *chưa hội tụ (còn giảm)* |
| MCSDCA-udLD | train / val | 0.594 / 0.556 | −1.2% / +0.4% | *đã đi ngang* |

**Hình hội tụ:** `outputs/hus/experiment1/data4/report_convergence.png` (one-step RMSE train nét đứt / val nét liền theo backprop_calls, trục log) và `report_convergence_rollout5.png` — sinh bởi `01_EXPERIMENT1/results_report.ipynb` mục 3. Đọc: AdamW giảm đều và *vẫn đang dốc* ở cuối 40 000; odLD giảm chậm, còn dốc, ở mức sai số cao gấp ~8 lần; udLD *phẳng sớm* ở RMSE ≈ 0.55 (đi ngang nhưng ở mức xấu).

**So sánh nhanh Set 1 vs Set 2:**

> *Ghi chú đối chiếu: hàng đầu ("rollout MSE@5...") bị mất tên cột trong file gốc; đã khôi phục bằng cách đối chiếu số 0.68/0.121→×5.6 và 0.35/0.043→×8 với hai bảng ngay phía trên.*

| Chỉ số | Thực nghiệm 1 — Set 1 (10%, tinh chỉnh) | Thực nghiệm 1 — Set 2 (4%, theo paper) |
|---|---|---|
| rollout MSE@5 (MCSDCA tốt nhất / AdamW) | 0.68 / 0.121 → *×5.6* | 0.35 / 0.043 → *×8* (MSE) |
| train–val gap | AdamW +0.003 · MCSDCA +0.018 | AdamW +0.005 · udLD *−0.044* · odLD +0.009 |
| biến thể MCSDCA tốt hơn | odLD | udLD |
| latent (đầu ra predictor) | MCSDCA pred_var ~0.025, drift ~−8 | (không hiển thị trong bảng notebook; mức sai số nhất quán với sụp đổ đầu ra predictor) |

Kết luận nhất quán giữa hai bộ: *AdamW cho sai số dự đoán và rollout thấp hơn hẳn ở mọi tầm nhìn và mọi mức dữ liệu.* Việc odLD thắng ở Set 1 còn udLD thắng ở Set 2 cho thấy thứ hạng nội bộ MCSDCA nhạy với cấu hình/ngân sách, chưa ổn định.

#### 3.2.5. Nhận xét

1. *Đầu ra predictor sụp về gần hằng số — đây là vấn đề nổi bật nhất.* Ở Set 1, mọi lần chạy MCSDCA có pred_latent_variance ≈ 0.02–0.03 (AdamW ≈ 0.8) và latent_norm_drift ≈ −7…−9, trong khi rollout_mse vẫn lớn. Nghĩa là predictor khi rollout xuất ra một vector gần như cố định, norm co lại, và không khớp target — không phải trường hợp "mọi thứ cùng sụp về một điểm" (khi đó MSE ≈ 0).
   - *Điều bằng chứng hiện có chứng minh được:* sụp đổ ở phía đầu ra predictor / rollout.
   - *Điều chưa kết luận được:* encoder có sụp đổ theo không. latent_norm_drift chỉ so norm giữa pred và target, không phân biệt "encoder khỏe, norm ~10" với "encoder sụp về hằng số norm ~10". Metric phân biệt được (phương sai embedding encoder) không được log; cần xem target_latent_norm theo backprop_calls và giá trị sigreg_loss trong `seed*/metrics.csv` (chưa có trên máy phân tích).
   - *Không phải* do SIGReg thiếu gradient (nó nằm trong training_objective, có mặt ở mọi bước Langevin) và *không phải* do BN bị nhiễu (buffer được lưu/khôi phục — §3.1). Nghi vấn hợp lý hơn: (a) ε cực nhỏ (3e-9 Set 1, 1e-8 Set 2) ⇒ gần như không có khám phá, chuỗi Markov chỉ là vài bước gradient ngắn quanh $x^k$; (b) bước ngoài DCA chỉ trộn nửa đường về $y_k$ (beta0 = 0.5) và chạy rất ít lần ⇒ áp lực của SIGReg trên mỗi đơn vị compute yếu hơn AdamW nhiều; (c) BN running-stats chỉ cập nhật ~1500 lần thay vì 8000.
   - Dù nguyên nhân là gì, sai số tuyệt đối cao gần như chắc chắn là hệ quả của sụp đổ này, không phải chuyện "cần thêm ngân sách".
2. *Gap train/val nhỏ của MCSDCA không phải tín hiệu tốt.* Gap nhỏ (thậm chí âm với udLD) nhưng nằm ở mức sai số cao gấp 5–20 lần ⇒ chỉ nói lên "train và val cùng kém", không phải "regularization chống overfit tốt hơn".
3. *udLD đi ngang sớm ở sai số cao* (Set 2): dấu hiệu underfitting / bước sampling chưa đủ mạnh, hoặc ε–beta0 (chọn ở ngân sách 8 000) không phù hợp cho ngân sách 40 000.
4. *odLD còn dốc ở 40 000 backprop* nhưng xuất phát quá cao ⇒ chưa thể kết luận nếu chưa chạy ngân sách lớn hơn và sửa sụp đổ đầu ra predictor trước.
5. *Chi phí tính toán:* trong 8 000 backprop, MCSDCA chỉ đi được ~1 500 outer step (chuỗi Markov ≈ 7, ~6 backward/outer step) ⇒ ít bước cập nhật "thật" hơn AdamW ~6 lần. Khi khớp theo backprop_calls, MCSDCA thiệt nếu mỗi outer step tiến bộ ít.
6. *Giả thuyết "dữ liệu ít thì MCSDCA lợi hơn" chưa được xác nhận:* khoảng cách với AdamW không thu hẹp giữa 10% và 1% (Set 1), cũng không ở 4% (Set 2).

---

### 3.3. Thực nghiệm 2: điều chỉnh số bước Langevin và số bước cập nhật

Xuất phát từ chẩn đoán ở §3.2.5 mục 1 (nghi vấn ε quá nhỏ, chuỗi Markov quá ngắn khiến áp lực SIGReg/BN cập nhật yếu hơn AdamW), Thực nghiệm 2 giữ nguyên *Set 1 (không tinh chỉnh)* làm mốc đối chứng và thay đổi cấu hình MCSDCA ở *Set 2* để tăng chất lượng ước lượng $y_k$ mỗi outer step.

#### 3.3.1. Cấu hình

| | Set 1 | Set 2 — tăng số bước Langevin, số bước cập nhật ×3 |
|---|---|---|
| Dữ liệu PushT | 4% | 4% |
| Seed | 3072 | 1 |
| Ngân sách | 22 320 backprop (AdamW) / 22 329 (MCSDCA) — batch 128, early stop tắt, ≈ 40 epoch tương đương ở 4% dữ liệu | ~40 epoch — giá trị số cụ thể cần bổ sung khi hoàn tất phân tích |
| Lệnh chạy | `uv run src/run_experiment1.py --only-fraction 0.04 --set train.early_stop.enabled=false --set 'seed=[3072]' --set mcsdca.langevin_steps=5 --set mcsdca.burn_in=2 --set mcsdca.max_langevin_steps=8` | tăng thêm ×3 so với Set 1 (giá trị cụ thể cần bổ sung) |
| Thay đổi MCSDCA so với `MCSDCAConfig.paper()` (Thực nghiệm 1, Set 2) | `langevin_steps`: lịch mặc định $n_k=20+\lfloor(k+1)^{0.1}\rfloor$ → cố định **5**; `burn_in`: 10 → **2**; thêm trần `max_langevin_steps` = **8** (paper không giới hạn trần) | (đang cập nhật) |

*Ghi chú: mô tả "Set 1 không tinh chỉnh, giống Set 2 của Thực nghiệm 1" trong bản nháp trước đã lỗi thời — Set 1 thực tế đã đổi `langevin_steps`/`burn_in`/`max_langevin_steps` so với paper. Ngân sách cũng thấp hơn nhiều so với Thực nghiệm 1 Set 2 (22 320 so với 40 000), nên **không so trực tiếp 1-1** giữa hai thực nghiệm — Set 1 của Thực nghiệm 2 là một baseline riêng, mới.*

#### 3.3.2. Kết quả — Set 1

RESULTS_DIR = `outputs/data4/20260911_014300_411__seed3072` (đọc bằng `01_EXPERIMENT1/results_report.ipynb`).

**Bảng kết quả cuối:**

| optimizer | val RMSE | train RMSE | train–val gap | rollout RMSE@5 | val MSE (ref) |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.14733 | 0.12706 | +5.56e-3 | 0.26179 | 0.02170 |
| MCSDCA-udLD | 0.72393 | 0.75453 | −4.52e-2 | 0.77288 | 0.52407 |
| MCSDCA-odLD | 1.18057 | 1.14751 | +7.70e-2 | 1.36712 | 1.39374 |

So với AdamW: udLD tệ hơn ×4.9 (RMSE) / ×24 (MSE); odLD tệ hơn ×8.0 (RMSE) / ×64 (MSE). udLD vẫn là biến thể tốt hơn giữa hai, giống Thực nghiệm 1 Set 2.

**Chẩn đoán plateau** (30% cuối ngân sách, ngưỡng tương đối 5%):

| optimizer | split | RMSE cuối | rel_drop_tail | rel_span_tail | kết luận |
|---|---|---:|---:|---:|---|
| AdamW | train / val | 0.127 / 0.147 | +13.7% / +7.6% | 15.2% / 8.0% | *chưa hội tụ (còn giảm)* |
| MCSDCA-odLD | train / val | 1.148 / 1.181 | −0.1% / −0.3% | 48.0% / 50.1% | *còn dao động mạnh* |
| MCSDCA-udLD | train / val | 0.755 / 0.724 | −7.2% / −12.1% | 18.7% / 26.2% | *còn dao động* |

odLD gần như không đổi trung bình trong 30% cuối, nhưng dao động biên độ tới ~50% quanh mức đó ⇒ chưa ổn định, không phải "đi ngang" theo nghĩa hội tụ.

**Chẩn đoán sụp đổ biểu diễn** (giá trị cuối cùng, trung bình theo optimizer — cột `col_*` trên val, xem định nghĩa ở §3.1):

| optimizer | col_enc_emb_var_mean | col_enc_dead_dim_frac | col_pred_target_var_ratio | col_pred_target_norm_ratio | col_enc_emb_norm_mean |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.928 | 0.0 | 0.999 | 0.999 | 13.83 |
| MCSDCA-odLD | 0.831 | 0.0 | 0.302 | 0.505 | 25.41 |
| MCSDCA-udLD | 0.388 | 0.0 | 0.242 | 0.499 | 9.57 |

- **Encoder không sụp:** `col_enc_dead_dim_frac = 0` ở cả ba optimizer, `col_enc_emb_var_mean` vẫn ở mức hợp lý (0.831 cho odLD, 0.388 cho udLD, so với 0.928 của AdamW), và `col_enc_emb_norm_mean` của odLD (25.4) thậm chí *lớn hơn* AdamW (13.8) còn udLD (9.6) chỉ hơi thấp hơn — không có dấu hiệu encoder co về hằng số. Đây là câu trả lời cho câu hỏi còn bỏ ngỏ ở §3.2.5 mục 1 ("encoder có sụp theo không") cho cấu hình Set 1 này: *không*.
- **Predictor cải thiện rõ nhưng chưa hết sụp đổ:** `col_pred_target_var_ratio` (0.24–0.30) và `col_pred_target_norm_ratio` (~0.50) cao hơn nhiều so với mức gần như sụp hoàn toàn (phương sai tuyệt đối ~0.02–0.03, drift ~−8) ghi nhận ở Thực nghiệm 1 — xác nhận quan sát ở phần Tóm tắt "không còn sụp đổ biểu diễn" đối với Set 1. Tuy nhiên tỉ lệ vẫn chỉ bằng ~1/4–1/2 so với AdamW (≈1.0), nên chính xác hơn nên gọi là *giảm sụp đổ đáng kể*, chưa phải *hết sụp đổ hoàn toàn* — predictor vẫn under-predict biên độ so với target.
- **Sai số tuyệt đối vẫn cao:** dù chỉ số sụp đổ cải thiện mạnh, val RMSE/MSE của cả hai biến thể MCSDCA vẫn cao hơn AdamW 5–8 lần (RMSE) / 24–64 lần (MSE) ở ngân sách này — nghĩa là triệu chứng "sụp đổ biểu diễn" đã giảm nhưng vấn đề "hội tụ chậm / kém chính xác hơn AdamW" vẫn còn nguyên.

**Kết luận Set 1:** so với cấu hình paper gốc (Thực nghiệm 1, Set 2), việc giảm `langevin_steps` xuống 5 (cố định), `burn_in` xuống 2 và giới hạn trần 8 giúp encoder/predictor không còn co gần về hằng số như trước (tỉ lệ phương sai/norm của predictor so với target tăng từ ~2–3%/nhỏ lên ~25–50%), nhưng MCSDCA vẫn chưa hội tụ ổn định (đặc biệt odLD dao động ~50% biên độ ở đoạn cuối) và sai số tuyệt đối vẫn thua AdamW nhiều lần. Set 2 (tăng thêm ×3 số bước Langevin/cập nhật so với Set 1) sẽ cho biết liệu tăng tiếp có giải quyết được phần dao động/sai số còn lại hay không.

---

## IV. Kết luận và hướng tiếp theo

**Kết luận tạm thời.** MCSDCA gắn được vào pipeline huấn luyện LeWM (cập nhật cả 5 module), chạy ổn định về số học (không phân kỳ ở mọi seed), và bước cập nhật DCA dạng đóng khiến chi phí mỗi outer step nhẹ. Tuy nhiên, trong cả hai bộ thực nghiệm hiện có, *MCSDCA chưa vượt AdamW ở bất kỳ chỉ số world model nào* (one-step MSE, rollout MSE mọi tầm nhìn, mọi mức dữ liệu), và đầu ra rollout của predictor sụp về gần hằng số (trạng thái encoder chưa kiểm chứng — §3.2.5). Chưa có bằng chứng cho luận điểm "local-entropy regularization ⇒ vùng phẳng ⇒ rollout dài ổn định hơn".

**Việc cần làm (thứ tự ưu tiên).**

1. *Trước hết: bổ sung log để biết sụp đổ nằm ở đâu* — ghi target_latent_norm và sigreg_loss theo backprop_calls, thêm phương sai embedding encoder. Nếu sigreg_loss tăng vọt / phương sai encoder → 0 thì encoder cũng sụp; nếu không, chỉ predictor hỏng. Cách khắc phục khác nhau tùy kết quả này.
2. *Sửa sụp đổ đầu ra predictor* (SIGReg đã có gradient, BN buffer đã được quản lý — không phải các nguyên nhân này). Thử theo thứ tự: tăng ε lên chế độ Langevin thật (ε/eta ∈ {1e-2, 1e0}) để chuỗi Markov có khám phá; tăng λ (trọng số SIGReg) riêng cho nhánh MCSDCA; tăng beta0 để bước ngoài đi gần hết về $y_k$; thử detach target để tách phần "học biểu diễn" khỏi phần "học dynamics". Không có ý nghĩa khi so tiếp nếu predictor vẫn sụp.
3. *Tinh chỉnh MCSDCA ngay tại ngân sách đích* (40 000) thay vì tái dùng winner của ngân sách 8 000 — ε, beta0, eta/delta nhiều khả năng phải khác ở chế độ dài.
4. *Chạy nhiều seed cho Set 2* để có sai số chuẩn, tránh kết luận trên 1 seed.
5. *Đánh giá planning PushT* (`src/evaluate_planning.py`): PushT success / score, CEM cost, imagined-vs-real rollout gap. Đây mới là mục tiêu cuối — world model tồn tại để phục vụ planning; hiện chưa chạy phần này.
6. *Data-scaling đầy đủ* 1% / 10% / 50% / 100% với winner cố định, để kiểm định giả thuyết "dữ liệu ít" một cách dứt khoát.
7. *Khảo sát độ nhạy* local_entropy_time $t$, gamma, langevin_steps, burn_in — các tham số hiện đang để theo lịch paper, chưa từng dò trên bài toán LeWM.

---

## Phụ lục

**Tái lập.**

```bash
# Set 1 — có tinh chỉnh, 10% dữ liệu
python src/run_experiment1.py --data 10                 # tune + eval, eval-budget 8000, 3 seed
python src/run_experiment1.py --data 1 --reuse-winners outputs/experiment1/data10/winners.json

# Set 2 — tái lập theo paper, 4% dữ liệu (bỏ tune; AdamW = config LeWM, MCSDCA = MCSDCAConfig.paper())
# --profile paper mặc định lấy budget = --paper-epochs (100) * steps/epoch;
# bộ đã chạy dùng ~40 epoch -> đặt --eval-budget 40000 (hoặc --paper-epochs 40).
python src/run_experiment1.py --data 4 --profile paper --eval-budget 40000
```

**Kết quả sinh trong `outputs/.../experiment1/data<X>/`:** `winners.json` (siêu tham số winner), `comparison.csv` / `comparison_agg.csv` (mỗi dòng optimizer × seed / gộp seed), `seed<n>/metrics.csv` (vệt huấn luyện), `sweep_*/sweep_summary.csv` (mọi điểm lưới).

**Notebook phân tích.**
- `01_EXPERIMENT1/results_report.ipynb` — bảng kết quả cuối, hình hội tụ, chẩn đoán plateau (Set 2, RESULTS_DIR = `outputs/hus/experiment1/data4`).
- `src/MCSDCA_Predictor_Optimizer_Walkthrough.ipynb` — cấu hình chi tiết, bảng per-seed, đường huấn luyện, đường theo tỉ lệ dữ liệu, vệt sampler MCSDCA (Set 1).

**Tài liệu liên quan:** `01_EXPERIMENT1/MCSDCA_Predictor_Optimizer_Report.md` (lý thuyết + placeholder chi tiết), `01_EXPERIMENT1/EXPERIMENT_PLAN.md`.
