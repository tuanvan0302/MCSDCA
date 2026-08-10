# Thí nghiệm 1: Dùng MCSDCA như optimizer cho predictor của LeWM

## I. Cơ sở lý thuyết MCSDCA

### 1. Từ bài toán tối ưu đến quy hoạch DC

Trong một bài toán tối ưu, ta có một biến quyết định $x$ và một hàm mục tiêu $F(x)$. Mục tiêu là tìm giá trị $x$ sao cho $F(x)$ nhỏ nhất:

$$
\min_x F(x).
$$

Nếu $F(x)$ là hàm lồi và trơn, bài toán thường dễ xử lý hơn. Nhưng trong học máy hiện đại, đặc biệt là deep learning, $F(x)$ thường không lồi, có thể không trơn, có nhiều cực tiểu cục bộ và bề mặt loss rất gồ ghề. Đây là lý do cần các phương pháp tối ưu mạnh hơn, hoặc ít nhất là các phương pháp có thể làm việc tốt hơn trong bối cảnh không lồi và ngẫu nhiên.

Trong đó, quy hoạch DC (Difference of Convex programming) là một trong các hướng tiếp cận được sử dụng để xử lý bài toán không lồi. Ý tưởng là phân tích một hàm không lồi thành hiệu của hai hàm lồi. Nhiều bài toán trong học máy, bao gồm một số loss của mạng neural, có thể được biểu diễn hoặc xấp xỉ dưới dạng DC. Một hàm DC là hàm có thể viết thành:

$$
F(x)=G(x)-H(x),
$$

trong đó $G$ và $H$ đều là hàm lồi.

Sau khi có phân tách DC, thuật toán kinh điển DCA - Difference-of-Convex Algorithm - thường được áp dụng để giải bài toán:

$$
\min_x \; F(x)=G(x)-H(x).
$$

DCA lặp qua các bước sau. Tại vòng lặp $k$, ta lấy một subgradient của $H$ tại $x_k$:

$$
y_k \in \partial H(x_k).
$$

Sau đó tuyến tính hóa $H$ quanh $x_k$:

$$
H(x) \approx H(x_k)+\langle x-x_k,y_k\rangle.
$$

Vì $H(x_k)$ là hằng số theo $x$, bài toán con trở thành:

$$
x_{k+1}\in \arg\min_x \left\{G(x)-\langle x,y_k\rangle\right\}.
$$

Ý tưởng cốt lõi của DCA là: thay vì giải trực tiếp một bài toán không lồi khó, ta giải một chuỗi bài toán lồi con. Mỗi vòng lặp dùng thông tin tại nghiệm hiện tại để tạo một xấp xỉ dễ xử lý hơn. Cần lưu ý rằng DCA thường hướng tới các điểm tới hạn (critical point), chứ không đảm bảo tìm được nghiệm tối ưu toàn cục trong bài toán không lồi.

### 2. Vì sao cần MCSDCA?

Sau khi có nền tảng cơ bản về DC/DCA, ta quay trở lại với thuật toán chính trong nghiên cứu này là **MCSDCA** - **Markov Chain Stochastic Difference-of-Convex Algorithm**. Thuật toán này được đề xuất để giải một lớp bài toán tối ưu khó hơn DCA thông thường: bài toán **quy hoạch DC ngẫu nhiên có bất định nội sinh**, trong đó không thể lấy được mẫu độc lập cùng phân phối, tức không có mẫu i.i.d., mà chỉ có thể sinh mẫu thông qua chuỗi Markov.

Xuất phát điểm vẫn là bài toán dạng DC:

$$
\min_x F(x)=G(x)-H(x).
$$

Điểm đặc biệt trong bài toán của paper là subgradient của $H$ không được tính trực tiếp, mà có dạng kỳ vọng:

$$
\mathbb{E}_{P(\xi\mid x)}[v(x,\xi)] \in \partial H(x).
$$

Ý nghĩa của công thức này:

- $x$ là biến quyết định cần tối ưu.
- $\xi$ là biến ngẫu nhiên.
- $P(\xi\mid x)$ là phân phối của $\xi$ khi biết $x$.
- $v(x,\xi)$ là vector được tính từ $x$ và mẫu $\xi$.
- Trung bình của $v(x,\xi)$ theo phân phối $P(\xi\mid x)$ cho ta một subgradient của $H(x)$.

Điểm khó nằm ở chỗ phân phối $P(\xi\mid x)$ phụ thuộc vào chính biến quyết định $x$. Đây là **bất định nội sinh** (endogenous uncertainty). Khi $x$ thay đổi, phân phối cần lấy mẫu cũng thay đổi theo. Vì vậy, ta không thể giả sử có sẵn một nguồn mẫu i.i.d. cố định như trong nhiều bài toán stochastic optimization cổ điển.

Luồng ý tưởng tổng quát có thể viết ngắn gọn như sau:

```text
Decision variable x thay đổi
-> Distribution P(xi | x) thay đổi
-> Không lấy được mẫu i.i.d. trực tiếp
-> Dùng Markov chain để sinh mẫu xi
-> Ước lượng subgradient của H(x)
-> Tuyến tính hóa H bằng DCA
-> Giải convex subproblem
-> Cập nhật x
-> Lặp lại
```

Trong paper, tác giả Hoang Phuc Hau Luu và các cộng sự giải quyết vấn đề này bằng cách xây dựng một chuỗi Markov có phân phối cân bằng là $P(\xi\mid x_k)$ tại mỗi vòng lặp $k$. Các mẫu đầu của chuỗi có thể bị loại bỏ qua giai đoạn burn-in, sau đó các mẫu còn lại được dùng để ước lượng subgradient của $H$.

### 3. Khung thuật toán MCSDCA tổng quát

Tại vòng lặp ngoài thứ $k$, ta chạy một chuỗi Markov:

$$
\xi_0^k \to \xi_1^k \to \cdots \to \xi_{n_k-1}^k,
$$

có phân phối cân bằng là:

$$
P(\xi\mid x_k).
$$

Sau khi bỏ đi $b$ mẫu burn-in đầu tiên, ta tính ước lượng:

$$
y_k=\frac{1}{n_k-b}\sum_{i=b}^{n_k-1}v(x_k,\xi_i^k).
$$

Giá trị $y_k$ này được dùng thay cho subgradient thật của $H(x_k)$. Cần diễn đạt chính xác rằng đây là một **ước lượng có thể bị chệch** (biased estimator), vì các mẫu trong chuỗi Markov phụ thuộc lẫn nhau và phân phối tại từng bước chưa chắc đã đúng chính xác phân phối mục tiêu. Điểm mạnh của paper là vẫn chứng minh được hội tụ trong bối cảnh ước lượng kiểu Markov như vậy.

Sau đó MCSDCA giải bài toán lồi con:

$$
\min_x \left\{G(x)-\langle x,y_k\rangle + \frac{\gamma_k}{2}\Vert x-x_k\Vert_A^2\right\},
$$

trong đó:

- $\gamma_k>0$ là hệ số proximal ở vòng lặp $k$.
- $A\succeq I$ là ma trận đối xứng xác định dương.
- $\Vert z\Vert_A^2=z^\top A z$.
- Bài toán con có thể được giải chính xác hoặc tới một mức sai số ngẫu nhiên $\epsilon_k$.

Có thể xem đây là DCA + stochastic approximation + Markov sampling. DCA cung cấp cấu trúc tuyến tính hóa $H$, còn Markov chain cung cấp cách lấy mẫu khi không thể lấy i.i.d.

### 4. PDE regularization và local entropy

Bài báo không chỉ dừng ở thuật toán tổng quát. Phần quan trọng tiếp theo là ứng dụng MCSDCA vào deep learning thông qua **PDE regularization**. Ý tưởng chính là thay vì cực tiểu hóa trực tiếp loss gốc $f(x)$ vốn rất gồ ghề, ta tối ưu một phiên bản được làm mượt hơn của nó.

Ở đây $x$ có thể hiểu là toàn bộ trọng số của mạng neural, còn $f(x)$ là training loss. Paper xét phương trình Hamilton-Jacobi:

$$
u_t + \frac{1}{2}\lVert \nabla u\rVert^2=0,
\quad (x,t)\in \mathbb{R}^n\times(0,+\infty),
$$

với điều kiện ban đầu:

$$
u(x,0)=f(x).
$$

Paper cũng nhắc đến phiên bản có độ nhớt:

$$
u_t + \frac{1}{2}\lVert \nabla u\rVert^2=\frac{\varepsilon}{2}\Delta u,
\quad (x,t)\in \mathbb{R}^n\times(0,+\infty),
$$

trong đó:

- $u_t$ là đạo hàm của $u$ theo tham số tiến hóa $t$.
- $\nabla u$ là gradient theo biến không gian $x$.
- $\Delta u$ là toán tử Laplace.
- $\varepsilon>0$ kiểm soát mức độ làm trơn.

Hạng $\frac{\varepsilon}{2}\Delta u$ có vai trò như độ nhớt: nó làm giảm các dao động quá sắc của bề mặt loss. Trực giác của phần này là: thay vì tối ưu loss ở thời điểm ban đầu $t=0$, ta để hệ PDE tiến hóa đến thời điểm $t>0$, rồi tối ưu hàm đã được làm mượt $u(x,t)$.

Paper tiếp tục xét dạng Hamilton-Jacobi tổng quát hơn:

$$
u_t+\frac{1}{2}\langle \nabla u,\mathcal{H}^{-1}\nabla u\rangle=0,
$$

với điều kiện ban đầu $u(x,0)=f(x)$, trong đó $\mathcal{H}$ là một ma trận đối xứng xác định dương. Theo công thức Hopf-Lax, nghiệm viscosity của phương trình này có dạng inf-convolution:

$$
u(x,t)=\inf_{x'}\left\{f(x')+\frac{1}{2t}\langle x'-x,\mathcal{H}(x'-x)\rangle\right\}.
$$

Công thức này có ý nghĩa rất trực quan: ta chọn một điểm $x'$ có loss $f(x')$ thấp, nhưng nếu $x'$ quá xa $x$ thì bị phạt bởi hạng khoảng cách. Vì vậy nghiệm tạo ra một phiên bản làm mượt của loss gốc: giảm các đỉnh sắc, mở rộng các thung lũng cực tiểu, và ưu tiên các vùng loss thấp rộng hơn.

Để tránh tính không trơn của toán tử $\inf$, paper dùng một xấp xỉ log-sum-exp, thường được gọi là local entropy. Nếu đặt:

$$
\mathcal{F}(x')=f(x')+\frac{1}{2t}\langle x'-x,\mathcal{H}(x'-x)\rangle,
$$

thì theo tính chất entropy penalization:

$$
-\varepsilon\log\int_{\mathbb{R}^n}\exp\left(-\frac{\mathcal{F}(v)}{\varepsilon}\right)dv
\xrightarrow[\varepsilon\to 0^+]{}
\inf_v \mathcal{F}(v).
$$

Áp dụng vào nghiệm Hopf-Lax, ta có local entropy:

$$
\tilde u(x,t) := -\varepsilon\log\int_{\mathbb{R}^n}
\exp\left(
-\frac{1}{\varepsilon}
\left[f(x')+\frac{1}{2t}\langle x'-x,\mathcal{H}(x'-x)\rangle\right]
\right)dx'.
$$

Khi $\varepsilon\to 0^+$, $\tilde u(x,t)$ tiến về nghiệm inf-convolution $u(x,t)$. Đây là điểm nối giữa PDE regularization và local entropy.

### 5. Phân tách DC của local entropy

Phần quan trọng nhất để áp dụng MCSDCA là paper chứng minh local entropy có thể viết dưới dạng DC:

$$
\tilde u(x,t)=G(x)-H(x).
$$

Khai triển hạng khoảng cách trong local entropy:

$$
\langle x'-x,\mathcal{H}(x'-x)\rangle
= {x'}^\top\mathcal{H}x' -2x^\top\mathcal{H}x' + x^\top\mathcal{H}x.
$$

Từ đó ta tách được:

$$
G(x)=\frac{1}{2t}x^\top\mathcal{H}x,
$$

và:

$$
H(x)=\varepsilon\log
\int_{\mathbb{R}^n}
\exp\left(
-\frac{1}{\varepsilon}
\left[
f(x')+\frac{1}{2t}{x'}^\top\mathcal{H}x'
-\frac{1}{t}x^\top\mathcal{H}x'
\right]
\right)dx'.
$$

Nhìn vào hai hàm $G(x)$ và $H(x)$, ta có thể nhận định nhanh:

- $G(x)$ là một hàm bậc hai, lồi, trơn và dễ tính toán vì $\mathcal{H}$ xác định dương.
- $H(x)$ là một hàm log-integral phức tạp, nhưng vẫn lồi theo $x$ nhờ cấu trúc log-sum-exp của các hàm affine theo $x$.

Trong bài báo, tác giả chứng minh được gradient của $H$ có dạng:

$$
\nabla H(x)=\frac{1}{t}\mathbb{E}_{p(x'\mid x)}[\mathcal{H}x'],
$$

với phân phối mục tiêu:

$$
p(x'\mid x)\propto
\exp\left(
-\frac{1}{\varepsilon}
\left[
f(x')+\frac{1}{2t}{x'}^\top\mathcal{H}x'
-\frac{1}{t}x^\top\mathcal{H}x'
\right]
\right).
$$

Điểm nghẽn ở đây là ta phải tính expectation theo phân phối $p(x'\mid x)$. Trong deep learning, không gian tham số $x$ có thể rất lớn, và phân phối đích $p(x'\mid x)$ phụ thuộc vào loss của mạng neural. Việc tính chính xác tích phân hoặc lấy mẫu i.i.d. từ phân phối này gần như bất khả thi. Đây chính là lý do Langevin dynamics được đưa vào.

Nếu lấy trường hợp đơn giản $\mathcal{H}=I$, ta có:

$$
G(x)=\frac{1}{2t}\lVert x\rVert^2,
$$

$$
\nabla H(x)=\frac{1}{t}\mathbb{E}_{p(x'\mid x)}[x'],
$$

và:

$$
p(x'\mid x)\propto
\exp\left(
-\frac{1}{\varepsilon}
\left[
f(x')+\frac{1}{2t}\lVert x'\rVert^2
-\frac{1}{t}\langle x,x'\rangle
\right]
\right).
$$

Tương đương, bỏ đi hằng số không phụ thuộc vào $x'$, phân phối này có thể viết dưới dạng:

$$
p(x'\mid x)\propto
\exp\left(
-\frac{1}{\varepsilon}
\left[
f(x')+\frac{1}{2t}\lVert x'-x\rVert^2
\right]
\right).
$$

Cách viết thứ hai trực quan hơn: ta lấy mẫu quanh $x$, ưu tiên các điểm $x'$ có loss thấp nhưng không quá xa $x$.

### 6. Langevin dynamics để lấy mẫu từ phân phối mục tiêu

Câu hỏi đặt ra là: làm sao lấy mẫu từ một phân phối mà ta không biết hằng số chuẩn hóa?

Giả sử ta cần lấy mẫu từ một phân phối đích:

$$
\pi(x)=\frac{\exp(-\beta U(x))}{Z},
$$

trong đó:

- $U(x)$ là hàm năng lượng (energy function).
- $\beta$ là inverse temperature.
- $Z=\int \exp(-\beta U(x))dx$ là hằng số chuẩn hóa.

Trong deep learning, $Z$ gần như không thể tính được vì không gian tham số quá lớn. Nhưng Langevin dynamics chỉ cần gradient của $U$, không cần biết $Z$. Với overdamped Langevin dynamics, phương trình liên tục là:

$$
dX_t=-\nabla U(X_t)dt+\sqrt{2\beta^{-1}}dB_t,
$$

trong đó $B_t$ là Brownian motion. Khi các điều kiện kỹ thuật phù hợp được thỏa mãn, quá trình này có phân phối bất biến là $\pi(x)\propto\exp(-\beta U(x))$.

Khi đưa vào máy tính, ta phải rời rạc hóa. Một bước Euler-Maruyama có dạng:

$$
X_{i+1}=X_i-\eta\nabla U(X_i)+\sqrt{2\eta\beta^{-1}}\,\mathcal{N}(0,I).
$$

Nếu chọn $\beta=1/\varepsilon$, ta có nhiễu:

$$
\sqrt{2\eta\varepsilon}\,\mathcal{N}(0,I).
$$

Trong bài toán local entropy với $\mathcal{H}=I$, hàm năng lượng có thể viết là:

$$
U(x')=f(x')+\frac{1}{2t}\lVert x'-x^k\rVert^2.
$$

Do đó gradient cần dùng trong Langevin là:

$$
\nabla U(x')=\nabla f(x')+\frac{1}{t}(x'-x^k).
$$

Trong huấn luyện neural network, $\nabla f(x')$ thường được thay bằng stochastic gradient trên mini-batch:

$$
\widetilde{\nabla}f(x').
$$

Từ đây ta có bước cập nhật overdamped Langevin trong MCSDCA-odLD:

$$
x_{i+1}^k
=
x_i^k
-\eta\left(\widetilde{\nabla}f(x_i^k)+\frac{1}{t}(x_i^k-x^k)\right)
+\sqrt{2\eta\varepsilon}\,\mathcal{N}(0,I).
$$

Trực giác của công thức này:

- Thành phần $\widetilde{\nabla}f(x_i^k)$ kéo mẫu về vùng loss thấp.
- Thành phần $\frac{1}{t}(x_i^k-x^k)$ giữ mẫu không đi quá xa nghiệm hiện tại $x^k$.
- Thành phần nhiễu Gaussian giúp chuỗi khám phá vùng lân cận thay vì kẹt ở một điểm.

### 7. Hai biến thể Langevin trong paper

Paper xây dựng hai biến thể thực tế của MCSDCA cho deep learning.

**(1) MCSDCA-odLD - overdamped Langevin dynamics**

Đây là biến thể đơn giản hơn. Nó dùng overdamped Langevin dynamics trong vòng lặp lấy mẫu. Tại vòng lặp ngoài $k$:

1. Khởi tạo chuỗi Markov tại $x_0^k=x^k$.
2. Chạy Langevin trong $n_k$ bước.
3. Bỏ $b$ mẫu đầu tiên.
4. Lấy trung bình các mẫu còn lại:

$$
y_k=\frac{1}{n_k-b}\sum_{i=b}^{n_k-1}x_i^k.
$$

5. Giải bài toán lồi con:

$$
x_{k+1}=\arg\min_x
\left\{
\frac{1}{2t}\lVert x\rVert^2
-\frac{1}{t}\langle x,y_k\rangle
+\frac{\gamma_k}{2}\lVert x-x^k\rVert^2
\right\}.
$$

Lấy đạo hàm theo $x$:

$$
\frac{1}{t}x-\frac{1}{t}y_k+\gamma_k(x-x^k)=0.
$$

Suy ra nghiệm dạng đóng:

$$
x^{k+1}=\frac{t\gamma_k}{1+t\gamma_k}x^k+
\frac{1}{1+t\gamma_k}y_k.
$$

Công thức này rất quan trọng vì nó cho thấy bước DCA ngoài cùng không cần giải một bài toán tối ưu phức tạp trong trường hợp local entropy với $\mathcal{H}=I$. Ta chỉ cần trộn giữa nghiệm hiện tại $x^k$ và trung bình Markov $y_k$.

**(2) MCSDCA-udLD - underdamped Langevin dynamics**

Biến thể thứ hai dùng underdamped Langevin dynamics. Khác với overdamped, underdamped thêm biến vận tốc $V_t$:

$$
dV_t=-\mu V_tdt-\nabla U(X_t)dt+\sqrt{2\mu\beta^{-1}}dB_t,
$$

$$
dX_t=V_tdt.
$$

Trực giác là thay vì để mẫu di chuyển kiểu random walk nhiều zigzag, ta thêm quán tính/momentum để quá trình lấy mẫu có thể đi xa hơn theo một hướng hợp lý. Vì vậy underdamped Langevin thường được kỳ vọng hội tụ nhanh hơn tới phân phối mục tiêu. Đổi lại, nó phải lưu thêm vận tốc và công thức rời rạc hóa phức tạp hơn.

Trong paper, MCSDCA-udLD dùng sơ đồ rời rạc hóa của Cheng và cộng sự (2018). Mỗi bước lấy mẫu cập nhật cả vị trí $x_i^k$ và vận tốc $v_i^k$, sau đó vẫn tính:

$$
y_k=\frac{1}{n_k-b}\sum_{i=b}^{n_k-1}x_i^k,
$$

và dùng cùng bài toán lồi con như MCSDCA-odLD:

$$
x^{k+1}=\frac{t\gamma_k}{1+t\gamma_k}x^k+
\frac{1}{1+t\gamma_k}y_k.
$$

Như vậy, khác biệt chính giữa odLD và udLD nằm ở cách sinh chuỗi Markov bên trong. Bước DCA bên ngoài vẫn giữ cùng cấu trúc.

### 8. Tóm tắt giả mã theo paper

**Algorithm 1 - Markov Chain Stochastic DCA tổng quát**

```text
Input:
  x0, dãy gamma_k > 0, độ dài chuỗi n_k,
  số burn-in b, ma trận A >= I

For k = 0, 1, 2, ...:
  1. Chạy chuỗi Markov xi_0^k -> ... -> xi_{n_k-1}^k
     có phân phối cân bằng P(xi | x_k)

  2. Ước lượng subgradient:
     y_k = 1/(n_k-b) * sum_{i=b}^{n_k-1} v(x_k, xi_i^k)

  3. Giải bài toán lồi con:
     min_x { G(x) - <x, y_k> + gamma_k/2 * ||x - x_k||_A^2 }
     để thu được x_{k+1}

  4. k = k + 1
```

**Algorithm 2 - MCSDCA-odLD**

```text
Input:
  x0, gamma_k, n_k, burn-in b,
  epsilon > 0, eta > 0, t > 0

For k = 0, 1, 2, ...:
  1. Khởi tạo x_0^k = x^k

  2. Với i = 0, ..., n_k - 2:
       nhận mini-batch dữ liệu
       tính stochastic gradient \tilde{\nabla}f(x_i^k)
       cập nhật:
       x_{i+1}^k = x_i^k
                   - eta * (\tilde{\nabla}f(x_i^k) + 1/t * (x_i^k - x^k))
                   + sqrt(2 * eta * epsilon) * N(0, I)

  3. Tính y_k = average(x_b^k, ..., x_{n_k-1}^k)

  4. Cập nhật:
       x^{k+1} = (t gamma_k)/(1 + t gamma_k) * x^k
                 + 1/(1 + t gamma_k) * y_k
```

**Algorithm 3 - MCSDCA-udLD**

```text
Input:
  x0, gamma_k, n_k, burn-in b,
  epsilon > 0, delta > 0, t > 0

For k = 0, 1, 2, ...:
  1. Khởi tạo x_0^k = x^k và v_0^k = 0

  2. Với i = 0, ..., n_k - 2:
       nhận mini-batch dữ liệu
       tính stochastic gradient \tilde{\nabla}f(x_i^k)
       dùng underdamped Langevin discretization
       để lấy mẫu cặp mới (x_{i+1}^k, v_{i+1}^k)

  3. Tính y_k = average(x_b^k, ..., x_{n_k-1}^k)

  4. Cập nhật:
       x^{k+1} = (t gamma_k)/(1 + t gamma_k) * x^k
                 + 1/(1 + t gamma_k) * y_k
```

Ở đây tôi không chép lại toàn bộ công thức điều kiện Gaussian của Algorithm 3 vì nó khá dài và dễ làm rối phần cơ sở lý thuyết. Điều cần giữ lại là: udLD lấy mẫu trên không gian mở rộng $(x,v)$, dùng vận tốc để tăng tốc hội tụ, còn bước cập nhật DCA cuối vẫn giống odLD.

### 9. Kết quả hội tụ cần hiểu đúng

Trong bài báo, tác giả chứng minh hai loại hội tụ: hội tụ không tiệm cận và hội tụ tiệm cận. Đây là phần đảm bảo thuật toán MCSDCA không chỉ là một heuristic lấy mẫu, mà có nền tảng lý thuyết cho bài toán DC ngẫu nhiên với mẫu Markov.

**1. Hội tụ không tiệm cận**

Hội tụ không tiệm cận trả lời câu hỏi: sau hữu hạn $T$ bước lặp, thuật toán đã tiến gần đến điểm tới hạn đến mức nào?

Trong tối ưu không lồi, ta không nên nói thuật toán chắc chắn tìm được nghiệm tối ưu toàn cục. Khái niệm đúng hơn là điểm tới hạn. Một điểm $x^*$ được gọi là $\epsilon$-critical nếu:

$$
\operatorname{dist}(\partial G(x^*),\partial H(x^*))\le \epsilon.
$$

Paper cũng dùng khái niệm nearly $\epsilon$-critical: tồn tại một điểm $\bar{x}$ gần $x^*$, với $\lVert \bar{x}-x^*\rVert=O(\epsilon)$, sao cho:

$$
\operatorname{dist}(\partial G(\bar{x}),\partial H(\bar{x}))\le \epsilon.
$$

Kết quả không tiệm cận của paper cho biết, dưới các giả thiết phù hợp về độ dài chuỗi Markov $n_k$, hệ số $\gamma_k$, sai số giải bài toán con và tính $L$-smooth của $G$ hoặc $H$, thuật toán có thể tìm được điểm $\epsilon$-critical hoặc nearly $\epsilon$-critical theo kỳ vọng với tốc độ hội tụ xác định.

**2. Hội tụ tiệm cận**

Hội tụ tiệm cận trả lời câu hỏi: nếu thuật toán chạy rất lâu, dãy nghiệm sinh ra sẽ đi về đâu?

Theo paper, dưới các giả thiết phù hợp, ta có các kết luận chính:

- Tổng bình phương độ dịch chuyển giữa các bước lặp là hữu hạn:

$$
\sum_{k=1}^{\infty}\lVert x^{k+1}-x^k\rVert_A^2 < +\infty
\quad \text{almost surely}.
$$

- Do đó khoảng cách giữa hai bước lặp liên tiếp dần tiến về $0$:

$$
\lVert x^{k+1}-x^k\rVert_A \to 0.
$$

- Nếu dãy $\{x^k\}$ và $\{y^k\}$ bị chặn gần như chắc chắn, thì mọi điểm tụ của $\{x^k\}$ đều là critical point của hàm DC gốc $F$.

Nói ngắn gọn: hội tụ tiệm cận không có nghĩa là tìm được nghiệm tối ưu toàn cục, mà là các điểm tụ của thuật toán thỏa điều kiện tới hạn DC.

**3. Chuỗi Markov không đồng nhất theo thời gian**

Paper còn mở rộng phân tích cho chuỗi Markov không đồng nhất theo thời gian (time-inhomogeneous Markov chains), tức là luật chuyển trạng thái của chuỗi có thể thay đổi theo thời gian. Đây là trường hợp khó hơn chuỗi Markov đồng nhất, nhưng lại xuất hiện tự nhiên trong một số cơ chế lấy mẫu dựa trên diffusion.

Kết quả chính là: với các giả thiết ergodic phù hợp, các định lý hội tụ không tiệm cận và tiệm cận ở trên vẫn tiếp tục đúng trong trường hợp time-inhomogeneous.

### 10. Kết luận ngắn cho phần cơ sở lý thuyết

Vậy toàn bộ ý tưởng MCSDCA có thể tóm lại như sau:

- DCA xử lý bài toán không lồi bằng cách viết $F(x)=G(x)-H(x)$ và tuyến tính hóa $H$.
- Trong bài toán stochastic DC nội sinh, subgradient của $H$ chỉ có dạng kỳ vọng theo $P(\xi\mid x)$.
- Vì không lấy được mẫu i.i.d. từ $P(\xi\mid x)$, MCSDCA dùng chuỗi Markov có phân phối cân bằng là $P(\xi\mid x)$ để ước lượng subgradient.
- Trong deep learning, PDE/local-entropy regularization tạo ra một objective được làm mượt và có cấu trúc DC.
- Langevin dynamics là công cụ để xây dựng chuỗi Markov lấy mẫu từ phân phối local entropy.
- Hai biến thể thực tế là MCSDCA-odLD và MCSDCA-udLD, khác nhau ở cách sinh chuỗi Markov, nhưng cùng dùng bước cập nhật DCA dạng đóng.

Điểm cần nhớ nhất cho thí nghiệm predictor sau này là: MCSDCA không đơn thuần là một optimizer thay Adam theo kiểu thay learning rate. Nó tối ưu một phiên bản local-entropy/DC của loss, dùng Markov chain để ước lượng phần subgradient khó tính, và nhờ đó có xu hướng ưu tiên các vùng nghiệm rộng, ổn định hơn trong landscape của neural network.

## II. Áp dụng MCSDCA như một optimizer cho predictor của LeWM với bộ dữ liệu PushT
## 1. Đặt vấn đề

Trong LeWM, predictor giữ vai trò học dynamics trong latent space. Pipeline liên quan trực tiếp tới predictor có thể tóm tắt như sau:

```text
observation -> encoder -> latent z_t
latent z_t + action a_t -> predictor -> predicted latent z_hat_{t+1}
predicted latent rollout -> CEM/MPC planning
```

Với một batch trajectory, encoder tạo latent:

$$
z_t = encoder_\theta(o_t)
$$

Predictor dự đoán latent kế tiếp:

$$
\hat z_{t+1} = predictor_\phi(z_t, a_t)
$$

Target latent:

$$
z_{t+1} = encoder_\theta(o_{t+1})
$$

Loss chính trong code LeWM hiện tại là:

$$
L_{LeWM} = L_{pred} + \lambda SIGReg(Z)
$$

trong đó:

$$
L_{pred} = \lVert \hat z_{t+1} - z_{t+1} \rVert^2
$$

Mục tiêu của thí nghiệm này là kiểm tra liệu **MCSDCA** có thể đóng vai trò như một optimizer riêng cho phần predictor-side của LeWM hay không. Ở prototype đầu tiên, ta không thay đổi kiến trúc LeWM và không train toàn bộ world model bằng MCSDCA. Ta chỉ thay optimizer cho nhóm tham số dynamics head:

```text
action_encoder + predictor + pred_proj
```

Encoder và projector được freeze hoặc detach để latent target ổn định hơn trong giai đoạn thử nghiệm đầu tiên.

## 2. Vì sao MCSDCA phù hợp với predictor training?

### 2.1. Objective theo tham số neural network là không lồi

Nếu chỉ nhìn MSE theo output dự đoán thì loss là lồi theo `pred_emb`. Tuy nhiên trong training thực tế, loss là hàm theo tham số neural network:

$$
x = (\theta, \phi)
$$

hoặc trong thí nghiệm predictor-only:

$$
x = \phi
$$

Khi đó:

$$
f(x) = L_{pred}(x) + \lambda SIGReg(Z)
$$

Hàm này không lồi vì encoder, action encoder, predictor transformer và projection head đều là neural networks có composition phi tuyến. Với rollout nhiều bước, predictor còn được gọi lặp lại trên chính output trước đó, làm loss landscape khó hơn one-step prediction.

### 2.2. Loss hiện tại chủ yếu khả vi, nên không nên nhấn mạnh quá mức tính nonsmooth

Trong `le-wm/train.py`, loss được tính trực tiếp:

```python
output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
```

`pred_loss` là MSE và `SIGReg` trong `le-wm/module.py` dùng các phép toán khả vi như random projection, `cos`, `sin`, `square`, `mean`. Vì vậy, với code hiện tại, luận điểm mạnh nhất không phải là "loss không trơn", mà là:

- objective theo tham số mạng là không lồi;
- gradient là stochastic do mini-batch;
- dữ liệu trajectory có tính Markov và không i.i.d.;
- MCSDCA có thể tối ưu phiên bản PDE/local-entropy regularized của objective.

### 2.3. Training predictor là stochastic optimization

Objective lý tưởng là kỳ vọng trên dữ liệu transition:

$$
f(x) =
\mathbb{E}_{(o_t,a_t,o_{t+1}) \sim \mathcal{D}}
\left[
\lVert predictor_\phi(encoder_\theta(o_t), a_t) - encoder_\theta(o_{t+1}) \rVert^2
\right]
$$

Trong thực tế, ta dùng mini-batch:

$$
\hat f_B(x) =
\frac{1}{|B|}
\sum_{(o_t,a_t,o_{t+1}) \in B}
\ell(x; o_t,a_t,o_{t+1})
$$

Do đó gradient $\nabla \hat f_B(x)$ là stochastic gradient.

### 2.4. Dữ liệu trajectory có tính Markov và không i.i.d.

Dữ liệu trong LeWM có dạng chuỗi:

$$
\tau = (o_0, a_0, o_1, a_1, \ldots, o_T)
$$

Transition kế tiếp phụ thuộc vào trạng thái và action hiện tại:

$$
o_{t+1} \sim P(o_{t+1} \mid o_t, a_t)
$$

Nếu coi một sample là:

$$
X_t = (o_t, a_t, o_{t+1})
$$

thì $X_t$ và $X_{t+1}$ không độc lập vì cùng chia sẻ $o_{t+1}$. Lập luận tương tự đúng trong latent space. Đây là điểm phù hợp tự nhiên với MCSDCA, vì thuật toán được thiết kế cho bối cảnh không lấy được mẫu i.i.d. hoàn hảo mà phải xử lý mẫu phụ thuộc kiểu Markov.

## 3. Cơ sở lý thuyết MCSDCA

### 3.1. DC programming và DCA

Một bài toán DC có dạng:

$$
\min_x F(x) = G(x) - H(x)
$$

trong đó $G$ và $H$ là hai hàm lồi. DCA giải bài toán này bằng cách tuyến tính hóa $H$ tại nghiệm hiện tại $x_k$:

$$
y_k \in \partial H(x_k)
$$

rồi giải bài toán con:

$$
x_{k+1} \in \arg\min_x \{G(x) - \langle x, y_k \rangle\}
$$

### 3.2. MCSDCA

Trong MCSDCA, subgradient của $H$ không được tính trực tiếp mà được ước lượng bằng mẫu từ một Markov chain:

$$
\mathbb{E}_{P(\xi \mid x)}[v(x,\xi)] \in \partial H(x)
$$

Tại mỗi outer iteration:

1. Cố định nghiệm hiện tại $x_k$.
2. Chạy Markov chain có phân phối cân bằng liên quan tới $P(\xi \mid x_k)$.
3. Bỏ `burn_in` sample đầu.
4. Lấy trung bình các trạng thái còn lại để ước lượng $y_k$.
5. Cập nhật nghiệm bằng bước DCA.

Có thể nhớ ngắn gọn:

$$
\boxed{MCSDCA = DCA + Markov\ chain\ subgradient\ estimation}
$$

### 3.3. PDE/local entropy regularization trong deep learning

Với deep learning, paper MCSDCA không tối ưu trực tiếp loss thô $f(x)$, mà tối ưu một phiên bản làm mịn sinh bởi PDE/local entropy. Phiên bản làm mịn có thể viết dưới dạng DC:

$$
\tilde f(x,t) = G(x) - H(x)
$$

Trong đó $G$ thường là quadratic form lồi, còn $H$ liên quan tới log-integral/local entropy và được chứng minh là lồi trong paper. Điều này tạo cầu nối:

```text
deep learning loss f(x)
-> PDE/local entropy smoothing
-> DC objective
-> MCSDCA optimizer
```

### 3.4. Hai biến thể dùng trong thí nghiệm

**MCSDCA-odLD** dùng overdamped Langevin dynamics:

$$
x'_{i+1}
= x'_i
- \eta\left[\nabla \hat f_B(x'_i) + \frac{1}{t}(x'_i-x_k)\right]
+ \sqrt{2\eta\epsilon}\xi_i
$$

**MCSDCA-udLD** thêm velocity để tăng tốc sampling:

```text
state = (position, velocity)
position = model parameters
velocity = auxiliary momentum-like variable
```

Sau khi lấy trung bình các sample còn lại:

$$
y_k = \frac{1}{n-b}\sum_{i=b}^{n-1}x'_i
$$

bước DCA closed-form được dùng:

$$
x_{k+1}
=
\frac{t\gamma_k}{1+t\gamma_k}x_k
+
\frac{1}{1+t\gamma_k}y_k
$$

## 4. Kịch bản thực nghiệm

### 4.1. Dataset

Dataset chính: **PushT**.

Lý do chọn PushT:

- LeWM đã có config training/evaluation cho PushT.
- PushT là task điều khiển theo trajectory, phù hợp với lập luận Markov/non-i.i.d.
- Chất lượng predictor ảnh hưởng trực tiếp tới rollout và planning bằng CEM.
- Task đủ nhỏ để thử nghiệm optimizer trước khi mở rộng sang Cube, Reacher hoặc TwoRoom.

Config tham chiếu từ LeWM:

```text
history_size = 3
num_preds = 1
batch_size = 128
train_split = 0.9
optimizer gốc = AdamW(lr=5e-5, weight_decay=1e-3)
loss = pred_loss + 0.09 * SIGReg
```

### 4.2. Thiết lập prototype đầu tiên

Phạm vi cập nhật:

```text
trainable with MCSDCA:
  action_encoder
  predictor
  pred_proj

frozen/detached:
  encoder
  projector
```

Loss chính:

$$
L_{one-step}
=
\lVert \hat z_{t+1} - stopgrad(z_{t+1}) \rVert^2
$$

Trong predictor-only setting, nếu `SIGReg(Z)` chỉ phụ thuộc encoder đã detach thì nó không tạo gradient hữu ích cho predictor. Vì vậy thí nghiệm đầu nên dùng `pred_loss` làm loss chính, đồng thời log `SIGReg` như metric phụ nếu cần.

### 4.3. Baseline optimizer

Nên so sánh với:

- `AdamW`: baseline chính vì đang là optimizer gốc trong LeWM.
- `Adam`: baseline phổ biến trong deep learning.
- `SGD + momentum`: baseline cổ điển, giúp thấy vai trò adaptive optimizer.
- `RMSprop`: baseline stochastic adaptive thường dùng cho dữ liệu noisy.
- `Adagrad`: baseline có trong paper MCSDCA.
- `MCSDCA-odLD`: biến thể đơn giản, dễ debug.
- `MCSDCA-udLD`: biến thể có velocity, có thể hội tụ nhanh hơn.

So sánh nên dùng trục `backprop_calls` ngoài trục epoch, vì một outer step MCSDCA dùng nhiều backward pass.

## 5. Metric đánh giá

### 5.1. Predictor metrics

| Metric | Ý nghĩa |
|---|---|
| One-step latent MSE | Đo trực tiếp chất lượng dự đoán $z_{t+1}$ |
| Multi-step rollout MSE | Đo sai số tích lũy khi predictor tự rollout |
| Latent norm drift | Kiểm tra latent rollout có bị nổ hoặc trôi khỏi phân phối thật không |
| Predicted latent variance | Kiểm tra predictor có collapse về output gần hằng số không |
| Train/val gap | Đánh giá overfit |

### 5.2. Planning metrics

| Metric | Ý nghĩa |
|---|---|
| PushT success/score | Metric task-level quan trọng nhất |
| CEM final cost | Kiểm tra planner có tìm được action sequence tốt không |
| Imagined-vs-real rollout gap | Đo mức CEM khai thác lỗi model |
| Wall-clock time | Chi phí thực tế của optimizer |
| Backprop calls | So sánh compute budget công bằng hơn |

## 6. Bảng kết quả placeholder

### 6.1. One-step predictor training

| Optimizer | Backprop calls | Train MSE | Val MSE | Train/Val gap | Latent drift | Time |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | TBD | TBD | TBD | TBD | TBD | TBD |
| Adam | TBD | TBD | TBD | TBD | TBD | TBD |
| SGD + momentum | TBD | TBD | TBD | TBD | TBD | TBD |
| RMSprop | TBD | TBD | TBD | TBD | TBD | TBD |
| Adagrad | TBD | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-odLD | TBD | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-udLD | TBD | TBD | TBD | TBD | TBD | TBD |

### 6.2. Multi-step rollout loss

| Optimizer | Horizon | Rollout MSE@1 | Rollout MSE@3 | Rollout MSE@5 | Latent drift | Time |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 5 | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-odLD | 5 | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-udLD | 5 | TBD | TBD | TBD | TBD | TBD |

### 6.3. PushT planning

| Optimizer | CEM horizon | Eval episodes | PushT score | CEM cost | Eval time |
|---|---:|---:|---:|---:|---:|
| AdamW | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-odLD | TBD | TBD | TBD | TBD | TBD |
| MCSDCA-udLD | TBD | TBD | TBD | TBD | TBD |

## 7. Ablation nên chạy sau prototype

1. **Dynamics head vs core predictor-only**
   - Dynamics head: `action_encoder + predictor + pred_proj`.
   - Core predictor-only: chỉ `predictor`.
   - Mục tiêu: xác định phần nào thật sự cần MCSDCA.

2. **One-step vs multi-step rollout**
   - `H=1`, `H=3`, `H=5`, có thể thêm `H=10`.
   - Mục tiêu: kiểm tra MCSDCA có giúp giảm sai số tích lũy không.

3. **odLD vs udLD**
   - odLD đơn giản và ổn định hơn để debug.
   - udLD có thể sampling hiệu quả hơn nhưng nhiều hyperparameter hơn.

4. **Compute-budget matching**
   - So sánh theo cùng số `backprop_calls`.
   - So sánh thêm theo cùng wall-clock time.

5. **Hyperparameter sensitivity**
   - `langevin_steps`: 4, 8, 16.
   - `burn_in`: 1, 2, 4.
   - `epsilon`: 1e-8, 1e-6, 1e-4.
   - `local_entropy_time`: 1e2, 1e3, 1e4.
   - `gamma`: 1e-5, 1e-4, 1e-3.

## 8. Kết luận kỳ vọng

Thí nghiệm đầu tiên không cần chứng minh MCSDCA thắng mọi optimizer. Mục tiêu đúng hơn là kiểm tra:

- MCSDCA có thể gắn vào predictor-side của LeWM mà không phá kiến trúc.
- Thuật toán chạy được với loss predictor và cập nhật đúng nhóm tham số.
- So sánh với AdamW/Adam/SGD/RMSprop/Adagrad có thể thực hiện công bằng theo `backprop_calls`.
- Nếu one-step MSE không vượt AdamW nhưng rollout MSE hoặc planning score tốt hơn, đó vẫn là tín hiệu quan trọng vì predictor trong LeWM được dùng chủ yếu cho imagined rollout và CEM planning.

Luận điểm nghiên cứu chính:

> MCSDCA không chỉ là một optimizer thay thế AdamW, mà là một cơ chế local-entropy regularized training cho latent dynamics predictor. Nếu thành công, nó có thể giúp predictor rơi vào vùng tham số phẳng và ổn định hơn, từ đó giảm lỗi tích lũy trong long-horizon imagined rollout.
