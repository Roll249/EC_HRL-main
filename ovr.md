# Tài liệu chi tiết: `Main_Simulation.ipynb`

Tài liệu này mô tả **toàn bộ** notebook mô phỏng mạng không gian–mặt đất kết hợp học tăng cường phân tầng (HRL) cho bài toán **offload tính toán** từ phương tiện (vehicles) lên **RSU, UAV, HAP, LEO**. Trong repo chỉ có **một** file notebook: `Main_Simulation.ipynb`.

---

## 1. Mục đích và bài toán

- **Bối cảnh**: Nhiều nút mạng với bán kính phủ khác nhau dọc trục (RSU gần mặt đất, UAV/HAP/LEO ở độ cao lớn dần).
- **Người dùng**: `V` phương tiện, mỗi xe có vị trí ngẫu nhiên trên đoạn đường, tốc độ `VN_spd` (phân phối chuẩn cắt), và yêu cầu **dịch vụ** trong tập `{1,…,6}` (mỗi loại nút chỉ phục vụ một số dịch vụ được gán ngẫu nhiên khi khởi tạo kịch bản).
- **Quyết định cho mỗi xe** (vector 3 thành phần trong nhiều đoạn code):
  - **Lớp** (layer): `0=RSU`, `1=UAV`, `2=HAP`, `3=LEO`.
  - **Chỉ số nút** trong lớp đó (phải nằm trong tập nút “có thể phủ” xe).
  - **Tỷ lệ offload** `∈ [0,1]`: phần tính toán gửi lên edge so với tính local (trong `Task_Proc_Main` dùng `decisions[v][2]` làm trọng số giữa thời gian/năng lượng offload và local).

- **Chi phí**: Tổng hợp **thời gian** và **năng lượng** (trọng số `gamma_1`, `gamma_2`), kèm các chỉ số vi phạm deadline, thời gian lưu lạng (sojourn), và yêu cầu dịch vụ không được nút hỗ trợ.

---

## 2. Thư viện (Cell 0)

Cell đầu tiên nạp:

| Nhóm | Thư viện |
|------|-----------|
| Số học / dữ liệu | `numpy`, `pandas`, `math`, `itertools`, `collections.deque` |
| Ngẫu nhiên | `random`, `numpy.random`, `scipy.stats.truncnorm` |
| Đồ họa | `matplotlib.pyplot`, `matplotlib.gridspec` |
| IO / khoa học | `scipy.io`, `scipy.stats.expon` |
| Giao diện notebook | `IPython.display.clear_output` |
| Học sâu | `tensorflow.keras`: `Model`, `Sequential`, `Dense`, `Embedding`, `Reshape`, `Adam` |

`from __future__ import print_function` hỗ trợ tương thích Python 2/3 (thừa kế từ code cũ).

---

## 3. Các hàm lõi mô hình mạng và người dùng

### 3.1. `Loc_Fun(...)` — Cell 1

**Vai trò**: Sinh **vị trí** các RSU, UAV, HAP, LEO và ma trận **liên kết phủ** (ai nằm trong vùng phủ của ai).

**Tham số**: `RSU_T, UAV_T, HAP_T, LEO_T` (số nút mỗi loại); `RSU_r, UAV_r, HAP_r, LEO_r` (bán kính phủ ngang); `LEO_alt, HAP_alt, UAV_alt` (độ cao).

**Logic chính**:

1. Vòng `while True` lặp cho đến khi cấu hình thỏa **đầy đủ kết nối** (mọi RSU có ít nhất một UAV/HAP/LEO trong tầm; mọi UAV/HAP/LEO cũng có liên kết theo các ma trận `U_HAP_asign`, `U_LEO_asign`, `H_LEO_asign` — tổng theo hàng đều > 0).
2. **RSU**: Đặt RSU đầu tại `(RSU_r/2, 5)`, các RSU sau dịch ngẫu nhiên theo trục x trong khoảng `[RSU_r/2, 1.5*RSU_r]`, `y=5` cố định → chuỗi RSU dọc trục x.
3. **UAV / HAP**: Điểm đầu tại `x = bán_kính/2`, `z = độ cao`, các nút sau tăng x ngẫu nhiên trong khoảng `[r, 2r]`, `y=0`.
4. **LEO**: Các vệ tinh cùng `x = LEO_r`, `z = LEO_alt`.
5. Ma trận gán:
   - `R_UAV_asign`, `R_HAP_asign`, `R_LEO_asign`: RSU–UAV/HAP/LEO trong khoảng cách 2D ≤ bán kính tương ứng.
   - `U_HAP_asign`, `U_LEO_asign`, `H_LEO_asign`: điều kiện so sánh khoảng trên trục x (logic “dải phủ” giữa các tầng).

**Trả về**: `(RSU_loc, UAV_loc, HAP_loc, LEO_loc)` — mỗi `*_loc` là mảng tọa độ (2D cho RSU, 3D cho các tầng bay/vũ trụ).

---

### 3.2. `VN_EN_Assign(IP, V_spd)` — Cell 2

**Vai trò**: Gán **từng phương tiện** tới các nút có thể phủ, tính **khoảng cách**, **thời gian lưu lạng (sojourn)** trong vùng phủ RSU, và các tương tự cho UAV/HAP/LEO.

**Đầu vào**: `IP` chứa `V`, số nút, `*_loc`, `*_r`; `V_spd` — vector tốc độ từng xe.

**Đặc điểm**:

- Vòng lặp `while True` đảm bảo mỗi xe có **ít nhất một** nút khả dụng trên **cả bốn** tầng (điều kiện `vr_sum`, `vu_sum`, `vh_sum`, `vs_sum` đều > 0 cho mọi xe).
- Với **RSU**: xe nằm trên đường `y=0`; kiểm tra xe có trong hình chiếu “đĩa phủ” của RSU (công thức căn bậc hai với `RSU_r`) và tính quãng đường dọc trục x trong vùng phủ, thời gian sojourn = quãng đường / `V_spd[i]`. Điều kiện `sum(V_RSU_asign[i][:]) < 4` để giới hạn số RSU “active” trên một xe (thiết kế thực nghiệm).
- Với **UAV/HAP/LEO**: điều kiện xe nằm trong khoảng x `[node_x - r, node_x + r]`; tính thành phần khoảng cách và khoảng cách 3D cho Shannon sau này.

**Trả về** (list `out`, **17** phần tử, chỉ số 0…16):

| Index | Biến trong code |
|-------|-----------------|
| 0 | `VN_loc` |
| 1–12 | `V_RSU_asign` … `V_LEO_dist` (cùng thứ tự như trên) |
| 13 | `V_RSU_d` |
| 14 | `V_UAV_d` |
| 15 | `V_HAP_d` |
| 16 | `V_LEO_d` |

Trong **Cell 28**, các biến tên `VN_RSU_dist_Sj` … `VN_LEO_dist_Sj` gán từ `VN_assign_OUT[13]`–`[16]` thực chất trùng với **`V_RSU_d` … `V_LEO_d`** của hàm (tên biến ở cell 28 dễ nhầm với khoảng cách sojourn).

Cell 9 chỉ gán đến `VN_LEO_dist` (index 12) rồi `append` các ma trận khoảng cách vào `VN_Main_IP` — không dùng index 13–16 trực tiếp với tên `*_Sj`.

---

### 3.3. `Chn_Capacity(N1N2_dist, BN2, PN1, b_0, theta)` — Cell 3

**Vai trò**: Công suất kênh Shannon (bit/s).

- Nhiễu `N0` cố định từ `-45 dBm`.
- Bước sóng `lam = c / BN2` (tần số mang `BN2`).
- **Path loss** dạng log-distance kết hợp tần số (92.45 + 20log10(dist/km) + 20log10(GHz)).
- **Channel gain**: `b_0 * (distance ** theta) * path_loss` (mô hình kênh tùy chỉnh trong code).
- **Công thức**: `C = BN2 * log2(1 + PN1 * gain / N0)`.

---

### 3.4. `Task_Proc_Main(IP, CR, DR_R_R, CU, DR_R_U, CH, DR_R_H, CL, DR_R_S, g1, g2, decisions)` — Cell 4

**Vai trò**: Với **toàn bộ** xe và một bảng quyết định `decisions[V][3]`, tính:

- Thời gian/năng lượng **tính local** (`TPcomp_v`, `EPcomp_v`) từ `TS`, `TSD`, `psi_dmp`, `Cm`, `Pcomp_m`.
- Với mỗi lớp offload: thời gian upload/download (`TS/DR`, `TSD/DR`), tính toán trên nút (`TS*psi_dmp/CR`…), cộng thành `TP_offl_*`, `EP_offl_*`.
- **Trộn local/offload** bằng `decisions[v][2]`: ví dụ `T_L[v] = max(α * T_offload, (1-α) * T_local)` và `T_E` là tổng có trọng số tương ứng (0.5 trên các nhánh truyền/tính trong nhiều chỗ).
- **Ràng buộc**:
  - Sojourn: phần thời gian offload `T_L_E` so với `Soj_T_V*` — nếu vượt thì tăng `No_of_Soj_L_Fail`.
  - Dịch vụ: `V_loc[2][v]` (loại dịch vụ xe cần) phải thuộc `Ser_RSU[r]` / `Ser_UAV[u]` / … — nếu không thì `No_of_Ser_H_Req`.
  - Deadline latency: nếu `T_L[v] > 4` thì `No_of_Ser_L_Fail`.
- **Chi phí tổng hợp**: `TC[v] = g1*T_L + g2*T_E`.

**Trả về** (list): `T_L`, `T_E`, `T_L_V`, `T_L_E`, `T_E_V`, `T_E_E`, `TP_loc`, `EP_loc`, `TC`, `No_of_Ser_L_Fail`, `No_of_Soj_L_Fail`, `No_of_Ser_H_Req`.

---

### 3.5. `Data_Rate(IP, decisions, RSU_B, UAV_B, HAP_B, LEO_B)` — Cell 5

**Vai trò**: Với băng thông đã **phân bổ theo xe** (`*_B[v][node]`) và khoảng cách từ `IP`, gọi `Chn_Capacity` để có `Rate_VR`, `Rate_VU`, `Rate_VH`, `Rate_VL` — dùng làm `DR_*` trong `Task_Proc_Main`.

---

### 3.6. `Resource_Allocation(IP, decisions, NOdes_assign_R, …)` — Cell 6

**Vai trò**: Chia đều **băng thông** và **tài nguyên tính toán** trên mỗi nút cho số người dùng đang chọn nút đó:

- `RSU_B_Assign[v][r] = RSU_B[0][r] / NOdes_assign_R[r]` (và tương tự UAV/HAP/LEO cho `C_Assign`).

**Trả về**: 8 ma trận `RSU_B_Assign` … `LEO_C_Assign`.

---

## 4. Agent DQN — `class Agent` (Cell 7)

Mạng **Q**: `Embedding(state_size → 10)` → `Reshape` → `Dense(50, relu)` × 2 → `Dense(action_size, linear)`. Có **target network** đồng bộ qua `alighn_target_model()` (tên hàm viết `alighn`).

- `store`: replay buffer `deque(maxlen=50000)`.
- `act`: nếu `rand ≤ epsilon` thì **vẫn** `predict` rồi `argmax` (epsilon không làm random action thuần; đây là chi tiết triển khai).
- `retrain`: sampling minibatch, cập nhật Q-learning kiểu `target[action] = r + gamma * max Q_target(next)`.

Cell 8 là bản comment của class (không chạy).

---

## 5. Tham số kịch bản và `VN_Main_IP` — Cell 9 (rất dài)

**Giá trị mẫu**:

- `gamma_1 = gamma_2 = 0.5` (cân bằng delay–energy).
- Bán kính: RSU 50 m, UAV 300 m, HAP 1000 m, LEO 3000 m.
- Số nút: `RSU_T = LEO_r/RSU_r`, `UAV_T = LEO_r/UAV_r`, `HAP_T = 1.5*LEO_r/HAP_r`, `LEO_T = 1` (công thức trong notebook).
- Băng/tính toán mặc định: ví dụ `B_RSU = 20e6`, `C_RSU = 20e9`, …
- **Phân bổ dịch vụ**: `service_allocation_*[node_id] = random.sample(services, services_per_*)`.
- Gọi `Loc_Fun` → tọa độ nút.
- `V = 200` xe, `TS`, `TSD`, `C_V`, công suất, v.v.
- `VN_spd`: **truncated normal** trong `[8,14]` m/s, mean 12.
- Xây list **`VN_Main_IP`** theo thứ tự append (chỉ số dùng xuyên suốt):

| Index | Nội dung (sau khi ghép đủ) |
|-------|----------------------------|
| 0 | `V` |
| 1 | `RSU_T` |
| 2 | `RSU_loc` |
| 3 | `RSU_r` |
| 4 | `UAV_T` |
| 5 | `UAV_loc` |
| 6 | `UAV_r` |
| 7 | `HAP_T` |
| 8 | `HAP_loc` |
| 9 | `HAP_r` |
| 10 | `LEO_T` |
| 11 | `LEO_loc` |
| 12 | `LEO_r` |
| 13–16 | `B_RSU` … `B_LEO` |
| 17–20 | `C_RSU` … `C_LEO` |
| 21–24 | `P_tx_r` … `P_tx_s` (ghi nhãn PNR, PNU, … trong `Data_Rate`) |
| 25–26 | `b_0`, `theta` |
| 27–30 | `VN_RSU_dist` … `VN_LEO_dist` |
| 31–32 | `TS`, `TSD` |
| 33 | `psi_dmp` |
| 34 | `Pcomp_m` |
| 35 | `P_tx_v` (Ptp_v) |
| 36–43 | `Pcomp_r`, `P_tx_r`, … xen kẽ theo code |
| 44 | `C_V` / `Cm` |
| 45–48 | `VN_RSU_Soj` … `VN_LEO_Soj` |
| 49–50 | `gamma_1`, `gamma_2` |
| 51–54 | `service_allocation_RSU` … `LEO` |
| 55 | `VN_loc` |

Sau đó notebook xây **`Random_decisions`**, đếm `NO_VUs_*`, gọi `Resource_Allocation` → `Data_Rate` → `Task_Proc_Main` để có baseline **Random**.

**Lưu ý**: Trong Cell 9, gán `T_E_V_R=Cost[3]` lặp lại ba lần — có vẻ là lỗi sao chép; nên kiểm tra khi dùng kết quả.

---

## 6. Không gian trạng thái / hành động phân tầng

### 6.1. Tầng cao (High) — Cell 12–13

- **`State_Space_H`**: tích Descartes của `H_S_D=[0,1,2,3]`, `H_S_CV=[0,1,2]`, `M_S=[0,1,2,3]`, `P_SL=[0,1,2,3]` → **4×3×4×4 = 192** trạng thái, mỗi state 4 chiều.
- **`Action_Space_H = [0,1,2,3]`**: chọn tầng RSU / UAV / HAP / LEO.

**`Next_State_H(a_h, Req_Ser, …)`**: với từng `a_h`, tính:

- Thành phần [0]: mã lớp (0–3).
- Thành phần [1]: mức **tải** trung bình trên các nút của lớp (chia `NO_VUs_* / Max_U_*`, rồi rời rạc hóa 3 mức).
- Thành phần [2]: cố định theo lớp (ví dụ UAV→2, HAP→1 trong code).
- Thành phần [3]: mức **xác suất** có nút trong lớp cung cấp `Req_Ser` (từ `service_allocation_*`).

Trả về `(S_N, row_index)` để map vào hàng của `State_Space_H`.

---

### 6.2. Tầng giữa (Mid) — Cell 17–19

- **`State_Space_M`**: định nghĩa trong Cell 17 (cấu trúc tương tự High, kích thước trong code).
- **`M_Action_Space(a_h, v_id, …)`**: với xe `v_id`, lấy vector gán nút (ví dụ `VN_RSU_asign[v_id]`) và tạo ma trận one-hot **mỗi hàng** là một nút khả dụng → không gian hành động **rời rạc theo nút thực tế**.

**`Next_State_M(...)`**: dựa trên nút chọn `a_m`, tính:

- [0]: mức tắc nghẽn `CV_EN` (4 bin).
- [1]: dịch vụ có trong `Service` của nút không (0/1).
- [2]: sojourn so với `cov/VU_speed`.

Trả về `(N_S_M, row_index_m, col_index_m, Soj_T)` — `Soj_T` dùng cho tầng thấp.

---

### 6.3. Tầng thấp (Low) — Cell 20

- `delta = 0.25`, `State_Space_L` từ tích các nhị phân (3 chiều, kích thước 2×2×2 = 8 nếu dùng đúng `F1_N,F2_N,F3_N`).
- **`Action_Space_L`**: các giá trị trong `[0, 1]` bước `delta` (tỷ lệ offload).

**Cảnh báo**: Trong vòng lặp khởi tạo state, code gán `state[0]=R_S_N[i1]` v.v.; trong cell lại định nghĩa `F1_N,F2_N,F3_N`. Cần **đồng bộ tên biến** (`F1_N` hoặc `R_S_N`) khi chạy, nếu không sẽ `NameError`.

---

### 6.4. `Learning_Cost(...)` — Cell 21

Phiên bản **một xe** (`v_id` — biến global trong notebook): tính chi phí tương tự `Task_Proc_Main` nhưng trả về thêm các **vi phạm**:

- `F1 = T_L_V - T_sog_me` (so với thời gian sojourn đo được từ `Next_State_M`).
- `F2 = T_L - DL_Req`.
- `F3 = T_L_E - EP_loc` (dạng so sánh trong code).

Output: `T_L`, `T_E`, các thành phần tách, `F1`, `F2`, `F3` cho reward và cập nhật `State_Space_L`.

---

## 7. Hàm phụ RL — Cell 14–16

- **`is_terminal_state`**, **`get_starting_location`**: chọn hàng ngẫu nhiên trong không gian trạng thái (không dùng ma trận reward đầy đủ kiểu gridworld cổ điển).
- **`get_next_action`**: ε-greedy trên `q_values` (bảng Q hoặc index hành động).
- **`get_next_action_M`**: chọn hành động **trong** tập cột hợp lệ `nonzero_indices_M_A_l` (map nút local sang chỉ số cột global `RSU_T+…`).

---

## 8. Huấn luyện HRL — Cell 23–27

**Cell 23**: Khởi tạo `num_episodes = 50`, ba cặp agent `agent_h`, `agent_h1`, `agent_h2` (cùng kích thước High), ba cặp `agent_m`, `agent_m1`, `agent_m2`, ba cặp `agent_l`, `agent_l1`, `agent_l2`. Siêu tham số: `gamma_h=0.7`, `epsilon_h=0.1`, `gamma_m`, `gamma_l` nhỏ hơn, `Max_U_*` giới hạn tải, `batch_size=2`.

**Cell 24–26**: Ba khối huấn luyện gần giống nhau (comment khác nhau ví dụ “LS DQN (0.5,0.5)” vs “(0.75,0.25)”) — mỗi khối dùng một bộ agent (`agent_h` vs `agent_h1` vs `agent_h2`, v.v.), khởi tạo `q_values_*` ngẫu nhiên trong [-1000,1000].

**Vòng lặp một episode** (tóm tắt):

1. Chọn `v_id` (thường 0 trong đoạn mẫu), `Req_Ser` ngẫu nhiên.
2. **High**: `agent_h.act` → `Next_State_H`.
3. **Mid**: `M_Action_Space` → `get_next_action_M` → `Next_State_M` (lấy `T_soj_me`).
4. **Low**: `get_next_action` trên `Action_Space_L` → `Learning_Cost` → từ `F1,F2,F3` cập nhật `new_state_l` (binary), `reward_l = 0.5*Tot_LC + 0.5*Tot_EC`, **cùng reward** gán cho `reward_m`, `reward_h`.
5. `store` vào replay của cả ba tầng; cuối episode `retrain` batch nhỏ, định kỳ `save` file `.weights.h5`.

**Cell 27**: `agent_h.load(...)` nạp trọng số đã lưu (infer / tiếp tục mô phỏng).

---

## 9. Mô phỏng đa phương pháp theo quy mô V — Cell 28 (file lớn)

- **`Tot_V`**: danh sách số người dùng (20→200, lặp nhiều nhóm).
- Với **mỗi** `V`: lặp lại toàn bộ pipeline gán vị trí, `VN_EN_Assign`, xây `VN_Main_IP`, rồi tính:
  - **Random** (`Random_decisions`),
  - **HRL** (dùng policy đã train / Q tables tùy đoạn),
  - Các biến thể `HRL1`, `HRL2`, **Local**, **DA** (decision algorithm khác trong notebook),
  - Các chỉ số có hậu tố `_SD_` (có thể “standard deviation” hoặc nhánh tính song — đọc đúng theo tên biến trong cell).

Thu thập: tổng/trung bình `Tot_T_L_*`, `Tot_T_E_*`, `Tot_TC_*`, số lần vi phạm latency/sojourn/service.

---

## 10. Hậu xử lý và hình vẽ — Cell 29–37

- **Cell 29**: Chuẩn hóa dữ liệu `Tot_V` cho biểu đồ.
- **Cell 30**: Comment hướng dẫn uncomment để xem kết quả mới.
- **Cell 31**: Ghi **`result.xlsx`** (cần `openpyxl` hoặc engine tương thích nếu chạy).
- **Cell 32–37**: `plt.plot(Users, ...)` so sánh các đường **HRL** vs baseline trên tổng chi phí, thời gian, năng lượng, số lỗi sojourn, số lỗi dịch vụ, v.v.

---

## 11. Cell ngắn / gỡ lỗi

| Cell | Nội dung |
|------|----------|
| 10 | Một biểu thức `service_allocation_UAV[1]` — kiểm tra nhanh dict |
| 11 | Trống hoặc output cũ |
| 22 | Trống (phân cách) |

---

## 12. Thứ tự chạy khuyến nghị

1. Cell 0 → 7 (import + hàm + Agent).
2. Cell 9 (tham số + `VN_Main_IP` + baseline Random) — **phải chạy trước** các cell phụ thuộc `VN_Main_IP`, `NO_VUs_*`, `Rate_*_R`.
3. Cell 12–21 (không gian trạng thái + `Learning_Cost`).
4. Cell 23 → 24–26 (train) → 27 (load).
5. Cell 28 trở đi (đánh giá quy mô + plot).

---

## 13. Phụ thuộc môi trường

- Python 3.x, **TensorFlow 2.x** (Keras), NumPy, SciPy, Pandas, Matplotlib.
- Để xuất Excel: `pandas` với engine phù hợp.

---

*Tài liệu được tạo để đối chiếu trực tiếp với `Main_Simulation.ipynb`; nếu notebook được chỉnh sửa, nên cập nhật lại các chỉ số `VN_Main_IP` và tên biến trong Cell 20 cho khớp.*
