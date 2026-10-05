# UR3 LLM Task Planner

Hệ thống điều khiển robot UR3/UR3e bằng câu lệnh ngôn ngữ tự nhiên. Câu lệnh được gửi tới LLM thông qua **9Router**, chuyển thành JSON Plan, kiểm tra bằng Plan Validator rồi thực thi các skill `pick`, `place` và `home` qua MoveIt 2 trong Gazebo.

## Thông tin sinh viên

- Họ và tên: **Lê Hồng Quang**
- Mã sinh viên: **23020757**
- Hai chữ số cuối: `57`
- `P = 57 mod 6 = 3`

Ánh xạ nhiệm vụ cá nhân:

| Vùng | Vật thể |
|---|---|
| Zone A | Yellow cube |
| Zone B | Blue cube |
| Zone C | Red cube |

## 1. Chuẩn bị 9Router

Khởi động 9Router:

```bash
9router
```

Mở dashboard:

```text
http://localhost:20128/dashboard
```

Kết nối provider **OpenCode Free** và sử dụng model:

```text
oc/muse-spark-1.2-contributor-free
```

Tạo API key trong dashboard của 9Router. Không đưa API key lên GitHub.

## 2. Cấu hình `.env`

Di chuyển vào workspace:

```bash
cd ~/ur_ws
```

Mở file cấu hình:

```bash
nano .env
```

Giữ nguyên tên các biến đang có trong project và cấu hình các giá trị tương ứng:

```dotenv
OPENAI_API_KEY=API_KEY_CUA_9ROUTER
OPENAI_BASE_URL=http://localhost:20128/v1
OPENAI_MODEL=oc/muse-spark-1.2-contributor-free
```

Nếu `.env.example` sử dụng tên biến khác, dùng đúng tên biến trong file đó. Thay `API_KEY_CUA_9ROUTER` bằng key thật.

Đảm bảo `.env` không được commit:

```bash
grep -qxF '.env' .gitignore || echo '.env' >> .gitignore
```

## 3. Build workspace

Mở terminal mới:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_llm_control --symlink-install
source install/setup.bash
```

Khi chỉ sửa file Python trong package, vẫn nên build và source lại trước khi chạy.

## 4. Chạy mô phỏng và MoveIt 2

### Terminal 1 — 9Router

```bash
9router
```

Giữ terminal này chạy.

### Terminal 2 — Gazebo, robot và MoveIt

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ur3_llm_control llm_robot.launch.py
```

Chờ Gazebo, robot, controller và MoveIt khởi động hoàn toàn.

### Terminal 3 — Kiểm tra TF

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run tf2_ros tf2_echo base_link tool0
```

Kết quả đúng phải liên tục hiển thị `Translation` và `Rotation`. Nhấn `Ctrl+C` sau khi kiểm tra.

Nếu báo hai cây TF không kết nối, chưa chạy Skill Executor. Kiểm tra lại launch, `robot_state_publisher`, `/joint_states` và controllers.

### Terminal 3 — Chạy Skill Executor

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

set -a
source .env
set +a

ros2 run ur3_llm_control skill_executor
```

Thông tin đúng khi khởi động:

```text
STUDENT NAME: Lê Hồng Quang
STUDENT ID: 23020757
PERSONALIZED TASK: P=3; A=Yellow, B=Blue, C=Red
9ROUTER BASE URL: http://localhost:20128/v1
9ROUTER MODEL: oc/muse-spark-1.2-contributor-free
```

OpenCode Free có danh sách model động nên chương trình có thể hiển thị cảnh báo model không nằm trong `GET /models`. Chương trình vẫn có thể tiếp tục nếu request thực tế tới model thành công.

## 5. Kiểm tra kết nối 9Router riêng

Sau khi nạp `.env`, kiểm tra endpoint:

```bash
curl -sS --connect-timeout 5 --max-time 90 \
  http://localhost:20128/v1/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"oc/muse-spark-1.2-contributor-free","messages":[{"role":"user","content":"Reply with only: CONNECTION OK"}],"stream":false}'
```

Kết quả mong đợi:

```text
CONNECTION OK
```

## 6. Chạy mức cơ bản

Reset mô phỏng về trạng thái ban đầu trước mỗi bài thử độc lập.

### Kiểm thử khối vàng

```text
Đưa vật màu đỏ  vào vùng A.

```

### Kiểm thử khối xanh dương

```text
Đưa vật màu xanh dương vào vùng B.
```

Kế hoạch mong đợi:

```text
pick(blue_cube)
place(blue_cube, zone_b)
home()
```

## 7. Chạy mức nâng cao

Reset Gazebo để robot và các vật trở về vị trí ban đầu, sau đó nhập:

```text
Sắp xếp tất cả vật theo mã sinh viên của tôi.
```

Kế hoạch đúng với MSSV `23020757`:

```text
pick(yellow_cube)
place(yellow_cube, zone_a)
pick(blue_cube)
place(blue_cube, zone_b)
pick(red_cube)
place(red_cube, zone_c)
home()
```

Kết quả thành công:

```text
[1/7] pick(yellow_cube) ........ SUCCESS
[2/7] place(yellow_cube, zone_a) SUCCESS
[3/7] pick(blue_cube) .......... SUCCESS
[4/7] place(blue_cube, zone_b) . SUCCESS
[5/7] pick(red_cube) ........... SUCCESS
[6/7] place(red_cube, zone_c) .. SUCCESS
[7/7] home() ................... SUCCESS

TASK SUCCESS
```

## 10. Lỗi thường gặp

### `Invalid API key` hoặc HTTP 401

- Kiểm tra `OPENAI_API_KEY` trong `.env`.
- Sử dụng key được tạo bởi 9Router, không dùng trực tiếp key Gemini.
- Nạp lại file bằng `set -a; source .env; set +a`.

### HTTP 429

Provider hoặc model đã hết quota. Chuyển sang model/provider còn hạn mức hoặc chờ quota được làm mới.

### HTTP 503 hoặc timeout

- Kiểm tra 9Router vẫn chạy.
- Thử request `curl` ở mục 5.
- Kiểm tra trạng thái provider trên dashboard.

### `FRAME_TRANSFORM_FAILURE`

Kiểm tra:

```bash
ros2 run tf2_ros tf2_echo base_link tool0
ros2 topic echo /joint_states --once
ros2 control list_controllers
```

Chỉ chạy robot khi transform `base_link -> tool0` hoạt động.

### `PLANNING_FAILED`

- Không tiếp tục gửi lệnh mới nếu robot vẫn đang giữ vật.
- Thoát executor và reset mô phỏng.
- Kiểm tra collision, IK, planning scene và pose của vật/vùng đích.

### Model OpenCode không xuất hiện trong `/models`

Đây có thể là hành vi bình thường do catalog động. Việc thực thi vẫn phải dừng nếu request thực tế lỗi, JSON không hợp lệ hoặc Plan Validator từ chối kế hoạch.

## 11. Luồng hoạt động

```text
User Command
    ↓
LLM qua 9Router
    ↓
JSON Plan
    ↓
Plan Validator
    ↓
Skill Executor
    ↓
MoveIt 2 + Gazebo attach/detach
    ↓
UR3/UR3e
```
