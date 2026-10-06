# Bài thực hành 03 – LLM Skill Planning với Gripper và Camera

**Sinh viên:** Lê Hồng Quang  
**MSSV:** 23020757

## Nội dung demo

Ban đầu, `blue_cube` nằm trong `zone_b`.

Người dùng yêu cầu: **“Đưa vật màu đỏ vào vùng B.”**

Hệ thống sử dụng camera để phát hiện vùng B bị chiếm,
chuyển vật xanh dương sang vị trí tạm, sau đó đặt
vật đỏ vào vùng B và đưa robot về home.

## Môi trường

- ROS 2 Jazzy
- Gazebo và MoveIt 2
- Robot UR3/UR3e, gripper và camera RGB-D
- 9Router: `http://localhost:20128/v1`
- Model: `oc/muse-spark-1.2-contributor-free`

## 1. Build package

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
```

## 2. Cấu hình API key

Khởi động 9Router và kết nối provider đang sử dụng.

Tạo file `.env` trong `~/ur_ws`:

```bash
OPENAI_API_KEY='YOUR_9ROUTER_API_KEY'
```

Thay giá trị mẫu bằng API key của 9Router.
Không đưa `.env` lên GitHub.

## 3. Terminal 1 – Khởi chạy mô phỏng

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch ur3_vision_planning llm_robot.launch.py \
  scenario:=zone_b_occupied launch_rviz:=true
```

Chờ Gazebo, MoveIt và camera khởi động.

## 4. Terminal 2 – Chạy điều khiển bằng ngôn ngữ tự nhiên

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

set -a
source .env
set +a

ros2 run ur3_vision_planning natural_language_planner \
  --ros-args \
  -p use_sim_time:=true \
  -p dry_run:=false \
  -p execute_robot:=true \
  -p precheck_only:=false
```

Khi xuất hiện `ENTER COMMAND`, nhập:

```text
Đưa vật màu đỏ vào vùng B.
```

## 5. Kế hoạch thực thi

```text
pick(blue_cube)
place(blue_cube, temporary_position)
pick(red_cube)
place(red_cube, zone_b)
home()
```

Vị trí tạm được lựa chọn và kiểm tra trước khi thực thi.

Kết quả hoàn thành:

- Vật xanh dương nằm tại vị trí tạm.
- Vật đỏ nằm trong vùng B.
- Robot trở về home.
- Terminal hiển thị `TASK SUCCESS`.

Nếu một bước thất bại, chương trình dừng và hiển thị lỗi.

## 6. Xem ảnh nhận diện camera (tùy chọn)

Mở terminal khác:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

## Mã nguồn và video demo

- [GitHub](https://github.com/hongquang100805-lab/ur3_vision)
- [Video demo](https://drive.google.com/drive/folders/1dvE37pidJxwDKPa43pn44Sc7h1g5m9Zx)
