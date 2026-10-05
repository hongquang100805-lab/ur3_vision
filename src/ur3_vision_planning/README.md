# ur3_vision_planning

## Gazebo attachment acknowledgement (Gazebo 8)

The attach subscriber uses `google.protobuf.Message` (generic ProtoMsg),
whereas detach uses `gz.msgs.Empty`. Discovery checks only subscriber rows
on the requested action's topic; a publisher of Empty is not evidence.
Dynamic plugin loading emits `attached`. RobotSkills retains that explicit
acknowledgement in this process and, with a live compatible subscriber,
does not republish an identical attach and wait for a nonexistent transition.
Only a matching previously observed acknowledgement can take this path;
subscriber presence alone never proves attach/detach. Changing state needs
a fresh acknowledgement. Timeout/missing subscriber invalidates the cached
state and stops before lift/release. Post-lift object-follow verification
is unchanged. The cache is not persisted or inferred across node restarts.

If execution stopped with `held_object=blue_cube`, do not rerun a new plan
against the partially grasped scene. Stop the failed executor and restart
the simulator/scenario and perception to restore the initial arrangement
before testing. This patch never automatically opens fingers, detaches,
homes or moves the failed grasp. Build/tests and topic discovery can be
checked without sending any Gazebo command or robot trajectory.

## Seed continuity and deterministic precheck comparison

`VisionRobotSkills` (adapter của RobotSkills trong execution pipeline Bài 03)
giữ seed actual → pre-grasp → Cartesian grasp/lift endpoints → pre/final-place
→ camera-retreat IK endpoint → HOME. Không giải IK lại để tạo home start nếu
đã có endpoint. Tất cả IK đều mang full seed joints, không có zero/empty seed.
Sau IK, xét `q + 2*pi*k` theo URDF hard/soft bounds và MoveIt position overrides;
chọn winding gần seed nhất có FK/whole-robot state validity PASS. Không wrap
prismatic. Giữ auxiliary gripper joints từ seed, attachment/ACM từ scene.
Log raw/normalized/seed, physical/angular deltas và wrap multiples. Angular
delta luôn <=pi theo định nghĩa nên kiểm tra cả **unwrapped command delta**;
không dùng modulo để giấu jump 2*pi.

HOME thử canonical trước. Chỉ khi planning failure/timeout mới xét các goal
equivalent riêng biệt, xếp theo khoảng cách tới seed, trong limits và cách
biên tối thiểu 0.05 rad, FK tương đương canonical, collision-valid. Không
retry cùng một HOME request ngẫu nhiên tới khi PASS; 99999 luôn là FAIL.

Execution pose/Cartesian/home được override trong adapter: plan-only trước,
validate trajectory rồi mới gửi ExecuteTrajectory. Guard kiểm tra đúng sáu
arm names, finite positions/derivatives, tăng thời gian, bounds, raw first
point khớp **fresh actual joint sample**, không jump giữa hai points, endpoint
FK/joint target và collision. Không sửa trajectory lỗi rồi tự execute. Mọi
fallback pose dùng cùng guard; action submission timeout không được xem là
planning failure để launch một action thứ hai. Legacy Bài 02 không thay đổi.
Entrypoint `skill_executor` của Bài 03 cũng khởi tạo adapter này; LLM/validator
của entrypoint không đổi. `natural_language_planner` vẫn có double confirmation
và precheck-only mặc định; `skill_executor` là entrypoint interactive legacy,
không phải lệnh dùng cho repeat test camera/precheck-only.

Chạy 3 lượt thực, không LLM, không restart Gazebo, không chuyển động:

```bash
ros2 run ur3_vision_planning motion_precheck --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=true -p object_name:=red_cube -p target_zone:=zone_b \
  -p repeat_runs:=3 -p repeat_expected_temporary:=temporary_8
```

Node in `DETERMINISTIC PRECHECK`, từng retreat vector/HOME code/point count,
và maximum **raw** branch variation. Chỉ PASS repeat nếu cả ba lượt có đúng
slot, fraction=1.0, HOME code=1/points>1/error<=0.02, restore/resync PASS,
variation<=0.02 rad. Camera geometry phải không đổi giữa lượt. Không chuyển
động để xác nhận vật đã đặt: tất cả attachment/detachment ở đây chỉ virtual
MoveIt. Solver/OMPL vẫn có thể fail; không che lỗi hoặc khẳng định repeat
PASS từ unit/service fixtures khi chưa chạy simulator thật.

Verified simulator precheck 2026-10-03, `red_cube -> zone_b`, blue occupying
zone_b: three consecutive runs without restart/motion selected temporary_8
at exactly (0.28,-0.27,0.22). All four Cartesian grasp/lift fractions per run
were 1.000, scene restore/resync PASS, canonical HOME code=1/points=64.
Final joint errors: run1=0.007441980117, run2=0.009833888258,
run3=0.009551708009 rad. Retreat vectors (pan,lift,elbow,w1,w2,w3), rounded:

```text
run1 (-3.548831501,-1.551173406,-0.896428080,-2.264787495,1.570796326,-1.978035174)
run2 (-3.548831501,-1.551173406,-0.896428080,-2.264787495,1.570796326,-1.978035174)
run3 (-3.548831501,-1.551173406,-0.896428080,-2.264787495,1.570796326,-1.978035174)
maximum raw branch variation = 2.2648549702353193e-14 rad
```

This verifies planning only in this scene, not real grasp/release execution
with the new trajectory guard. Execution motion was deliberately NOT tested.

## HOME feasibility — simulated retreat, planning-only

HOME precheck dùng `GetMotionPlan /plan_kinematic_path`, không dùng MoveGroup
execution/ExecuteTrajectory. Start được lưu riêng từ **nghiệm IK cuối của
camera retreat sau place zone_b**; `start_state.is_diff=false`, không thay
bằng joint state robot thật còn ở HOME ban đầu. Log `action goal status: N/A`
vì dùng service, kèm planning group/time/attempts, start/goal joints,
MoveItErrorCodes.val/name/message/source và timeout/planning failure.
`99999` theo enum Jazzy đang cài là `FAILURE` chung, không phải SUCCESS hay
undefined; log cũ không lưu action status/message/source nên không xác định
được nguyên nhân chi tiết từ chỉ số đó.

Model/limits lấy từ runtime, không yêu cầu URDF luôn là parameter của MoveIt.
Launch UR hiện tại cấp URDF từ robot_state_publisher qua `/robot_description`;
MoveIt's RDFLoader nhận topic này khi không có parameter. HOME dùng
`/move_group/list_parameters` để chỉ đọc parameter đã khai báo qua
`/move_group/get_parameter_types` để phân biệt **declared** và **initialized**.
MoveIt có thể khai báo typed min/max mà chưa gán giá trị; Jazzy rclcpp trả cả
`GetParameters.values=[]` khi batch chứa một typed parameter chưa khởi tạo.
HOME chỉ đọc từng initialized parameter bằng `/move_group/get_parameters`,
log tên parameter UNINITIALIZED và không để nó làm mất response của SRDF.
Response rỗng cho initialized parameter vẫn FAIL, trừ khi RPC kiểm tra lại
chứng minh value đã chuyển về NOT_SET. Không bỏ qua override có giá trị.
Với URDF/SRDF thiếu, chưa khởi tạo hoặc rỗng, subscribe `/robot_description` hoặc
`/robot_description_semantic` bằng `std_msgs/String`, QoS reliable/transient-local.
Log nguồn parameter/topic cho từng description. Không có description runtime
hợp lệ trong timeout thì FAIL, không dựng URDF/SRDF giả hoặc bỏ kiểm tra limits.
Position-limit overrides lấy từ parameter MoveIt nếu khai báo; các override
không khai báo/chưa khởi tạo dùng bounds trong URDF live. Min/max được áp dụng
nếu có giá trị, kể cả khi has_position_limits chưa được gán.
Group phải có đúng 6 arm joints, goal constraints đúng canonical order.
Giữ winding của bounded revolute joints, chỉ normalize joints continuous;
ngoài limits/missing joint/nonfinite đều FAIL. Auxiliary seed joints như
hai ngón mở chỉ dùng cho collision geometry, **không** thêm vào arm group,
goal constraints hoặc trajectory. Không đoán UR3/UR3e limits từ file khác.

Trước planning, kiểm tra state validity toàn robot cho simulated retreat và
home goal, in contacts nếu có. `ALREADY_AT_HOME` chỉ PASS sau khi cả hai state
valid và mọi joint error <= 0.01 rad. Planning mặc định 15 s, 3 attempts trong
`execution.yaml/home_precheck`; không chọn intermediate chưa kiểm chứng.
Response phải SUCCESS, group/trajectory đúng 6 arm joints, có đường nối,
trajectory_start và point đầu khớp retreat (0.0001 rad), mọi waypoint trong
joint limits, endpoint khớp home <= 0.01 rad và collision-valid. Mọi code
khác SUCCESS, timeout hoặc response sai đều dừng; rollback/resync không đổi.

Kiểm tra không gọi LLM, không chuyển động (giữ simulator hiện tại đang chạy):

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
ros2 run ur3_vision_planning motion_precheck --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=true -p object_name:=red_cube -p target_zone:=zone_b
```

## Chẩn đoán red lowering không chuyển động, không dùng quota LLM

```bash
ros2 run ur3_vision_planning motion_precheck --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=true -p object_name:=red_cube -p target_zone:=zone_b
```

`motion_precheck` nhận structured goal bằng parameter, resolve/validate plan
thật từ camera, không gọi LLM và không chấp nhận `precheck_only=false`.
Mặc định vẫn dry-run, không khởi tạo RobotSkills nếu thiếu xác nhận.
Node này có thể chạy với simulator hiện tại để lấy contacts trước restart.

Mỗi Cartesian lowering log state validity toàn robot tại các waypoint cách
nhau tối đa 2 mm: index, z, FK, VALID/INVALID, từng contact pair, robot link,
collision object và contact depth. Geometric IK (`avoid_collisions=false`)
**chỉ** lấy joint state để chẩn đoán, sau đó luôn gọi GetStateValidity với
collision checking/ACM hiện hành. Không dùng nghiệm này làm quỹ đạo hoặc
coi nó là motion PASS. Branch IK chẩn đoán có thể khác Cartesian solver;
joint_step_max được log, và khi không có nghiệm thì state là UNKNOWN,
không tự kết luận collision. Fraction thiếu vẫn thất bại, dù diagnostic IK PASS.
ACM chỉ cho phép target/touch_links lúc approach/lowering; sau attach reset
ACM, chỉ attachment touch_links còn được phép contact. Restore được kiểm tra
trên world/attachments/ACM, sau đó resync từ camera cả khi precheck FAIL.

Temporary giữ **đúng** tọa độ candidate, không áp dụng offset zone. Log cũ
`selected y_offset=-0.040` là chênh lệch giữa hai slot, không phải dịch thêm
khi execute. Log mới là `EXACT cube-centre coordinates; NO ZONE OFFSET`.
Ngay sau selection, validate và in `FINAL EXECUTION PLAN`; nếu grasp tiếp theo
FAIL thì đây vẫn chỉ là plan đã validate, chưa phải motion feasibility PASS.
Trusted slot ledger được rollback khi precheck thất bại.

Layout **spawn ban đầu** mới cho `zone_b_occupied` (default không đổi):

| Object | Cube centre (m) |
|---|---|
| red_cube | (0.38, -0.15, 0.22) |
| blue_cube | (0.28, 0.00, 0.22) |
| yellow_cube | (0.40, -0.03, 0.22) |
| purple_cube | (0.38, 0.09, 0.22) |
| green_cube | (0.35, 0.21, 0.22) |

Source geometry test: layout cũ red/green cách 75 mm; tại tool pose
(0.380,-0.148,0.380), orientation (1,0,0,0), ngón trái mở từ URDF có box
y=[-0.104,-0.092], z=[0.230,0.330], giao green cube cũ tại
y=[-0.095,-0.055], z=[0.200,0.240]. Đây là **đối chiếu geometry nguồn**,
không thay cho contact response thật từ MoveIt. Layout mới có khoảng cách
tâm tối thiểu 120 mm, kiểm tra footprint/camera và không giao các ngón mở
với cube khác. Reachability IK thực tế vẫn phải qua precheck.
Tool offset 0.16 m/finger reach 0.15 m không đổi; ngón cách bàn 30 mm ở
grasp danh định. Không nâng pose để che collision, không teleport vật.

**Restart Gazebo** để nhận layout mới. Build/source rồi launch lại:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied
```

## Thực thi expanded plan bằng camera + RobotSkills

Mặc định `dry_run=true`, `execute_robot=false`, `precheck_only=true`:
không tạo RobotSkills/MoveIt/gripper clients.
Chỉ khi **cả** `dry_run=false` và `execute_robot=true` mới vào executor.
Thiếu cờ xác nhận bị từ chối trước khi gọi LLM/khởi tạo motion clients.
`execute_robot=true` riêng lẻ trong dry-run cũng không khởi tạo precheck.
Trong executor, `precheck_only=true` chỉ kiểm tra, không chuyển động;
phải thêm `precheck_only=false` mới thực thi sau khi toàn bộ precheck PASS.

`plan_execution.py` lazy-create `VisionRobotSkills` (kế thừa RobotSkills),
kiểm tra lại plan, chờ servers/TF và thực hiện feasibility precheck trước
lệnh chuyển động đầu tiên. Vị trí pick lấy lại từ camera trước **mỗi** pick,
không lấy spawn YAML hoặc Gazebo pose làm pick target. Gazebo pose chỉ được
cơ chế skill cũ dùng để kiểm chứng contact/grasp thật và đồng bộ detach.

Precheck sync world collision boxes bằng camera, mô phỏng attach/detach
và chuyển cube trong MoveIt scene để kiểm tra đúng tình huống tương lai
(Zone B trống sau khi blue được đặt vào temporary). Kiểm tra collision-aware
IK pre-grasp, Cartesian lowering/lift từ đúng seed pre-grasp, IK pre-place/final-place, camera
retreat và `home` bằng **GetMotionPlan planning-only**. Chỉ cho phép contact
giữa target và touch links gripper, không tắt collision bàn/robot.
Cartesian dùng cùng orientation xuống, `max_step=0.005`, collision checking bật,
fraction phải **1.0** và FK start/end đúng pose; không gọi IK grasp độc lập.
Seed gripper ảo lấy từ `execution.yaml`, khớp URDF/controller: mở trái=0,
phải=0.043; giữ cube 40 mm danh định trái=0.024, phải=0.019. Đây không phải
lệnh gripper hay bằng chứng grasp vật lý; execution vẫn kiểm chứng grasp thật.
Các diff mô phỏng được rollback cả world objects, attached objects và ACM;
không gửi joint state mô phỏng cho robot. Đọc lại scene để kiểm tra world
geometry/poses, không còn attachment và ACM đã khôi phục, rồi mới in
`PLANNING SCENE RESTORED`. Pose nào fail được log rõ. Bất kỳ
pose bắt buộc không có candidate hợp lệ hoặc rollback lỗi đều chặn thực thi.

Temporary search hai tầng: Environment Manager lọc **toàn bộ** candidate trong
`temporary_positions.yaml` theo camera/object/zone/table/frustum. RobotSkills
thử từng candidate còn lại với cube đang attach **chỉ trong MoveIt**, yêu cầu
cả pre-place và final-place IK PASS. Log tất cả candidate đã thử và lý do loại.
Điểm mới được cập nhật vào destination, `temporary_positions` và trusted ledger,
rồi validate lại toàn bộ expanded plan. Không có candidate reachable thì dừng;
không tự coi `NO_IK_SOLUTION` là PASS hay dùng một vị trí dự phòng chưa kiểm tra.
Trước execution đồng bộ lại toàn bộ cube từ snapshot camera mới và kiểm tra drift.

Z camera/temporary là tâm cube. Zone marker Z không phải tâm cube: manager
resolve tâm cube tại table top + cube_height/2. Tool0 = cube centre + offset
0.16 m; final release thêm 5 mm, pre-place thêm 15 mm theo `execution.yaml`.
Độ cao pre-place mặc định khoảng 0.400 m thay vì 0.436 m từng không có IK.
Zone có thể chọn offset Y nhỏ **trong** footprint zone nếu tâm không có IK;
chỉ thực thi đúng candidate đã đạt cả pre-place và final-place precheck.

Executor lặp các step đã validate, không hard-code màu. Place tái sử dụng
Cartesian descent/retreat, pose fallback có kiểm tra TF, gripper và Gazebo/
MoveIt attach/detach. Fraction < 1.0 không bao giờ được thực thi. Trước mở
gripper phải xác nhận TF đến đúng final pose. Sau release, retreat khỏi ROI
(pose cấu hình cũng phải PASS precheck), chờ ít nhất **3 camera measurements
khác timestamp** sau retreat. Temporary phải gần điểm chọn và zone nguồn
phải trống; zone destination phải khớp cả object.zone và zone.object.
Chỉ sau verification mới chạy pick tiếp theo hoặc home, rồi TASK SUCCESS.

Lỗi bất kỳ bước nào dừng toàn bộ plan, không tự nhả/teleport/home hoặc chạy
step tiếp theo. Executor giữ attachment nếu lift/place lỗi trước release,
cố gắng cancel action còn chạy và gửi vận tốc gripper bằng 0. Nếu stop không
được xác nhận sẽ báo `STOP_NOT_CONFIRMED`. Khôi phục simulator/robot và vật
đang cầm thủ công trước khi retry; precheck cũng từ chối scene còn attached.

Build và scene — terminal 1:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied
```

Terminal 2 — 9Router và camera phải sẵn sàng. Kiểm tra không chuyển động trước:

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
  -p precheck_only:=true
```

Nhập `Đưa vật màu đỏ vào vùng B.`. Nếu mọi kiểm tra đạt, terminal in
`MOTION FEASIBILITY PRECHECK: PASS` và `PRECHECK ONLY — ROBOT MOTION NOT STARTED`.
Không có `TASK SUCCESS` hoặc lệnh gripper/trajectory trong chế độ này, cả khi lỗi.

Chỉ khi đã kiểm tra vùng chuyển động an toàn, chạy thực thi thật trong simulator:

```bash
ros2 run ur3_vision_planning natural_language_planner \
  --ros-args -p use_sim_time:=true \
  -p dry_run:=false -p execute_robot:=true -p precheck_only:=false
```

Nhập `Đưa vật màu đỏ vào vùng B.` hoặc thêm parameter `command` để chạy một
lần. `execution_config` có thể chỉ định file khác bằng ROS parameter.
Đây là thực thi chuyển động trong Gazebo/MoveIt hiện tại, **không** bypass
simulator guard để điều khiển UR phần cứng. Package Bài 02 không sửa.

Theo yêu cầu, lượt implement này chỉ build/compile/unit tests, không tự
launch simulator, gọi motion node hoặc thực hiện precheck MoveIt thật.
Unit tests không tạo ROS nodes/servers thật. Test service fixtures kiểm tra
candidate đầu NO_IK rồi chọn candidate tiếp, revalidation, scene rollback khi
thành công/lỗi, fraction thiếu và cổng precheck-only không gửi actuator command.
Test fixtures không phải kết quả IK/Cartesian trong simulator thật và không được
dùng thay response MoveIt trong node. Candidate reachable thực tế chỉ xác định
khi chạy precheck-only với scene/camera/MoveIt đang hoạt động.
IK endpoint của place chưa chứng minh toàn bộ đường chuyển vật tới đó;
runtime vẫn lập kế hoạch collision-aware từng đường và từ chối đường không hoàn chỉnh.
Camera clearance pose `(0.18,-0.20,0.45)` chưa kiểm thử live; runtime phải
PASS IK và camera phải thật sự thấy đủ vật, nếu không plan dừng an toàn.

## Ngôn ngữ tự nhiên → 9Router → camera → dry-run

Executable mới `natural_language_planner` giữ độc lập với `skill_executor`:

```text
User command → 9Router LLM → validated {object, target_zone}
             → fresh /vision/environment_state → Environment Manager
             → expanded JSON plan → goal-aware Plan Validator → dry-run output
```

LLM chỉ phân tích một goal, hiểu tiếng Việt/Anh và biết đủ năm cube/ba zone.
Schema chỉ gồm `object`, `target_zone`; Markdown, field lạ, coordinate, skill,
trajectory, temporary position và tên ngoài allowlist đều bị từ chối.
Tọa độ tạm chỉ do manager chọn từ YAML dựa trên vị trí camera. Validator
kiểm tra lại clearance của slot, vật đang giữ, destination, thứ tự đưa
occupant ra trước và mục tiêu cuối cùng đúng goal; không cho thêm thao tác
ngoài kế hoạch cần thiết. Không thay đổi executor/validator cũ của Bài 02.

HTTP transport dùng lại `llm_planner.py`, `/chat/completions`, `stream=false`
và JSON mode. Model precedence: ROS parameter `llm_model` → `OPENAI_MODEL`
→ biến cũ `LLM_MODEL`. Pipeline mới không tự chọn model khác nếu thiếu cấu
hình; API key chỉ đọc biến môi trường `OPENAI_API_KEY`. Base URL từ
`OPENAI_BASE_URL` hoặc parameter `openai_base_url`. Cấu hình hiện tại là
`http://localhost:20128/v1`, `oc/muse-spark-1.2-contributor-free`.
`.env` hiện có `LLM_MODEL` vẫn chạy được; có thể đổi tên biến đó thành
`OPENAI_MODEL`. Không cần đổi key và không ghi key vào YAML/source/log/Git.

Mỗi lần khởi động chỉ GET `/models` như kiểm tra advisory; model vắng mặt
hoặc GET lỗi chỉ warning. Không POST connectivity/diagnostic. Mỗi command
gửi một POST thật, không retry tự động kể cả JSON sai, không mock/fallback.
API lỗi, goal sai, camera stale hoặc plan sai đều dừng dry-run với lý do.
Trong khi chờ API, ROS vẫn xử lý camera; sau validate goal bắt buộc nhận
một environment message mới và kiểm tra cả ROS/steady-clock freshness.

Terminal 1 — build và scene (chỉ chạy một simulator trong domain này):

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied
```

Terminal 2 — đợi camera status `ok`, nạp cấu hình hiện có và chạy:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -a
source .env
set +a
ros2 run ur3_vision_planning natural_language_planner --ros-args \
  -p use_sim_time:=true -p dry_run:=true \
  -p command:='Đưa vật màu đỏ vào vùng B.'
```

Bỏ `command` để nhập nhiều câu tương tác, dùng `quit` để thoát. Ví dụ câu
tiếng Anh: `Put the red cube in Zone B.` Không dò toàn bộ câu bằng hard-code.
`use_sim_time` và `dry_run` mặc định true; explicit ROS override của
`use_sim_time` vẫn được tôn trọng. Thực thi yêu cầu hai cờ xác nhận như mục
đầu README; chỉ `dry_run=false` mà thiếu `execute_robot=true` vẫn bị từ chối.

Đã kiểm thử thật với camera và 9Router: LLM trả
`{"object":"red_cube","target_zone":"zone_b"}`, nhận blue chiếm Zone B,
chọn `temporary_1=(0.43,-0.23,0.22)` và PLAN VALID:
pick blue → place temporary_1 → pick red → place zone_b → home.
Trong phép thử dry-run ở bước trước không có lệnh robot, node
không có motion publisher/client. Quan sát trong lần gọi thật: 0 arm
trajectory, 0 gripper command, 0 MoveIt goal status; các khớp không đổi.
Unit tests kiểm tra goal/plan/HTTP lỗi;
transport double chỉ tồn tại trong unit tests, không có trong runtime.
Các executable camera/environment_manager_test/scene_publisher/skill_executor
cũ được giữ; package.xml đã có đủ dependencies nên không thêm dependency mới.
Chỉ hỗ trợ một goal object→zone mỗi command, chưa lệnh nhiều goal trong một câu.

## Environment Manager — chỉ lập kế hoạch dry-run

`environment_manager.py` subscribe `/vision/environment_state`; vị trí và
occupancy chỉ lấy từ camera. Không gọi LLM, dịch vụ pose Gazebo hay robot skills.
Node từ chối JSON sai/duplicate key/NaN, dữ liệu chưa nhận, status không `ok`,
frame sai, object mất/stale, zone không hợp lệ, nhiều occupant hoặc occupancy
không nhất quán. Để chọn slot an toàn cần thấy đủ năm cube, không coi vật bị
mất dấu là vị trí trống. Timeout kiểm tra cả tuổi dữ liệu ROS và steady clock;
latest message sai luôn thay thế cache, không tái sử dụng snapshot tốt trước đó.

`config/temporary_positions.yaml` khai báo ba candidate (tọa độ **tâm cube**,
không phải tool0), timeout và clearance. Duyệt theo thứ tự YAML, loại slot nếu
cube vượt biên bàn, sai độ cao, ngoài vùng nhìn camera/vùng làm việc cấu hình,
gần base, gần marker zone hoặc cách bất kỳ cube camera thấy dưới 0.08 m.
Slot đầu tiên đạt được chọn; hết slot báo `NO_FREE_TEMPORARY_POSITION`.
Các candidate là phương án thay thế, không phải ba chỗ có thể dùng đồng thời;
vật đặt vào slot cũng phải được camera phát hiện trước lần lập kế hoạch sau.

Goal đã thỏa → chỉ `home`; zone trống → pick/place/home; zone có vật khác →
chuyển occupant vào slot, rồi chuyển target vào zone, cuối cùng home.
`DryRunPlanValidator` tái sử dụng whitelist skill/zone hiện có, bổ sung đủ năm
object và kiểm tra schema `destination`, trạng thái vật đang giữ, occupancy và
đúng thứ tự. Tọa độ slot phải khớp chính xác ledger vừa được manager chọn;
không chấp nhận tọa độ tự thêm/sửa bởi người dùng hoặc LLM. Validator dry-run
độc lập; không sửa validator hay executor đang chạy của Bài 02/Bài 03.

Build và launch trong terminal 1 (không chạy đồng thời simulator khác cùng domain):

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied
```

Đợi camera có `status="ok"`, chạy terminal 2:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run ur3_vision_planning environment_manager_test --ros-args \
  -p use_sim_time:=true -p object_name:=red_cube -p target_zone:=zone_b
```

Node in `DRY-RUN PLAN VALID`, JSON và thoát, không gửi lệnh chuyển động.
Nếu dùng ROS_DOMAIN_ID tùy chọn thì cả hai terminal phải cùng giá trị.
`temporary_config`, `scene_file`, `camera_file` cũng có thể đổi bằng ROS parameter.

Đã thử bằng camera simulator trong scenario `zone_b_occupied`: nhận occupant
`blue_cube`, chọn `temporary_1={frame_id:base_link,x:0.43,y:-0.23,z:0.22}`,
plan pick blue → place temporary_1 → pick red → place zone_b → home.
Build symlink-install thành công; 72 unit tests đạt. Hai nhánh zone trống và
task already satisfied cũng đã thử trực tiếp; pause simulator khiến node
từ chối `status=stale`, không tái sử dụng vị trí cũ để lập kế hoạch.
Kiểm tra collision-aware IK tại các slot với tool0 hướng xuống và Z=0.38,
0.386, 0.40 m đạt SUCCESS. Đây chỉ là nghiệm IK từng điểm trong scene ban đầu,
không phải chứng minh quỹ đạo/attached-object collision-free. Một số điểm
tiếp cận cao hơn Z=0.436 m không có nghiệm trong phép thử; trước tích hợp thực
thi cần chọn độ cao tiếp cận phù hợp và kiểm tra toàn bộ đường đi, trạng thái
attached cube cùng dữ liệu camera mới. Chưa cho robot chạy tới temporary slot.

## Scenario spawn để kiểm thử occupancy

Mặc định `scenario:=default` giữ nguyên bố trí năm cube ngoài zone.
Để spawn blue_cube ngay tại tâm Zone B, chạy:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied
```

Scenario trong `config/scene.yaml` chỉ thay blue_cube thành `[0.28, 0.0, 0.22]`;
bốn cube còn lại giữ nguyên. Launch và MoveIt scene publisher dùng chung
`scene_config.load_scene` để chọn bố trí ban đầu. Không có lệnh thay pose sau
spawn. Camera perception không đọc scenario để quyết định occupancy.

Trong terminal đã source ROS/workspace:

```bash
ros2 topic echo /vision/environment_state --once
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

Đã xác minh bằng camera thật sau launch: `blue_cube.zone="zone_b"`,
`zones.zone_b={"occupied":true,"object":"blue_cube","objects":["blue_cube"],
"stale":false,"status":"occupied"}`; Zone A/C empty. Tọa độ tâm cube từ camera
xấp xỉ `[0.28149, 0.0, 0.22000]`. Build thành công, 39 tests passed.

ROS 2 Jazzy package implementing the strict LLM-plan validator, MoveIt skill
executor, Gazebo scene, and simulator grasp synchronization for student
23020757 (P=3: A=Yellow, B=Blue, C=Red).

## Bài 03 — bước 2: scene và RGB-D camera

Chỉ package này bổ sung camera, green_cube và purple_cube. Package Bài 02
`ur3_llm_control` độc lập, giữ nguyên. LLM planner, validator và robot skills
vẫn là bản sao Bài 02; hai cube mới hiện chỉ thuộc scene, chưa mở rộng
danh sách object cho LLM hoặc viết perception node.

`config/scene.yaml` chứa kích thước bàn/cube, khối lượng, màu và spawn pose
của năm cube và ba zone. Green/purple nằm tại `(0.38, -0.075, 0.22)` và
`(0.38, 0.075, 0.22)`, cách tâm cube lân cận 75 mm, trong mặt bàn.
`scene_publisher` thêm cả năm cube vào world collision objects của MoveIt.

`config/camera.yaml` chứa pose, frame, topics, FOV, resolution và clipping.
Gazebo Sim 8.11/Harmonic dùng sensor `rgbd_camera` với plugin
`gz-sim-sensors-system`, renderer `ogre2`. World được tạo từ
`worlds/vision_world.sdf.xacro` vào file runtime tạm; world name vẫn là
`empty` để giữ đường dẫn service của robot skills.

Camera ở `(0.38, 0, 1.15)` m, nhìn vuông góc xuống bàn cao 0.2 m.
Horizontal FOV 60°, 640×480, 15 Hz simulation time. Vùng nhìn trên bàn
xấp xỉ 0.823 m theo X × 1.097 m theo Y, bao phủ bàn 0.30×0.65 m.
Chuỗi TF: `world → camera_link → camera_optical_frame`; robot description
đã nối `world` với `base_link`. Optical frame: +Z nhìn xuống, +X sang phải,
+Y xuống dưới ảnh. Pose SDF và TF cùng đọc một cấu hình YAML.

### Build và chạy

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py
```

Chạy server không GUI:

```bash
ros2 launch ur3_vision_planning llm_robot.launch.py gazebo_gui:=false launch_rviz:=false
```

Camera vẫn cần renderer OpenGL/EGL trên máy kể cả khi không mở GUI.
Không chạy đồng thời scene Bài 02 và Bài 03 trong cùng ROS domain/Gazebo partition.
Launch không gọi LLM hoặc điều khiển pick/place.

### Topics và kiểm tra

| ROS topic | Type / encoding | Gazebo topic |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image`, `rgb8` | `/camera/image` |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | `/camera/camera_info` |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image`, `32FC1`, mét | `/camera/depth_image` |

Mở terminal mới, source ROS và workspace trước khi chạy:

```bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
# Hoặc mở rqt, chọn Plugins → Visualization → Image View.
ros2 topic echo /camera/camera_info --once --qos-reliability best_effort
ros2 topic hz /camera/image_raw
ros2 topic hz /camera/depth/image_raw
ros2 run tf2_ros tf2_echo base_link camera_optical_frame
gz model --list
```

Trong rqt_image_view chọn QoS Best Effort nếu ảnh không hiện.
Để xem depth, chọn `/camera/depth/image_raw` và bật normalize range.
Frame_id của RGB, depth và CameraInfo phải là `camera_optical_frame`.
Gazebo còn xuất `/camera/points`, chưa bridge sang ROS ở bước này.

Đã kiểm thử simulator thật: đủ 5 cube và camera, có RGB/depth/CameraInfo,
TF camera hợp lệ, ảnh RGB thấy toàn bàn và 3 zone. CameraInfo:
fx=fy≈554.256, cx=320, cy=240. Depth tại bàn khoảng 0.95 m, tại cube top
khoảng 0.91 m. Robot có thể che ảnh khi di chuyển qua mặt bàn.
MoveIt trả collision-aware IK SUCCESS cho pre-grasp của green_cube và
purple_cube; bài kiểm tra này chỉ tính IK, không chạy chuyển động.
Chưa kiểm thử perception hoặc pick/place năm cube ở bước này.

Nếu cần RGB-only ở bước sau, dùng CameraInfo K để tạo tia optical,
biến đổi tia bằng TF rồi giao với mặt phẳng bàn. Chiều cao mặt phẳng
lấy từ `scene.yaml` (table.center.z + table.size.z/2), không lấy model
pose simulator làm dữ liệu ảnh.

## Bài 03 — bước 3: camera perception

Launch tự chạy executable `camera_perception`. Node dùng RGB/depth đã align,
CameraInfo không distortion và QoS sensor data (best effort). Ngưỡng HSV,
contour, morphology, ROI, kích thước cube, depth và temporal filter đặt trong
`config/perception.yaml`; scene/zone cố định đọc từ `config/scene.yaml`.
Kích thước cube trong hai cấu hình phải khớp nhau.

Phân biệt cube/zone bằng HSV kết hợp ROI mặt bàn, độ cao từ depth, diện tích,
aspect ratio, fill ratio và kích thước vật lý suy ra từ CameraInfo/depth.
Depth loại mặt zone trước bước tìm contour, nên cube trên zone cùng màu
vẫn được tách riêng. Node chấm điểm các contour thỏa điều kiện thay vì chọn
contour màu lớn nhất. Không đọc vị trí spawn của object để nhận diện.

Tâm contour → median depth vùng 5×5 → back-project bằng K → TF tại timestamp
của ảnh sang `base_link`. Vị trí công bố là **tâm cube**: depth đo top face,
node trừ một nửa chiều cao cube theo Z của base_link. Cách này phù hợp cube
đứng thẳng trên bàn; chưa hỗ trợ vật nghiêng hoặc calibration có distortion.
Depth 32FC1 là mét, 16UC1 được đổi đơn vị theo cấu hình. RGB/depth phải cùng
frame, resolution và timestamp lệch tối đa 80 ms. Không có depth hợp lệ thì
node báo thiếu dữ liệu/stale; không tự đoán tọa độ bằng pose simulator.

Vị trí lấy median tối đa 5 frame. Khi mất contour ngắn hạn, trạng thái giữ
tối đa 1 giây, `stamp` vẫn là lần đo thật và `age_s` tăng, confidence giảm.
Sau timeout, `detected=false`, `stale=true`, `position=null`. Khi mất ảnh,
depth hoặc xử lý TF liên tục thất bại, trạng thái toàn môi trường là stale.
Timer dùng steady clock để timeout vẫn hoạt động khi Gazebo /clock dừng.

Zone chỉ bị chiếm khi toàn bộ footprint cube nằm trong vùng khả dụng và
Z tâm cube gần mặt bàn. Nếu chưa biết đủ vị trí năm block, zone chưa có vật
được xác nhận sẽ có `occupied=null`, `status=unknown`; không coi nó là trống.
JSON luôn có đủ năm tên object và ba zone. Trường `stamp` là thời gian camera;
`published_at` là thời gian xuất JSON; `pixel`, `bbox`, `confidence`, `age_s`
giúp theo dõi từng detection. Confidence là điểm hình học, không phải xác suất
đã được hiệu chuẩn.

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning
source install/setup.bash
ros2 launch ur3_vision_planning llm_robot.launch.py
```

Terminal mới đã source ROS/workspace:

```bash
ros2 run rqt_image_view rqt_image_view /vision/debug_image
ros2 topic echo /vision/environment_state
ros2 topic echo /vision/environment_state --once
```

Debug image dùng sensor QoS; chọn Best Effort trong rqt. Nếu chỉ chạy node
với simulator đang chạy, dùng lệnh sau và tránh chạy trùng node từ launch:

```bash
ros2 run ur3_vision_planning camera_perception --ros-args -p use_sim_time:=true
```

Đã kiểm thử RGB-D thật: phát hiện đủ năm màu, tọa độ tâm sai lệch khoảng
0–2.3 mm so với spawn pose; debug image có tên, bounding box, tọa độ và zone.
Chuyển blue_cube vào zone_b trong simulator để tạo ca kiểm thử: JSON đổi
zone_b thành occupied/blue_cube, các zone còn lại trống. Việc di chuyển model
chỉ là thao tác kiểm thử; perception không gọi dịch vụ hoặc đọc pose Gazebo.
Dừng simulator làm ngừng ảnh và /clock: JSON vẫn được xuất, đánh dấu stale,
position=null và zone occupancy unknown. Các unit tests kiểm tra median depth
loại 0/NaN/Inf/ngoài giới hạn và kiểm tra footprint/Z khi xác định zone.

Bạn có thể kiểm tra thêm bằng kéo cube trong Gazebo GUI hoặc che block để
quan sát timeout từng object. Các bước sau mới tích hợp môi trường camera
vào LLM/planning; bước này chưa thay planner, validator hoặc robot skills.
