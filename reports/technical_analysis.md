# Tổng hợp kỹ thuật Bài thực hành 03

Đọc nguồn tại `/home/lenovo/ur_ws` ngày 05/10/2026. Không sửa mã nguồn, không khởi chạy simulator/motion node, không gọi LLM, không đọc `.env` hoặc credential. Chỉ tạo các tài liệu trong `reports/` theo yêu cầu bổ sung. Báo cáo hoàn chỉnh: [main.tex](main.tex).

Quy ước: **Nguồn** = xác nhận có triển khai; **Log** = có log đã đọc; **Tài liệu** = README ghi đã thử, chưa kiểm chứng raw log; **TODO** = thiếu bằng chứng hoặc cần kiểm tra runtime. Các chức năng cũ không thuộc Bài 03 được đánh dấu hạn chế, không được dùng thành nhiệm vụ của bài.

Trong các đường dẫn rút gọn bên dưới, `VP=src/ur3_vision_planning`, `PY=src/ur3_vision_planning/ur3_vision_planning`, `SIM=src/ur_simulation_gz/ur_simulation_gz` (tất cả tương đối với workspace).

## 1. Cấu trúc chương trình

**Nguồn:** `VP/` gồm `launch/`, `config/`, `worlds/`, `urdf/`, module Python cùng tên package, `test/`, `README.md`, `setup.py`, `setup.cfg`, `package.xml`. `setup.py` cài launch/YAML/xacro và đăng ký bảy console scripts. Có 12 tệp test.

- `launch/llm_robot.launch.py`: `launch_setup`, `generate_launch_description`; khởi tạo Gazebo/MoveIt, table, năm cube, ba zone, camera bridge/TF, perception, scene và gripper spawner.
- `config/scene.yaml`, `scene_config.load_scene`: hình học, spawn ban đầu và scenario. Block 40 mm, khối lượng 0.1 kg; mặt bàn z=0.200 m, tâm cube z=0.220 m. Tọa độ này là geometry/spawn, không thay camera pick target.
- `config/camera.yaml`, `perception.yaml`: camera và HSV/depth/temporal.
- `temporary_positions.yaml`, `execution.yaml`: 12 candidate tâm vật, clearance, timeout, precheck/verification/continuity.
- `worlds/vision_world.sdf.xacro`: RGB-D trong world `empty`.
- `urdf/ur3_table_mount.urdf.xacro`: include UR và `simple_parallel_gripper`; macro Robotiq có trong thư mục nhưng không chứng minh được dùng.

**Kế thừa:** `src/ur3_llm_control` có transport planner, validator legacy, scene publisher, robot skills, URDF và controller nền. **Bổ sung:** RGB-D, green/purple, perception, Environment Manager, structured-goal planner, expanded executor, camera verification, precheck và guard quỹ đạo. Package `src/ur3_circle` không nằm trong luồng chính.

## 2. Node/package chính

Executable lấy từ `VP/setup.py`, tên node lấy từ constructor:

| Executable | Node thực tế / class | Chức năng |
|---|---|---|
| `camera_perception` | `camera_perception` / `CameraPerception` | RGB-D → debug và JSON |
| `scene_publisher` | `scene_publisher` / `ScenePublisher` | Khởi tạo MoveIt scene, dừng timer khi apply thành công |
| `environment_manager_test` | `environment_manager_test` / `EnvironmentManager` | Camera → plan từ object/zone parameter, không LLM/motion |
| `natural_language_planner` | `natural_language_planner` và `llm_planner` | Manager cộng `StructuredGoalPlanner`; có dry-run/precheck/execution |
| `motion_precheck` | tạo `NaturalLanguagePlanner`, giữ tên `natural_language_planner` | Goal từ parameter, không LLM, chỉ precheck-only |
| `llm_planner` | `llm_planner` / `LLMPlanner` | Luồng legacy, LLM trả plan ba màu |
| `skill_executor` | tạo `llm_planner` và `robot_skills` | Luồng interactive legacy, không camera expanded pipeline |

`VisionRobotSkills` kế thừa `RobotSkills(Node)`, giữ tên `robot_skills`. `EnvironmentPlanner`, `DryRunPlanValidator`, `TaskValidator`, `CameraPlacementConfirmation`, `ContinuousMotionMixin`, module `home_precheck` và `joint_continuity` là hỗ trợ, không node/executable riêng.

**Nguồn:** `VP/package.xml` và `SIM/package.xml`: rclpy, geometry/std/sensor/moveit/shape messages, rcl_interfaces, action_msgs, tf2_ros, cv_bridge, OpenCV, NumPy, YAML, xacro, ament_index_python; hạ tầng UR còn cần `ur_description`, `ur_moveit_config`, `ur_controllers`, `gz_ros2_control`, `ros_gz_sim`, `ros_gz_bridge`, controller_manager. Node hạ tầng lấy từ `SIM/launch/ur_sim_control.launch.py::launch_setup` và `ur_sim_moveit.launch.py::launch_setup`: robot_state_publisher, spawner, create, clock bridge, MoveIt và RViz. Chưa lấy runtime node list.

## 3. Giao tiếp ROS 2

**Nguồn:** `PY/camera_perception.py::CameraPerception.__init__`, `robot_skills.py::RobotSkills.__init__`, `vision_robot_skills.py::VisionRobotSkills.__init__`, `environment_manager.py::EnvironmentManager.__init__`.

| Topic | Message / nguồn |
|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image`, Gazebo `/camera/image` |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image`, Gazebo `/camera/depth_image` |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo`, Gazebo `/camera/camera_info` |
| `/vision/debug_image` | `sensor_msgs/msg/Image`, perception |
| `/vision/environment_state` | `std_msgs/msg/String`, JSON perception |
| `/joint_states` | `sensor_msgs/msg/JointState`, controller |
| `/simple_gripper_controller/commands` | `std_msgs/msg/Float64MultiArray`, hai velocity commands |
| `/planning_scene` | `moveit_msgs/msg/PlanningScene`; publisher được tạo, scene khởi tạo chủ yếu qua service |
| `/robot_description`, `/robot_description_semantic` | `std_msgs/msg/String`, fallback description runtime |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` |
| `/clock` | `rosgraph_msgs/msg/Clock`, Gazebo bridge |

Services `moveit_msgs/srv`: `/compute_ik` (`GetPositionIK`), `/compute_fk` (`GetPositionFK`), `/compute_cartesian_path` (`GetCartesianPath`), `/check_state_validity` (`GetStateValidity`), `/get_planning_scene` (`GetPlanningScene`), `/apply_planning_scene` (`ApplyPlanningScene`), `/plan_kinematic_path` (`GetMotionPlan`). HOME đọc `/move_group/list_parameters`, `/get_parameter_types`, `/get_parameters` theo namespace `/move_group`, kiểu tương ứng `rcl_interfaces/srv`.

Actions nền: `/move_action` (`moveit_msgs/action/MoveGroup`), `/execute_trajectory` (`moveit_msgs/action/ExecuteTrajectory`). Adapter mới dùng planning service rồi ExecuteTrajectory. Controller `scaled_joint_trajectory_controller` cung cấp tầng FollowJointTrajectory; LLM không gửi action này.

**Phân biệt:** `/ur3_vision_planning/grasp/<object>/{attach,detach,state}` và `/world/empty/entity/system/add` là Gazebo Transport, không phải ROS topic/service được bridge. Lệnh attach/detach phát `gz.msgs.Empty`; attach discovery Gazebo 8 dùng generic `google.protobuf.Message`, detach subscriber dùng `gz.msgs.Empty`; state được parse chuỗi attached/detached. Đối chiếu `RobotSkills.attachment_subscriber_present`, `load_gazebo_attachment_plugin`, `set_gazebo_attachment`.

## 4. Kiến trúc và luồng xử lý

**Nguồn:** `PY/natural_language_planner.py::resolve_command`, `StructuredGoalPlanner.structured_goal`, `NaturalLanguagePlanner.fresh_snapshot`; `environment_manager.py::EnvironmentPlanner.resolve_goal`, `DryRunPlanValidator.validate`; `plan_execution.py::execute_validated_plan`.

```text
Câu lệnh → 9Router → {object,target_zone} strict JSON
 → snapshot camera mới → EnvironmentPlanner.resolve_goal
 → expanded plan → validator → feasibility precheck
 → execute_validated_plan → VisionRobotSkills
 → MoveIt planning → trajectory guard → ExecuteTrajectory → Gazebo UR3
 → camera confirmation sau place → bước tiếp theo/home
```

LLM chỉ quyết định ý định object và zone. Occupant, temporary, expanded steps, coordinates, motion và verification là logic chương trình. HTTP chạy thread riêng; ROS nhận camera trong lúc đợi; cần snapshot mới sau goal validation. Goal/plan đi qua gọi hàm Python, không có topic riêng truyền plan LLM sang executor.

## 5. Camera và nhận diện môi trường

**Nguồn:** `VP/config/camera.yaml`, `perception.yaml`, `worlds/vision_world.sdf.xacro`, `PY/camera_perception.py::{process_pending,process_pair,median_depth,object_zone,publish_state}`, `environment_manager.py::{check_environment,snapshot}`.

Camera overhead RGB-D 640×480, cấu hình 15 Hz, FOV ngang 60°, tại (0.38,0,1.15); static TF world → camera_link → camera_optical_frame. RGB/depth ghép nearest timestamp ≤0.08 s, queues dài 8. Yêu cầu aligned RGB-D, frame/dimensions trùng CameraInfo, focal dương, distortion gần 0. `16UC1` đổi mm→m, `32FC1` dùng m. RGB-only fallback chưa triển khai.

HSV: đỏ hai khoảng hue 0–10/170–179, vàng 20–38, blue 95–125, green 40–85, purple 130–165. ROI hình học bàn, độ cao mặt vật loại pad zone; morphology kernel 3, contour area 200–1400 px, aspect 0.65–1.55, fill ≥0.65, kích thước vật lý 0.65–1.45 lần block 40 mm. Chọn score cao nhất sau lọc; giả định một block mỗi màu.

Tại centroid (u,v), depth median patch radius 2 px, ít nhất 5 mẫu hợp lệ. CameraInfo cho điểm optical `[(u-cx)d/fx,(v-cy)d/fy,d]`; TF timestamp RGB đưa sang base_link; z giảm nửa chiều cao để báo tâm block. Lọc median 5 mẫu. Không lấy spawn pose để tạo detection. YAML dùng hình học table/zone/block, **zone marker chưa được nhận diện độc lập từ ảnh**.

Zone occupied khi full footprint cube nằm trong usable 75×75 mm trừ margin 1 mm và z trên bàn ±15 mm. Nếu thiếu vật, zone không có occupant nhìn thấy là unknown, không empty. Camera xuất JSON 5 Hz qua steady timer; timeout ảnh/depth/object 1 s. Vật mất dấu/stale: detected false, position null, confidence 0. Score heuristic giảm theo tuổi, **manager chưa kiểm tra ngưỡng confidence**. Manager yêu cầu đủ năm vật, receive/measurement 1.5 s, object 1 s, frame/occupancy nhất quán và timestamp còn tiến.

JSON environment_state có `stamp,published_at,frame_id,stale,status,reason,objects,zones`. Mỗi object: `detected,stale,position,zone,confidence,age_s,stamp,pixel,bbox`; mỗi zone: `occupied,object,objects,stale,status`. Ví dụ đầy đủ và được ghi rõ chỉ minh họa nằm ở Phụ lục B của `main.tex`.

## 6. LLM Planner và Plan Validator

**Nguồn:** `PY/llm_planner.py::{_configured_model,_chat_completion,_request_json}`, `natural_language_planner.py::{goal_system_prompt,StructuredGoalPlanner,advisory_catalog}`, `environment_manager.py::{parse_json,validate_structured_goal,DryRunPlanValidator.validate}`.

Base URL default `http://localhost:20128/v1`; parameter `openai_base_url` override biến môi trường `OPENAI_BASE_URL`. Model: parameter `llm_model` > `OPENAI_MODEL` > `LLM_MODEL`. StructuredGoalPlanner default rỗng, thiếu model phải fail. `LLMPlanner` legacy default `gemini/gemini-3.8-flash` không phải default của pipeline mới. README ghi model từng dùng `oc/muse-spark-1.2-contributor-free`; **chưa xác nhận runtime hiện tại**. Không đọc `.env`, không công bố credential.

POST `/chat/completions`, `stream=false`, JSON mode, temperature 0, timeout default 45 s. GET `/models` advisory. Một POST mỗi command, không retry/model fallback/mock runtime. Lỗi 401/404/HTTP khác, timeout, empty body, SSE, content-type sai, JSON sai hoặc message thiếu đều từ chối. StructuredGoalPlanner redacts credential trong preview.

LLM schema đúng `{object,target_zone}`; năm cube/ba zone. Expanded root đúng `{plan,temporary_positions}`; chỉ pick/place/home; pick `{skill,object}`, place `{skill,object,destination}`, home `{skill}`. Destination là zone hoặc temporary đúng trusted ledger, frame base_link và XYZ hữu hạn. Legacy `TaskValidator` chỉ ba màu/place field `zone`, không dùng để mô tả pipeline camera.

Từ chối extra fields, duplicate keys, NaN, Markdown, object/zone/skill ngoài allowlist, tọa độ LLM tự thêm, pick khi giữ vật, place sai object, home không cuối/còn vật, target occupied chưa dời occupant, slot không tin cậy hoặc plan không khớp chính xác goal. Không thấy đủ vật hoặc environment partial/stale thì dừng.

## 7. Robot skill

**Nguồn:** `PY/robot_skills.py::{pick,place_at,command_gripper,load_gazebo_attachment_plugin,set_gazebo_attachment,update_attached_object,verify_tool_pose}`, `vision_robot_skills.py::{pick_from_camera,place_destination,camera_retreat,sync_camera_scene}`, `continuous_motion.py::home`, `plan_execution.py::{CameraPlacementConfirmation.observe,wait_camera_confirmation}`.

Pick: lấy lại camera trước mỗi pick, mở ngón → pre-grasp → Cartesian lowering → đóng có contact heuristic → Gazebo DetachableJoint và MoveIt attached cube → lift → kiểm chứng cube follow tool bằng pose Gazebo, tolerance mỗi trục 15 mm. Pose Gazebo không làm pick target của camera pipeline.

Place: đúng selected center → Cartesian pre-place/descent, fallback pose chỉ khi planning failure và vẫn guard → TF final pose tolerance 8 mm/0.08 rad → xác nhận detach → mở → đọc pose thực và detach MoveIt world box → retreat → camera-retreat → tối thiểu ba frame mới xác nhận vị trí/zone và zone nguồn trống. Verification 20 s, XY 20 mm, Z 12 mm. Tool offset 0.16 m; nominal final z=0.385 m, pre-place z=0.400 m.

Home: canonical sáu arm joint `(0,-1.5707,0,0,0,0)`; adapter dùng actual state, plan-only, guard rồi ExecuteTrajectory. Goal tương đương có bounds/FK/validity chỉ thử khi canonical planning failure/timeout. Không home khi còn attachment.

Gripper velocity loop `Float64MultiArray` hai ngón, targets `[position,0.043-position]`, speed 0.03 m/s; open position 0, close 0.043 và require_object. Contact là ngón dừng giữa hành trình đối xứng, không phải lực đo.

Plugin nối wrist_3_link tới object/link sau contact; require explicit attached/detached state. Cache ack chỉ session hiện tại và subscriber đúng kiểu. Subscriber presence không tự chứng minh attachment. Đồng bộ MoveIt world/attached boxes và touch_links; camera re-sync trước execution/pick, camera xác nhận sau place.

**Kiểm tra set-pose:** package còn `set_gazebo_model_pose` gọi `/world/empty/set_pose`; `restore_grasp_object` gọi nó. `VisionRobotSkills.__init__` đặt `preserve_failed_grasp=True`, các nhánh gọi restore trong pick được chặn bởi cờ này. Không thấy set-pose trong pick/place bình thường của adapter camera. Không kết luận toàn package không có API đó. Spawn `ros_gz_sim/create` với XYZ ban đầu là khởi tạo; diff pose ảo MoveIt precheck không thay vật Gazebo. **TODO:** trace runtime để kiểm chứng toàn chuỗi không teleport.

## 8. Zone bị chiếm và temporary

**Nguồn:** `PY/environment_manager.py::EnvironmentPlanner.{resolve_goal,candidate_rejection,free_temporary_candidates,adopt_prechecked_plan}`, `vision_robot_skills.py::{feasibility_precheck,destination_center}`, `VP/config/temporary_positions.yaml`.

Manager lấy occupant từ camera zone. Tầng 1 lọc đủ 12 candidate: đúng frame/finite, tâm trên bàn ±5 mm, full footprint và table margin 10 mm, base radius 0.25–0.52 m, tránh zone marker, cách camera cubes ≥80 mm, full top face trong frustum cấu hình. Hết slot → NO_FREE_TEMPORARY_POSITION. Đây không phải IK proof.

Tầng 2 thử collision-aware pre/final-place IK với displaced cube attach ảo. Chọn candidate đầu cả hai PASS, cập nhật plan/temporary/ledger và validate lại; hết reachable slot thì fail. Temporary giữ chính xác XYZ, không cộng zone offset. Zone có thể chọn Y offset nằm trong usable footprint.

Ví dụ sau selection **giả định** temporary_8 hợp lệ:

```json
{"plan":[
 {"skill":"pick","object":"blue_cube"},
 {"skill":"place","object":"blue_cube","destination":"temporary_8"},
 {"skill":"pick","object":"red_cube"},
 {"skill":"place","object":"red_cube","destination":"zone_b"},
 {"skill":"home"}],
 "temporary_positions":{"temporary_8":{"frame_id":"base_link","x":0.28,"y":-0.27,"z":0.22}}}
```

**Tài liệu:** dry-run từng chọn temporary_1; precheck 03/10 ghi temporary_8. Không hard-code lựa chọn này thành kết quả cho mọi lượt.

## 9. Khả thi và an toàn

**Nguồn:** `PY/vision_robot_skills.py::{feasibility_precheck,_cartesian_precheck,_diagnose_cartesian_waypoints,_home_model,_home_plan_candidate,_verify_scene_restored,stop_pending_motion}`, `home_precheck.py::home_joint_model`, `continuous_motion.py::{_normalize_branch,_guard_trajectory,_execute_checked}`, `joint_continuity.py::{branch_candidates,check_trajectory}`, `plan_execution.py::execute_validated_plan`.

Precheck: simulator models, TF tool0, sáu joints, table collision, không attached object; camera sync/baseline → collision-aware pre-grasp IK → lowering/lift Cartesian fraction=1.0/FK, step 5 mm → attached-place IK → retreat IK → HOME GetMotionPlan từ retreat seed giả lập. ACM chỉ cho target/touch links; geometric diagnostic IK phải qua whole-robot state validity, không thay Cartesian fraction thiếu thành PASS.

HOME lấy URDF/SRDF/limits runtime, đọc declared/initialized parameters riêng và fallback transient-local topic. HOME trajectory phải nối từ simulated start, bounds, endpoint và waypoint validity hợp lệ. 99999 không SUCCESS.

Guard kiểm tra đúng sáu arm names, hữu hạn/độ dài vector, increasing time, bounds, first point khớp fresh actual ≤0.03 rad khi execute, raw delta mỗi point ≤0.5 rad, FK/joint endpoint/endpoint collision. Không sửa trajectory lỗi; branch chọn winding gần seed, kiểm raw delta không che jump bằng modulo. Guard chung không tự check collision liên tục mọi đoạn; dựa planner và Cartesian checking, HOME có waypoint checking riêng. Chưa thấy kiểm biên độ velocity/acceleration so dynamic limits trong guard.

Lỗi bất kỳ bước: dừng plan, cancel active action, gripper velocity zero nếu motion đã bắt đầu; không tự release/home, giữ conservative attached state. Stop chưa xác nhận báo STOP_NOT_CONFIRMED. Restore precheck world/attachments/ACM trong finally, đọc lại verify, resync camera; restore/resync fail chặn execution. Trước motion validate lại và camera drift tối đa 10 mm.

**Giới hạn:** place pre/final và camera-retreat chỉ endpoint IK; precheck không plan mọi connecting path hoặc chứng minh grasp vật lý. Execution vẫn có thể fail sau precheck PASS. Target cube bị remove khỏi world trước execution lowering, trong khi precheck giữ target/touch ACM: cần đối chiếu collision fidelity giữa hai chế độ.

## 10. Hướng dẫn chạy

**Nguồn:** `VP/setup.py`, `launch/llm_robot.launch.py::{launch_setup,generate_launch_description}`, `PY/natural_language_planner.py::{__init__,main}`, `motion_precheck.py::main`. Những lệnh sau **chỉ mô tả**, chưa chạy trong lần phân tích:

```bash
cd ~/ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur3_vision_planning --symlink-install
source install/setup.bash
# Nếu chưa có package mô phỏng local, build thêm ur_simulation_gz.

# Chọn một scenario, không chạy đồng thời hai simulator:
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=default
# Hoặc:
ros2 launch ur3_vision_planning llm_robot.launch.py scenario:=zone_b_occupied

# Launch đã chạy perception. Chỉ chạy riêng khi chưa có node đó:
ros2 run ur3_vision_planning camera_perception --ros-args -p use_sim_time:=true
ros2 topic echo /vision/environment_state --once
ros2 run rqt_image_view rqt_image_view /vision/debug_image

# Camera resolve không gọi LLM/motion:
ros2 run ur3_vision_planning environment_manager_test --ros-args \
  -p use_sim_time:=true -p object_name:=red_cube -p target_zone:=zone_b

# Chuẩn bị 9Router và credential qua môi trường, không in credential.
# README ghi model từng dùng; cần xác nhận model khả dụng thực tế:
export OPENAI_BASE_URL=http://localhost:20128/v1
export OPENAI_MODEL=oc/muse-spark-1.2-contributor-free
ros2 run ur3_vision_planning natural_language_planner --ros-args \
  -p use_sim_time:=true -p dry_run:=true \
  -p command:='Đưa vật màu đỏ vào vùng B.'

# Precheck LLM + camera, không motion:
ros2 run ur3_vision_planning natural_language_planner --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=true -p command:='Đưa vật màu đỏ vào vùng B.'

# Precheck không LLM; node không hỗ trợ precheck_only=false:
ros2 run ur3_vision_planning motion_precheck --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=true -p object_name:=red_cube -p target_zone:=zone_b

# Execution có chuyển động Gazebo, dành cho operator kiểm thử:
ros2 run ur3_vision_planning natural_language_planner --ros-args \
  -p use_sim_time:=true -p dry_run:=false -p execute_robot:=true \
  -p precheck_only:=false -p command:='Đưa vật màu đỏ vào vùng B.'
```

Bỏ parameter command để nhập tương tác. GUI parameters thực là `launch_rviz`, `gazebo_gui`; scenario chỉ `default`, `zone_b_occupied`. Launch chính cố định `ur_type=ur3`, không expose UR3e. Launch trung gian local không khai báo/chuyển tiếp tường minh world_file; control đọc từ launch context, **cần kiểm tra runtime camera world được nạp**, chưa đủ căn cứ khẳng định bị mất argument.

Default zone trống nhưng block có cặp cách 75 mm; phải precheck, không coi layout an toàn tương đương occupied giãn vật. Sau execution fail/held object, phục hồi simulator trước retry. Lệnh dành cho Gazebo, không thử UR phần cứng. Guard presence Gazebo không chứng minh mọi endpoint controller cùng thuộc simulator; cần ROS domain/runtime kiểm tra.

## 11. Kết quả và hạn chế

**Log:** `log/build_2026-10-04_20-52-35/ur3_vision_planning/stdout.log` ghi cài script và package, stderr 0 byte. Có các build log trước đó. Chưa tìm thấy log colcon test/JUnit package này trong `build/log` đã rà soát; không chạy lại tests.

**Tài liệu:** README ghi camera xác định blue ở B (x≈0.28149 m), dry-run 9Router red→B mở rộng plan; ba precheck 03/10/2026 chọn temporary_8, grasp/lift fraction 1.000, scene restore/resync PASS, HOME code 1/64 points. Các sai số HOME và branch variation chỉ là số README ghi, raw runtime log chưa có để đối chiếu. README nói execution với guard mới chủ động chưa thử. Các số 39/72 unit tests ở đoạn cũ không được coi kết quả suite hiện tại.

**Nguồn triển khai:** camera five colors, occupied-zone expansion, trusted temporary, strict goal/plan validator, Gazebo attachment acknowledgement, scene sync/rollback, home precheck, trajectory guard và camera confirmation. **Chưa đủ bằng chứng:** hoàn thành chuỗi gắp/giữ/dời/thả expanded plan hiện tại, TASK SUCCESS, hiệu năng/tỷ lệ thành công, UR3e, phần cứng thật, trace không teleport.

**Rủi ro:** legacy skill_executor không camera-first, chỉ ba màu; zone pose cấu hình, chưa detect marker; HSV giả định một vật mỗi màu; confidence chưa threshold; đủ năm vật dễ bị occlusion block plan; precheck endpoint IK không full path proof; target collision xử lý khác execution/precheck; dynamic limits chưa guard độc lập; recovery state sau release có thể bảo thủ; API set-pose legacy vẫn tồn tại. Nội dung cá nhân hóa cũ còn trong prompt/header, cần loại bỏ khi được phép sửa code, không đưa vào yêu cầu Bài 03. README có các đoạn lịch sử không khớp mã hiện tại (số candidate, layout, perception), ưu tiên nguồn khi mô tả chức năng.

Ảnh cần bổ sung vào `reports/images/`:

| Ảnh | Nội dung / phần báo cáo |
|---|---|
| gazebo_environment.png | Toàn cảnh robot/gripper/camera/table/5 cube/3 zone — Camera |
| camera_detection.png | Debug bbox/labels — Camera |
| zone_b_occupied.png | Blue ở B sau spawn — Xử lý occupied |
| environment_state.png | JSON camera thấy blue occupant — Camera |
| expanded_plan.png | Command, goal, final plan/selected slot — Xử lý occupied |
| blue_at_temporary.png | Dời blue vật lý, B trống — Xử lý occupied |
| red_at_zone_b.png | Red sau đặt, blue vẫn ở temporary — Xử lý occupied |
| task_success.png | Steps + camera verification + TASK SUCCESS cùng một lượt execution thật — Thực nghiệm |

**TODO người thực nghiệm xác nhận:** raw logs, revision, model runtime, slot chọn, camera verification, attach/detach/follow, không set-pose, test suite và cả hai scenario. Không tự tạo ảnh hoặc log thành công. XeLaTeX không có executable trong PATH lúc kiểm tra; chưa biên dịch. `main.tex` có TikZ và tám figure dùng IfFileExists/placeholder. Khi có XeLaTeX, chạy hai lần từ reports: `xelatex -interaction=nonstopmode -halt-on-error main.tex`.
