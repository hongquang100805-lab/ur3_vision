"""Interactive, fail-fast execution of validated LLM plans."""

import os
import json
from ament_index_python.packages import get_package_share_directory
import rclpy
import yaml

from .llm_planner import LLMPlanner, LLMPlanningError
from .vision_robot_skills import VisionRobotSkills
from .plan_execution import validate_execution_config
from .task_validator import TaskValidator


def _print_student_header(config):
    mapping = config["zone_mapping"]
    xx = int(str(config["student_id"])[-2:])
    labels = {"yellow_cube": "Yellow", "blue_cube": "Blue", "red_cube": "Red"}
    print(f"STUDENT NAME: {config['student_name']}")
    print(f"STUDENT ID: {config['student_id']}")
    print(
        f"PERSONALIZED TASK: P={xx % 6}; "
        f"A={labels[mapping['zone_a']]}, B={labels[mapping['zone_b']]}, "
        f"C={labels[mapping['zone_c']]}"
    )


def main(args=None):
    rclpy.init(args=args)
    pkg_share = get_package_share_directory("ur3_vision_planning")
    scene_path = os.path.join(pkg_share, "config", "scene.yaml")
    student_config_path = os.path.join(pkg_share, "config", "student_config.yaml")
    with open(student_config_path, "r", encoding="utf-8") as config_file:
        student_config = yaml.safe_load(config_file)

    planner = LLMPlanner(student_config)
    validator = TaskValidator()
    skills = None
    _print_student_header(student_config)

    try:
        print(f"9ROUTER BASE URL: {planner.base_url}")
        print(f"9ROUTER MODEL: {planner.model}")
        try:
            model_ids = planner.list_models()
            if planner.model not in model_ids:
                print(
                    f"9ROUTER WARNING: configured model {planner.model!r} is "
                    "not listed by GET /models; continuing because OpenCode "
                    "Free catalogs are dynamic"
                )
            else:
                print("9ROUTER MODEL CATALOG: configured model is listed")
        except LLMPlanningError as error:
            # GET /models is advisory. The real plan request below remains the
            # authority and must succeed before any robot motion is attempted.
            print(
                "9ROUTER WARNING: GET /models advisory check failed; "
                f"continuing to the real user request: {error}"
            )
        print("9ROUTER PREFLIGHT: no connectivity or diagnostic plan request sent")

        # Constructing the clients does not command motion. Every plan must
        # still come from the configured model and pass TaskValidator below.
        with open(os.path.join(pkg_share, 'config', 'execution.yaml'), encoding='utf-8') as config_file:
            motion_config = yaml.safe_load(config_file)
        validate_execution_config(motion_config)
        skills = VisionRobotSkills(scene_path, motion_config)
        while rclpy.ok():
            command = input("\nENTER COMMAND (or 'quit'): ").strip()
            if command.lower() in ("quit", "exit", "q"):
                break

            print(f"\nUSER COMMAND:\n{command}")
            try:
                plan_json = planner.plan(command)
            except LLMPlanningError as error:
                print(f"\nLLM PLAN: FAILED ({error})")
                print("\nTASK FAILED")
                continue

            print(f"\nLLM PLAN:\n{plan_json}")
            is_valid, message = validator.validate(
                plan_json, occupied_zones=skills.get_occupied_zones()
            )
            print(f"\nVALIDATION: {message}")
            if not is_valid:
                print("\nTASK FAILED")
                continue

            plan = json.loads(plan_json)["plan"]
            print("\nEXECUTION:")
            all_successful = True
            for index, step in enumerate(plan, start=1):
                skill_name = step["skill"]
                if skill_name == "home":
                    label = "home()"
                    result = skills.home()
                elif skill_name == "pick":
                    label = f"pick({step['object']})"
                    result = skills.pick(step["object"])
                else:
                    label = f"place({step['object']}, {step['zone']})"
                    result = skills.place(step["object"], step["zone"])
                print(f"[{index}/{len(plan)}] {label} ........ {result}")
                if result != "SUCCESS":
                    all_successful = False
                    print("Execution stopped after failed step.")
                    break

            print("\nTASK SUCCESS" if all_successful else "\nTASK FAILED")
    except LLMPlanningError as error:
        print(f"\n9ROUTER PREFLIGHT: FAILED ({error})")
        print("TASK FAILED - robot execution was not started")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        planner.destroy_node()
        if skills is not None:
            skills.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
