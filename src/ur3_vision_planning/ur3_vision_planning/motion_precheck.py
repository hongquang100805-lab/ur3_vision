"""Camera + MoveIt diagnostics only; no LLM requests or execution mode."""

import rclpy
import yaml
import copy
import json

from .environment_manager import EnvironmentError, validate_structured_goal
from .natural_language_planner import NaturalLanguagePlanner, print_result
from .plan_execution import execute_validated_plan


def repeat_summary(reports, tolerance, expected_temporary):
    print('\nDETERMINISTIC PRECHECK:', flush=True)
    all_passed = True
    for index, report in enumerate(reports, 1):
        passed = (report.get('pass') is True and report.get('scene_restored') is True
            and report.get('scene_resynced') is True and report.get('temporary') == expected_temporary
            and report.get('home_error_code') == 1 and report.get('home_points', 0) > 1
            and report.get('home_final_error', float('inf')) <= .02
            and bool(report.get('cartesian'))
            and all(item['pass'] and item['fraction'] == 1. for item in report['cartesian']))
        all_passed &= passed
        print(f'run {index} ... {"PASS" if passed else "FAIL"}; temporary={report.get("temporary")}; '
              f'home_code={report.get("home_error_code")}; points={report.get("home_points")}; '
              f'final_error={report.get("home_final_error")}; reason={report.get("error", "")}', flush=True)
        print(f'  simulated retreat joints: {json.dumps(report.get("retreat_joints"), ensure_ascii=False)}', flush=True)
    complete = bool(reports) and all(isinstance(r.get('retreat_joints'), dict) for r in reports)
    variation = float('inf')
    if complete:
        names = set(reports[0]['retreat_joints'])
        complete = all(set(r['retreat_joints']) == names for r in reports)
        if complete:
            variation = max(max(r['retreat_joints'][n] for r in reports)-min(r['retreat_joints'][n] for r in reports)
                            for n in names)
    stable = complete and variation <= tolerance
    print(f'retreat joint branch stable ... {"PASS" if stable else "FAIL"}\n'
          f'maximum branch variation ... {variation:.9f} rad\nROBOT MOTION NOT STARTED', flush=True)
    return all_passed and stable


def main(args=None):
    rclpy.init(args=args)
    manager = None
    try:
        manager = NaturalLanguagePlanner()
        if manager.get_parameter('precheck_only').value is not True:
            raise EnvironmentError('DIAGNOSTIC_NODE_REQUIRES_PRECHECK_ONLY: execution is not supported')
        manager.declare_parameter('repeat_runs', 1)
        manager.declare_parameter('repeat_expected_temporary', 'temporary_8')
        repeat = manager.get_parameter('repeat_runs').value
        if type(repeat) is not int or not 1 <= repeat <= 3:
            raise EnvironmentError('INVALID_REPEAT_RUNS: require integer 1..3')
        if repeat > 1 and not manager.motion_enabled:
            raise EnvironmentError('REPEAT_PRECHECK_REQUIRES_DOUBLE_CONFIRMATION: no robot motion is authorized')
        goal = validate_structured_goal({
            'object': manager.get_parameter('object_name').value,
            'target_zone': manager.get_parameter('target_zone').value})
        reports, first_state = [], None
        for index in range(repeat):
            manager.motion_precheck_report = {}
            try:
                state = manager.fresh_snapshot()
                if first_state is not None and any(
                    abs(state['objects'][name]['position'][axis]-obj['position'][axis]) > .01
                    for name,obj in first_state['objects'].items() for axis in ('x','y','z')):
                    raise EnvironmentError('REPEAT_SCENE_CHANGED: camera objects moved between runs')
                first_state = first_state or copy.deepcopy(state)
                plan = manager.planner.resolve_goal(goal['object'], goal['target_zone'], state)
                print(f'DIRECT STRUCTURED GOAL: run {index+1}/{repeat}; no LLM request; plan resolved from camera', flush=True)
                print_result(goal, state, plan, dry_run=not manager.motion_enabled)
                if manager.motion_enabled:
                    with open(manager.get_parameter('execution_config').value, encoding='utf-8') as config_file:
                        config = yaml.safe_load(config_file)
                    execute_validated_plan(manager, goal, state, plan, config)
                reports.append(copy.deepcopy(manager.motion_precheck_report))
            except EnvironmentError as error:
                if repeat == 1:
                    raise
                report = copy.deepcopy(manager.motion_precheck_report)
                report['error'] = str(error)
                reports.append(report)
        if repeat > 1:
            tolerance = config['continuity']['repeat_branch_tolerance_rad'] if 'config' in locals() else .02
            return 0 if repeat_summary(reports, tolerance,
                manager.get_parameter('repeat_expected_temporary').value) else 1
        return 0
    except (EnvironmentError, OSError, ValueError) as error:
        print(f'PRECHECK REFUSED: {error}\nROBOT MOTION NOT STARTED', flush=True)
        return 1
    except (KeyboardInterrupt, EOFError):
        return 130
    finally:
        if manager is not None:
            manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
