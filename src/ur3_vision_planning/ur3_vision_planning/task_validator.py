"""Strict validation for plans produced by the LLM."""

import json


class TaskValidator:
    ALLOWED_SKILLS = frozenset(("home", "pick", "place"))
    ALLOWED_OBJECTS = frozenset(("red_cube", "yellow_cube", "blue_cube"))
    ALLOWED_ZONES = frozenset(("zone_a", "zone_b", "zone_c"))
    REQUIRED_FIELDS = {
        "home": frozenset(("skill",)),
        "pick": frozenset(("skill", "object")),
        "place": frozenset(("skill", "object", "zone")),
    }

    def validate(self, plan_json, occupied_zones=None):
        """Validate schema, state transitions, and known zone occupancy."""
        try:
            data = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
        except (json.JSONDecodeError, TypeError) as error:
            return False, f"JSON_ERROR: {error}"

        if not isinstance(data, dict):
            return False, "INVALID_PLAN: root must be a JSON object"
        if set(data) != {"plan"}:
            return False, "INVALID_PLAN: root must contain only 'plan'"
        if not isinstance(data["plan"], list) or not data["plan"]:
            return False, "INVALID_PLAN: 'plan' must be a non-empty array"

        occupancy = dict(occupied_zones or {})
        object_zones = {obj: zone for zone, obj in occupancy.items()}
        held_object = None
        home_seen = False

        for index, step in enumerate(data["plan"], start=1):
            prefix = f"step {index}"
            if not isinstance(step, dict):
                return False, f"INVALID_PLAN: {prefix} must be an object"
            skill = step.get("skill")
            if not isinstance(skill, str) or skill not in self.ALLOWED_SKILLS:
                return False, f"INVALID_SKILL: {prefix} has unsupported skill {skill!r}"
            if set(step) != self.REQUIRED_FIELDS[skill]:
                expected = sorted(self.REQUIRED_FIELDS[skill])
                return False, f"INVALID_PLAN: {prefix} fields must be {expected}"
            if home_seen:
                return False, f"INVALID_ORDER: {prefix} appears after home"

            if skill == "home":
                if held_object is not None:
                    return False, f"INVALID_ORDER: home while holding {held_object}"
                if index != len(data["plan"]):
                    return False, "INVALID_ORDER: home must be the final step"
                home_seen = True
                continue

            obj = step["object"]
            if not isinstance(obj, str) or obj not in self.ALLOWED_OBJECTS:
                return False, f"INVALID_OBJECT: {prefix} has unsupported object {obj!r}"

            if skill == "pick":
                if held_object is not None:
                    return False, (
                        f"INVALID_ORDER: cannot pick {obj} while holding {held_object}"
                    )
                held_object = obj
                old_zone = object_zones.pop(obj, None)
                if old_zone is not None:
                    occupancy.pop(old_zone, None)
                continue

            zone = step["zone"]
            if not isinstance(zone, str) or zone not in self.ALLOWED_ZONES:
                return False, f"INVALID_ZONE: {prefix} has unsupported zone {zone!r}"
            if held_object is None:
                return False, f"INVALID_ORDER: cannot place {obj} before pick"
            if held_object != obj:
                return False, (
                    f"INVALID_ORDER: place object {obj} "
                    f"does not match held {held_object}"
                )
            occupant = occupancy.get(zone)
            if occupant is not None and occupant != obj:
                return False, f"ZONE_OCCUPIED: {zone} already contains {occupant}"
            occupancy[zone] = obj
            object_zones[obj] = zone
            held_object = None

        if held_object is not None:
            return False, f"INVALID_ORDER: plan ends while holding {held_object}"
        if not home_seen:
            return False, "INVALID_ORDER: plan must end with home"
        return True, "PLAN VALID"
