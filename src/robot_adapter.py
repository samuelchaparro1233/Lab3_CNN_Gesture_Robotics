"""
Robot Adapter for 3-DoF Articulated Arm & Parallel Gripper in CoppeliaSim
Laboratorio 3: CNN para Reconocimiento de Gestos
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import time
import os
import sys
import math
from typing import Optional, Dict, Any, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.coppelia_client import CoppeliaSimClient
except ImportError:
    from coppelia_client import CoppeliaSimClient


class RobotAdapter:
    """
    Translates discrete, filtered CNN gesture classes (0-4) into safe,
    bounded actuator commands for a robotic arm + gripper in CoppeliaSim.
    
    Mapping Specifications (from Laboratory Guide):
    - Class 0: LOGICAL STOP / INHIBITION (Hold state, reject new movement)
    - Class 1: Discrete step on Joint 1 (Base Yaw, +/- step_deg)
    - Class 2: Discrete step on Joint 2 (Shoulder Pitch, +/- step_deg)
    - Class 3: Discrete step on Joint 3 (Elbow Pitch, +/- step_deg)
    - Class 4: Toggle Gripper state (Open <-> Close)

    Supports: Generic 3-DoF arm, uArm Swift Pro with Gripper, UR5, etc.
    Active robot is read from config/config.yaml -> coppeliasim.active_robot.
    """

    def __init__(
        self,
        coppelia_client: Optional[CoppeliaSimClient] = None,
        joint1_step_deg: float = 25.0,
        joint2_step_deg: float = 20.0,
        joint3_step_deg: float = 20.0,
        joint1_limits: Tuple[float, float] = (-170.0, 170.0),
        joint2_limits: Tuple[float, float] = (-30.0, 120.0),
        joint3_limits: Tuple[float, float] = (-110.0, 110.0)
    ):
        self.client = coppelia_client or CoppeliaSimClient()

        # Override defaults with values from config if available
        j1s, j2s, j3s, lim1, lim2, lim3 = self._read_config_limits(
            joint1_step_deg, joint2_step_deg, joint3_step_deg,
            joint1_limits, joint2_limits, joint3_limits
        )

        self.joint1_step = math.radians(j1s)
        self.joint2_step = math.radians(j2s)
        self.joint3_step = math.radians(j3s)

        self.limits = {
            "Joint1": (math.radians(lim1[0]), math.radians(lim1[1])),
            "Joint2": (math.radians(lim2[0]), math.radians(lim2[1])),
            "Joint3": (math.radians(lim3[0]), math.radians(lim3[1]))
        }

        # Joint stepping directions (+1 or -1)
        self.step_directions = {"Joint1": 1.0, "Joint2": 1.0, "Joint3": 1.0}

        self.gripper_open = True
        self.is_busy = False
        self.execution_history: List[Dict[str, Any]] = []

        # Metric counters
        self.commands_received = 0
        self.commands_executed = 0
        self.commands_clamped = 0
        self.commands_inhibited = 0

    def _read_config_limits(
        self,
        d_j1s, d_j2s, d_j3s,
        d_lim1, d_lim2, d_lim3
    ):
        """Reads joint limits from config.yaml active_robot profile. Falls back to defaults."""
        try:
            import yaml
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            cfg_path = os.path.join(project_root, "config", "config.yaml")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            active = cfg.get("coppeliasim", {}).get("active_robot", "robot3dof")
            profile = cfg.get("coppeliasim", {}).get("robots", {}).get(active, {})
            joints = profile.get("joints", {})

            def _get(j, key, default):
                return joints.get(j, {}).get(key, default)

            j1s = _get("joint1", "step_deg", d_j1s)
            j2s = _get("joint2", "step_deg", d_j2s)
            j3s = _get("joint3", "step_deg", d_j3s)
            lim1 = (_get("joint1", "min_limit_deg", d_lim1[0]), _get("joint1", "max_limit_deg", d_lim1[1]))
            lim2 = (_get("joint2", "min_limit_deg", d_lim2[0]), _get("joint2", "max_limit_deg", d_lim2[1]))
            lim3 = (_get("joint3", "min_limit_deg", d_lim3[0]), _get("joint3", "max_limit_deg", d_lim3[1]))

            print(f"[RobotAdapter] Loaded limits from config (robot: '{active}')")
            print(f"  J1 step={j1s}° lim={lim1} | J2 step={j2s}° lim={lim2} | J3 step={j3s}° lim={lim3}")
            return j1s, j2s, j3s, lim1, lim2, lim3
        except Exception:
            return d_j1s, d_j2s, d_j3s, d_lim1, d_lim2, d_lim3


    def connect_simulator(self) -> bool:
        """Initializes connection to CoppeliaSim."""
        connected = self.client.connect()
        if connected:
            self.client.start_simulation()
        return connected

    def toggle_joint_direction(self, joint_name: str):
        """Reverses stepping direction (+/-) for a given joint."""
        if joint_name in self.step_directions:
            self.step_directions[joint_name] *= -1.0
            print(f"[RobotAdapter] Direction for {joint_name} set to: {'+' if self.step_directions[joint_name] > 0 else '-'}")

    def execute_command(self, command_class: int) -> Dict[str, Any]:
        """
        Executes a discrete mechatronic action based on the accepted gesture class.
        
        Args:
            command_class (int): Filtered gesture class (0 to 4).
            
        Returns:
            Dict[str, Any]: Execution result details.
        """
        self.commands_received += 1
        t_start = time.time()
        
        result: Dict[str, Any] = {
            "command_class": command_class,
            "timestamp": t_start,
            "success": False,
            "action_description": "",
            "joint_affected": None,
            "previous_angle_deg": 0.0,
            "target_angle_deg": 0.0,
            "gripper_state": "OPEN" if self.gripper_open else "CLOSED",
            "execution_time_ms": 0.0,
            "status": "INIT"
        }
        
        # Check 1: Class 0 (Logical Stop / Inhibit)
        if command_class == 0:
            self.commands_inhibited += 1
            result["success"] = True
            result["action_description"] = "LOGICAL_STOP: Inhibit actuation / Hold current pose"
            result["status"] = "INHIBITED_HOLD"
            result["execution_time_ms"] = (time.time() - t_start) * 1000.0
            self.execution_history.append(result)
            return result
            
        # Check 2: Joint 1 Step (Class 1)
        elif command_class == 1:
            joint_name = "Joint1"
            curr_rad = self.client.get_joint_position(joint_name)
            delta = self.step_directions[joint_name] * self.joint1_step
            target_rad = curr_rad + delta
            
            # Clamp limits
            min_lim, max_lim = self.limits[joint_name]
            if target_rad > max_lim or target_rad < min_lim:
                self.commands_clamped += 1
                self.toggle_joint_direction(joint_name)
                target_rad = max(min_lim, min(max_lim, target_rad))
                result["status"] = "CLAMPED_LIMIT_DIRECTION_REVERSED"
            else:
                result["status"] = "SUCCESS"
                
            self.client.set_joint_target_position(joint_name, target_rad)
            self.commands_executed += 1
            result["success"] = True
            result["joint_affected"] = joint_name
            result["previous_angle_deg"] = math.degrees(curr_rad)
            result["target_angle_deg"] = math.degrees(target_rad)
            result["action_description"] = f"Step {joint_name}: {result['previous_angle_deg']:.1f}° -> {result['target_angle_deg']:.1f}°"

        # Check 3: Joint 2 Step (Class 2)
        elif command_class == 2:
            joint_name = "Joint2"
            curr_rad = self.client.get_joint_position(joint_name)
            delta = self.step_directions[joint_name] * self.joint2_step
            target_rad = curr_rad + delta
            
            min_lim, max_lim = self.limits[joint_name]
            if target_rad > max_lim or target_rad < min_lim:
                self.commands_clamped += 1
                self.toggle_joint_direction(joint_name)
                target_rad = max(min_lim, min(max_lim, target_rad))
                result["status"] = "CLAMPED_LIMIT_DIRECTION_REVERSED"
            else:
                result["status"] = "SUCCESS"
                
            self.client.set_joint_target_position(joint_name, target_rad)
            self.commands_executed += 1
            result["success"] = True
            result["joint_affected"] = joint_name
            result["previous_angle_deg"] = math.degrees(curr_rad)
            result["target_angle_deg"] = math.degrees(target_rad)
            result["action_description"] = f"Step {joint_name}: {result['previous_angle_deg']:.1f}° -> {result['target_angle_deg']:.1f}°"

        # Check 4: Joint 3 Step (Class 3)
        elif command_class == 3:
            joint_name = "Joint3"
            curr_rad = self.client.get_joint_position(joint_name)
            delta = self.step_directions[joint_name] * self.joint3_step
            target_rad = curr_rad + delta
            
            min_lim, max_lim = self.limits[joint_name]
            if target_rad > max_lim or target_rad < min_lim:
                self.commands_clamped += 1
                self.toggle_joint_direction(joint_name)
                target_rad = max(min_lim, min(max_lim, target_rad))
                result["status"] = "CLAMPED_LIMIT_DIRECTION_REVERSED"
            else:
                result["status"] = "SUCCESS"
                
            self.client.set_joint_target_position(joint_name, target_rad)
            self.commands_executed += 1
            result["success"] = True
            result["joint_affected"] = joint_name
            result["previous_angle_deg"] = math.degrees(curr_rad)
            result["target_angle_deg"] = math.degrees(target_rad)
            result["action_description"] = f"Step {joint_name}: {result['previous_angle_deg']:.1f}° -> {result['target_angle_deg']:.1f}°"

        # Check 5: Toggle Gripper (Class 4)
        elif command_class == 4:
            self.gripper_open = not self.gripper_open
            self.client.set_gripper_state(self.gripper_open)
            self.commands_executed += 1
            result["success"] = True
            result["gripper_state"] = "OPEN" if self.gripper_open else "CLOSED"
            result["action_description"] = f"Gripper Toggled -> {result['gripper_state']}"
            result["status"] = "SUCCESS"

        else:
            result["status"] = f"UNKNOWN_COMMAND_{command_class}"
            result["action_description"] = f"Invalid command class {command_class}"

        result["execution_time_ms"] = (time.time() - t_start) * 1000.0
        self.execution_history.append(result)
        return result

    def execute_pick_and_place_cycle(self, target_object_index: int = 0) -> Dict[str, Any]:
        """
        Executes a complete, coordinated pick-and-place sequence in CoppeliaSim
        for validation and multi-trial experimental benchmarking.
        """
        poses = [
            # 1. Approach Object
            {"j1": 30.0 * (target_object_index - 1), "j2": 25.0, "j3": -35.0, "grip": True, "desc": "Approach"},
            # 2. Lower to Object
            {"j1": 30.0 * (target_object_index - 1), "j2": 45.0, "j3": -55.0, "grip": True, "desc": "Descend"},
            # 3. Close Gripper (Grasp)
            {"j1": 30.0 * (target_object_index - 1), "j2": 45.0, "j3": -55.0, "grip": False, "desc": "Grasp"},
            # 4. Lift Object
            {"j1": 30.0 * (target_object_index - 1), "j2": 10.0, "j3": -20.0, "grip": False, "desc": "Lift"},
            # 5. Rotate to Drop Zone
            {"j1": -60.0, "j2": 10.0, "j3": -20.0, "grip": False, "desc": "Transport"},
            # 6. Lower into Drop Box
            {"j1": -60.0, "j2": 35.0, "j3": -45.0, "grip": False, "desc": "Descend_Drop"},
            # 7. Open Gripper (Release)
            {"j1": -60.0, "j2": 35.0, "j3": -45.0, "grip": True, "desc": "Release"},
            # 8. Return Home
            {"j1": 0.0, "j2": 0.0, "j3": 0.0, "grip": True, "desc": "Home"}
        ]
        
        cycle_start = time.time()
        step_results = []
        is_real_connected = self.client.is_connected and self.client.sim is not None
        
        for pose in poses:
            ok1 = self.client.set_joint_target_position("Joint1", math.radians(pose["j1"]))
            ok2 = self.client.set_joint_target_position("Joint2", math.radians(pose["j2"]))
            ok3 = self.client.set_joint_target_position("Joint3", math.radians(pose["j3"]))
            ok4 = self.client.set_gripper_state(pose["grip"])
            self.gripper_open = pose["grip"]
            time.sleep(0.05)  # Simulated motion time
            step_results.append(pose["desc"])
            
        total_time = (time.time() - cycle_start) * 1000.0
        return {
            "cycle_success": is_real_connected,
            "is_connected_to_sim": is_real_connected,
            "target_object_index": target_object_index,
            "steps_completed": len(poses) if is_real_connected else 0,
            "execution_duration_ms": total_time,
            "status": "SUCCESS" if is_real_connected else "FAILED_SIMULATOR_DISCONNECTED"
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistics for reporting."""
        return {
            "commands_received": self.commands_received,
            "commands_executed": self.commands_executed,
            "commands_clamped": self.commands_clamped,
            "commands_inhibited": self.commands_inhibited,
            "current_status": self.client.get_status()
        }
