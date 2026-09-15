"""
CoppeliaSim ZeroMQ Remote API Client & Fallback Interface
Laboratorio 3: CNN para Reconocimiento de Gestos y Control en CoppeliaSim
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import time
import math
import os
import sys
from typing import Optional, Tuple, Dict, Any, List

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class CoppeliaSimClient:
    """
    Manages robust connection, object handle acquisition, and step execution
    with CoppeliaSim robotics simulation platform.
    
    Supports:
    - ZeroMQ Remote API (CoppeliaSim v4.3+)
    - Automatic fallback to dry-run emulation when simulator is not active.
    - Config-driven robot profile (uArm, Robot3DoF, UR5, etc.)
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 23000, timeout_s: float = 2.0,
                 config_path: str = "config/config.yaml"):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.client = None
        self.sim = None
        self.is_connected = False
        self.is_simulation_running = False

        # Load robot profile from config
        self._robot_profile: Dict[str, Any] = {}
        self._gripper_type: str = "toggle"   # "toggle" or "suction"
        self._gripper_signal: str = "RG2_open"
        self._gripper_linear_joint: Optional[str] = None
        self._joint_aliases: Dict[str, List[str]] = {}
        self._load_robot_profile(config_path)

        # Object handles
        self.joint_handles: Dict[str, Any] = {}
        self.gripper_handle: Optional[Any] = None
        self.object_handles: Dict[str, Any] = {}

        # Simulated joint angles cache (rad)
        self.simulated_angles: Dict[str, float] = {
            "Joint1": 0.0, "Joint2": 0.0, "Joint3": 0.0
        }
        self.simulated_gripper_open: bool = True

    def _load_robot_profile(self, config_path: str):
        """Reads config.yaml and populates the active robot alias tables."""
        # Resolve config path relative to project root
        if not os.path.isabs(config_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(project_root, config_path)

        if not YAML_AVAILABLE or not os.path.exists(config_path):
            # Default fallback aliases
            self._joint_aliases = {
                "Joint1": ["Joint1", "uArm_servoJ1", "motor1", "joint", "axis"],
                "Joint2": ["Joint2", "uArm_servoJ2", "motor2", "joint0"],
                "Joint3": ["Joint3", "uArm_servoJ3", "motor3", "joint1"],
            }
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            active = cfg.get("coppeliasim", {}).get("active_robot", "robot3dof")
            robots = cfg.get("coppeliasim", {}).get("robots", {})
            profile = robots.get(active, {})
            self._robot_profile = profile

            # Build alias tables for each joint
            joints_cfg = profile.get("joints", {})
            for logical_name, jcfg in [("Joint1", joints_cfg.get("joint1", {})),
                                        ("Joint2", joints_cfg.get("joint2", {})),
                                        ("Joint3", joints_cfg.get("joint3", {}))]:
                primary = jcfg.get("alias", logical_name)
                fallbacks = jcfg.get("fallback_aliases", [])
                self._joint_aliases[logical_name] = [primary] + fallbacks + [logical_name]

            # Gripper config
            gripper_cfg = profile.get("gripper", {})
            self._gripper_type = gripper_cfg.get("type", "toggle")
            self._gripper_signal = gripper_cfg.get("signal", "RG2_open")
            self._gripper_linear_joint = gripper_cfg.get("linear_joint", None)
            print(f"[CoppeliaClient] Loaded robot profile: '{profile.get('name', active)}'")
            print(f"[CoppeliaClient] Joint aliases: {self._joint_aliases}")
            print(f"[CoppeliaClient] Gripper type: {self._gripper_type} | signal: {self._gripper_signal}")
        except Exception as ex:
            print(f"[CoppeliaClient] Warning: could not parse config ({ex}). Using defaults.")
            self._joint_aliases = {
                "Joint1": ["Joint1", "uArm_servoJ1", "motor1", "joint", "axis"],
                "Joint2": ["Joint2", "uArm_servoJ2", "motor2", "joint0"],
                "Joint3": ["Joint3", "uArm_servoJ3", "motor3", "joint1"],
            }

    def connect(self) -> bool:
        """Attempts to establish connection with CoppeliaSim ZeroMQ server."""
        if not ZMQ_AVAILABLE:
            print("[CoppeliaClient] Warning: coppeliasim_zmqremoteapi_client not installed. Running in emulation mode.")
            self.is_connected = False
            return False
            
        try:
            import socket
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s):
                pass
        except Exception as sock_err:
            # Simulator not running on port 23000 -> instant fallback
            self.is_connected = False
            self.client = None
            self.sim = None
            return False

        try:
            self.client = RemoteAPIClient(host=self.host, port=self.port)
            self.sim = self.client.require('sim')
            version = self.sim.getInt32Param(self.sim.intparam_program_version)
            self.is_connected = True
            print(f"[CoppeliaClient] Connected to CoppeliaSim (Program Version: {version})")
            return True
        except Exception as e:
            print(f"[CoppeliaClient] Connection failed ({e}). Operating in Standalone/Emulation mode.")
            self.is_connected = False
            self.client = None
            self.sim = None
            return False

    def disable_demo_scripts(self):
        """Attempts to disable embedded demo child scripts inside uArm model so gesture control has 100% full priority."""
        if not (self.is_connected and self.sim):
            return
        try:
            # 1. Search for script objects in tree
            all_scripts = self.sim.getObjectsInTree(self.sim.handle_scene, self.sim.object_script_type)
            disabled_count = 0
            if all_scripts:
                for script_h in all_scripts:
                    try:
                        alias = self.sim.getObjectAlias(script_h)
                        # Disable script if it belongs to uArm or has demo logic
                        if any(kw in alias.lower() for kw in ["uarm", "demo", "script", "child"]):
                            # Try setting script parameter enabled = 0
                            if hasattr(self.sim, 'scriptintparam_enabled'):
                                self.sim.setScriptInt32Param(script_h, self.sim.scriptintparam_enabled, 0)
                            disabled_count += 1
                    except Exception:
                        pass
            if disabled_count > 0:
                print(f"[CoppeliaClient] Solicitada desactivacion de {disabled_count} script(s) de demostracion embedidos.")
        except Exception as e:
            pass

    def start_simulation(self) -> bool:
        """Starts physics simulation in CoppeliaSim if connected."""
        if self.is_connected and self.sim:
            try:
                self.sim.startSimulation()
                self.is_simulation_running = True
                print("[CoppeliaClient] Simulation started.")
                self.disable_demo_scripts()
                return True
            except Exception as e:
                print(f"[CoppeliaClient] Error starting simulation: {e}")
                return False
        else:
            self.is_simulation_running = True
            print("[CoppeliaClient] Emulated simulation started.")
            return True

    def stop_simulation(self) -> bool:
        """Stops physics simulation in CoppeliaSim."""
        if self.is_connected and self.sim:
            try:
                self.sim.stopSimulation()
                self.is_simulation_running = False
                print("[CoppeliaClient] Simulation stopped.")
                return True
            except Exception as e:
                print(f"[CoppeliaClient] Error stopping simulation: {e}")
                return False
        else:
            self.is_simulation_running = False
            print("[CoppeliaClient] Emulated simulation stopped.")
            return True

    def get_joint_handle(self, joint_name: str) -> Optional[Any]:
        """Gets and caches handle for a joint using config aliases + auto-discovery."""
        if joint_name in self.joint_handles:
            return self.joint_handles[joint_name]

        if self.is_connected and self.sim:
            # 1. Try all aliases from config (primary + fallbacks)
            aliases = self._joint_aliases.get(joint_name, [joint_name])
            candidate_paths: List[str] = []
            for alias in aliases:
                candidate_paths += [
                    f":/{alias}",
                    f"/{alias}",
                    f"./{alias}",
                    f"/uArm/{alias}",
                    f"/Robot3DoF/{alias}",
                    f"/UR5/{alias}",
                    alias,
                ]

            for path in candidate_paths:
                try:
                    handle = self.sim.getObject(path)
                    if handle is not None and handle != -1:
                        self.joint_handles[joint_name] = handle
                        print(f"[CoppeliaClient] '{joint_name}' resolved via path '{path}' (handle: {handle})")
                        return handle
                except Exception:
                    continue

            # 2. Auto-discovery: index-based fallback across all scene joints
            try:
                all_joints = self.sim.getObjectsInTree(
                    self.sim.handle_scene, self.sim.object_joint_type
                )
                if all_joints:
                    index_map = {"Joint1": 0, "Joint2": 1, "Joint3": 2,
                                 "joint1": 0, "joint2": 1, "joint3": 2}
                    idx = index_map.get(joint_name)
                    if idx is not None and idx < len(all_joints):
                        handle = all_joints[idx]
                        try:
                            alias = self.sim.getObjectAlias(handle)
                        except Exception:
                            alias = "unknown"
                        self.joint_handles[joint_name] = handle
                        print(f"[CoppeliaClient] Auto-discovered {joint_name} -> Object '{alias}' (handle: {handle})")
                        return handle
            except Exception as ex:
                print(f"[CoppeliaClient] Auto-discovery warning: {ex}")

        # Emulation fallback
        return f"EMU_HANDLE_{joint_name}"

    def get_joint_position(self, joint_name: str) -> float:
        """Gets current position (radians) of a joint."""
        handle = self.get_joint_handle(joint_name)
        if self.is_connected and self.sim and handle is not None and not str(handle).startswith("EMU_"):
            try:
                return float(self.sim.getJointPosition(handle))
            except Exception as e:
                print(f"[CoppeliaClient] Error reading joint {joint_name}: {e}")
                return self.simulated_angles.get(joint_name, 0.0)
        return self.simulated_angles.get(joint_name, 0.0)

    def set_joint_target_position(self, joint_name: str, target_rad: float) -> bool:
        """Sets target position (radians) for a revolute joint."""
        self.simulated_angles[joint_name] = target_rad
        handle = self.get_joint_handle(joint_name)
        if self.is_connected and self.sim and handle is not None and not str(handle).startswith("EMU_"):
            try:
                self.sim.setJointTargetPosition(handle, target_rad)
                return True
            except Exception as e:
                print(f"[CoppeliaClient] Error setting target for {joint_name}: {e}")
                return False
        return True

    def set_gripper_state(self, open_gripper: bool) -> bool:
        """Controls end-effector gripper — supports both toggle (RG2) and suction (uArm) types."""
        self.simulated_gripper_open = open_gripper
        if self.is_connected and self.sim:
            try:
                if self._gripper_type == "suction":
                    # uArm: activate (1) = grip/suction ON, deactivate (0) = release
                    val = 1 if not open_gripper else 0  # open_gripper=True means RELEASE
                    self.sim.setInt32Signal(self._gripper_signal, val)
                    # Also move the linear servo if configured
                    if self._gripper_linear_joint:
                        linear_h = self.get_joint_handle(self._gripper_linear_joint)
                        if linear_h and not str(linear_h).startswith("EMU_"):
                            target = 0.0 if not open_gripper else 0.03  # 3 cm extension = release
                            self.sim.setJointTargetPosition(linear_h, target)
                else:
                    # Generic toggle gripper (RG2, parallel jaw, etc.)
                    val = 1 if open_gripper else 0
                    self.sim.setInt32Signal(self._gripper_signal, val)
                    self.sim.setFloatSignal("gripper_action", 1.0 if open_gripper else 0.0)
                return True
            except Exception as e:
                print(f"[CoppeliaClient] Error controlling gripper: {e}")
                return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic dictionary of connection and state."""
        return {
            "connected": self.is_connected,
            "simulation_running": self.is_simulation_running,
            "joint_angles_deg": {
                k: math.degrees(v) for k, v in self.simulated_angles.items()
            },
            "gripper_open": self.simulated_gripper_open,
            "api_type": "ZeroMQ Remote API" if self.is_connected else "Emulation Fallback"
        }
