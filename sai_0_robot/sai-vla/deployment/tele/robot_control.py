#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_control_without_consist_finger.py (UPDATED + robust hand state)

Integrated from robot_control_with_good_action.py:
- action(): optimized (single robot_state call + ref_joints IK + hand command caching)
- move_path(): waypoint interpolation executor
- ik_left_from_xyzabc / ik_right_from_xyzabc: accept optional ref_joints to avoid extra robot_state calls

Kept from original:
- get_obs() and step() environment-style interface
- FK helpers + hand state reading helpers
- sequential hand_move_6f() implementation

FIXED (important):
- _hand_binary_from_state(): now robust to per-finger 500 errors (skip failures, no crash)
"""

from __future__ import annotations
import math
import json
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests
import numpy as np
from gr00t_client import Gr00tClient



from fx_kine import Marvin_Kine

# Camera (optional)
try:
    from camera import HKCamera
    CAMERA_AVAILABLE = True
except Exception:
    HKCamera = None
    CAMERA_AVAILABLE = False


class RobotAPIError(RuntimeError):
    pass


@dataclass
class Robot:
    host: str = "192.168.50.52"
    port: int = 8080
    timeout_s: float = 5.0
    robot_ip: str = "192.168.10.190"

    kine_config_path: str = os.path.expanduser("/home/nvidia/Desktop/kinematics/ccs_m6.MvKDCfg")

    def __post_init__(self) -> None:
        self.base_url = f"http://{self.host}:{self.port}"
        self.camera = None

        # --- ADDED: cache last hand commands to avoid re-sending every step ---
        self._last_left_hand: Optional[float] = None
        self._last_right_hand: Optional[float] = None

        # --- ADDED: throttle warnings for failing finger state reads ---
        self._warned_finger_state: set = set()

        # Kinematics
        self._kk = Marvin_Kine()
        self._kine_ini = self._kk.load_config(self.kine_config_path)
        if self._kine_ini is None:
            raise RuntimeError("load_config failed in Robot __post_init__")

        ok_l = self._kk.initial_kine(
            robot_serial=0,
            robot_type=self._kine_ini["TYPE"][0],
            dh=self._kine_ini["DH"][0],
            pnva=self._kine_ini["PNVA"][0],
            j67=self._kine_ini["BD"][0],
        )
        ok_r = self._kk.initial_kine(
            robot_serial=1,
            robot_type=self._kine_ini["TYPE"][1],
            dh=self._kine_ini["DH"][1],
            pnva=self._kine_ini["PNVA"][1],
            j67=self._kine_ini["BD"][1],
        )
        if not (ok_l and ok_r):
            raise RuntimeError("initial_kine failed in Robot __post_init__")

    # ---------------- Camera ----------------
    def init_camera(self, device_serial: str = "DA6567795", target_fps: float = 30.0) -> bool:
        if not CAMERA_AVAILABLE or HKCamera is None:
            print("❌ Camera module not available")
            return False
        try:
            self.camera = HKCamera(device_serial, target_fps)
            return True
        except Exception as e:
            print(f"❌ init_camera failed: {e}")
            self.camera = None
            return False

    def capture_image(
        self,
        timeout_ms: int = 1000,
        save: bool = False,
        filename: Optional[str] = None
    ) -> Optional[np.ndarray]:
        if self.camera is None:
            if not self.init_camera():
                return None
        try:
            return self.camera.capture_one_frame_bgr(timeout_ms=timeout_ms, save=save, filename=filename)
        except Exception as e:
            print(f"❌ capture_image failed: {e}")
            return None

    def close_camera(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

    # ---------------- HTTP helpers ----------------
    def _url(self, path: str) -> str:
        return self.base_url + (path if path.startswith("/") else "/" + path)

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._url(path)
        try:
            if payload is None:
                r = requests.post(url, data=b"", timeout=self.timeout_s)
            else:
                r = requests.post(
                    url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout_s,
                )
        except requests.RequestException as e:
            raise RobotAPIError(f"POST {url} failed: {e}") from e

        if r.status_code >= 400:
            raise RobotAPIError(f"POST {url} -> {r.status_code}: {r.text}")

        if not r.text.strip():
            return {"ok": True, "status_code": r.status_code}

        try:
            return r.json()
        except Exception:
            return {"raw": r.text, "status_code": r.status_code}

    def _get(self, path: str) -> Dict[str, Any]:
        url = self._url(path)
        try:
            r = requests.get(url, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise RobotAPIError(f"GET {url} failed: {e}") from e

        if r.status_code >= 400:
            raise RobotAPIError(f"GET {url} -> {r.status_code}: {r.text}")

        if not r.text.strip():
            return {"ok": True, "status_code": r.status_code}

        try:
            return r.json()
        except Exception:
            return {"raw": r.text, "status_code": r.status_code}

    # ---------------- Robot / Arm ----------------
    def robot_connect(self) -> Dict[str, Any]:
        if not self.robot_ip:
            raise ValueError("robot_ip is empty")
        return self._post("/robot/connect", {"ip": self.robot_ip})

    def arm_enable(self, side: str) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self._post(f"/arms/{side}/enable")

    def arm_joint_positions(self, positions: List[float], side: str) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not positions:
            raise ValueError("positions must not be empty")
        return self._post(f"/arms/{side}/commands/joint_positions", {"positions": positions})

    def arm_set_control_mode(self, side: str, state: int) -> Dict[str, Any]:
        """
        Added for compatibility (was in the other script).
        Endpoint may differ depending on firmware; adjust if needed.
        """
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        state = int(state)
        # Matches robot_control_with_good_action.py behavior:
        return self._post(f"/arms/{side}/commands/control_mode", {"state": state, "imp_type": 1})

    def robot_state(self) -> Dict[str, Any]:
        return self._get("/robot/state")

    # ---------------- Hands ----------------
    def hand_connect(self, side: str, interface: str) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        return self._post(f"/hands/{side}/connect", {"interface": interface})

    def hand_finger_state(self, side: str, finger_id: int) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if finger_id <= 0:
            raise ValueError("finger_id must be positive integer")
        return self._get(f"/hands/{side}/fingers/{int(finger_id)}/state")

    def hand_move_mix(self, side: str, finger_id: int, position: int, speed: int, current: int) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if finger_id <= 0:
            raise ValueError("finger_id must be positive integer")

        return self._post(f"/hands/{side}/commands/move_mix", {
            "id": int(finger_id),
            "position": int(position),
            "speed": int(speed),
            "current": int(current),
        })

    # ✅ original sequential hand_move_6f (kept)
    def hand_move_6f(self, side: str, move: float) -> Dict[str, Any]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        mv = float(move)
        if mv < 0.0 or mv > 1.0:
            raise ValueError("move must be in [0, 1]")

        max_positions = [2648, 1168, 2536, 2748, 2881, 2803]
        min_positions = [2648, 0,    0,    0,    0,    0]

        motor_speeds = [5000, 600, 2000, 2000, 2000, 2000]
        motor_currents = [1000, 120, 200, 100, 500, 1000]

        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        targets: List[int] = []
        for i in range(6):
            pos = _lerp(min_positions[i], max_positions[i], mv)
            targets.append(int(round(pos)))

        out: Dict[str, Any] = {"success": True, "side": side, "move": mv, "results": {}}
        errors: Dict[str, str] = {}

        for idx in range(6):
            fid = idx + 1
            key = f"finger_{fid}"
            try:
                out["results"][key] = self.hand_move_mix(
                    side=side,
                    finger_id=fid,
                    position=targets[idx],
                    speed=motor_speeds[idx],
                    current=motor_currents[idx],
                )
            except Exception as e:
                errors[key] = str(e)
                out["success"] = False
            time.sleep(0.001)

        if errors:
            out["errors"] = errors
        return out

    # ---------------- IK helpers ----------------
    def _ref_joints_from_state(self, state: Dict[str, Any], side: str) -> List[float]:
        side = side.lower()
        key = "left_arm" if side == "left" else "right_arm"
        try:
            joints = state[key]["feedback"]["joint_positions_deg"]
        except Exception as e:
            raise RuntimeError(f"robot_state missing {key}.feedback.joint_positions_deg: {e}")
        if joints is None or len(joints) != 7:
            raise RuntimeError(f"Invalid joint_positions_deg for {side}: {joints}")
        return [float(x) for x in joints]

    def left_world_xyzabc_to_local(self, xyzabc_world: List[float]) -> List[float]:
        if xyzabc_world is None or len(xyzabc_world) != 6:
            raise ValueError("xyzabc_world must be length 6")
        l = [float(v) for v in xyzabc_world]
        l[1] = -l[1]
        l[2] = -l[2]-56
        l[3] = -l[3]
        l[4] = 180.0 - l[4]
        l[5] = -l[5]
        return l
    def right_world_xyzabc_to_local(self, xyzabc_world: List[float]) -> List[float]:
        """
        Convert RIGHT arm world xyzabc -> RIGHT arm local xyzabc (for kinematics lib).

        IMPORTANT:
        I don’t know your exact right-arm world/local convention.
        Below is the “mirror” style transform (same sign flips as left, WITHOUT the +56 z offset),
        matching what your old right IK did (it previously assumed world == local).

        If your right arm also needs an offset like +56, add it here.
        """
        if xyzabc_world is None or len(xyzabc_world) != 6:
            raise ValueError("xyzabc_world must be length 6")

        r = [float(v) for v in xyzabc_world]

        # Option 1 (most conservative): keep right unchanged (world == local)
        # return r

        # Option 2 (mirror-style, like left but without the +56):
        r[2] = r[2]-56
        return r


    # --- UPDATED: accept ref_joints to avoid calling robot_state repeatedly ---
    def ik_left_from_xyzabc(self, xyzabc: List[float], ref_joints: Optional[List[float]] = None) -> List[float]:
        if xyzabc is None or len(xyzabc) != 6:
            raise ValueError("xyzabc must be length 6")

        if ref_joints is None:
            st = self.robot_state()
            ref_joints = self._ref_joints_from_state(st, "left")

        xyzabc_local = self.left_world_xyzabc_to_local(xyzabc)

        T = self._kk.xyzabc_to_mat4x4(xyzabc_local)
        if not T:
            raise RuntimeError("xyzabc_to_mat4x4 failed (left)")

        sp = self._kk.ik(robot_serial=0, pose_mat=T, ref_joints=ref_joints)
        if not sp:
            raise RuntimeError("ik failed (left)")

        return sp.m_Output_RetJoint.to_list()

    # --- UPDATED: accept ref_joints to avoid calling robot_state repeatedly ---
    def ik_right_from_xyzabc(self, xyzabc: List[float], ref_joints: Optional[List[float]] = None) -> List[float]:
        if xyzabc is None or len(xyzabc) != 6:
            raise ValueError("xyzabc must be length 6")

        if ref_joints is None:
            st = self.robot_state()
            ref_joints = self._ref_joints_from_state(st, "right")

        xyzabc_local = self.right_world_xyzabc_to_local(xyzabc)

        T = self._kk.xyzabc_to_mat4x4(xyzabc_local)
        if not T:
            raise RuntimeError("xyzabc_to_mat4x4 failed (right)")

        sp = self._kk.ik(robot_serial=1, pose_mat=T, ref_joints=ref_joints)
        if not sp:
            raise RuntimeError("ik failed (right)")

        return sp.m_Output_RetJoint.to_list()
    def left_delta_local_to_world(self, d: Sequence[float]) -> List[float]:
        # local = [ x, -y, -z-56, -a, 180-b, -c ]
        # => world delta is sign flips only (offsets cancel)
        if d is None or len(d) != 6:
            raise ValueError("left delta must be length 6")
        dx, dy, dz, da, db, dc = [float(v) for v in d]
        return [dx, -dy, -dz, -da, -db, -dc]

    def right_delta_local_to_world(self, d: Sequence[float]) -> List[float]:
        # Your right conversion is currently unclear. If right "local == world", keep identity.
        if d is None or len(d) != 6:
            raise ValueError("right delta must be length 6")
        return [float(v) for v in d]


    # ---------------- Action (UPDATED: optimized "good_action") ----------------
    
    def action(
        self,
        a: Sequence[float],
        ref_left_joints_deg: Optional[Sequence[float]] = None,
        ref_right_joints_deg: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        If ref_*_joints_deg provided: use them as IK ref joints (NO robot_state call).
        If not provided: read robot_state ONCE and extract both.
        """
        if a is None or len(a) != 14:
            raise ValueError("action() expects length-14: [Lxyzabc(6), Rxyzabc(6), Lhand, Rhand]")

        left_xyzabc  = [float(x) for x in a[0:6]]
        right_xyzabc = [float(x) for x in a[6:12]]
        left_hand    = float(a[12])
        right_hand   = float(a[13])

        if not (0.0 <= left_hand <= 1.0):
            raise ValueError(f"left_hand must be in [0,1], got {left_hand}")
        if not (0.0 <= right_hand <= 1.0):
            raise ValueError(f"right_hand must be in [0,1], got {right_hand}")

        out: Dict[str, Any] = {
            "input": {
                "left_xyzabc": left_xyzabc,
                "right_xyzabc": right_xyzabc,
                "left_hand": left_hand,
                "right_hand": right_hand,
            },
            "ik": {},
            "arms": {},
            "hands": {},
            "success": True,
            "errors": {},
        }

        # ---- resolve ref joints (NO robot_state if caller provides both) ----
        try:
            if ref_left_joints_deg is not None and ref_right_joints_deg is not None:
                ref_l = [float(x) for x in ref_left_joints_deg]
                ref_r = [float(x) for x in ref_right_joints_deg]
            else:
                st = self.robot_state()  # ONLY here if missing either
                ref_l = self._ref_joints_from_state(st, "left") if ref_left_joints_deg is None else [float(x) for x in ref_left_joints_deg]
                ref_r = self._ref_joints_from_state(st, "right") if ref_right_joints_deg is None else [float(x) for x in ref_right_joints_deg]

            if len(ref_l) != 7 or len(ref_r) != 7:
                raise ValueError(f"ref joints must be length 7. got left={len(ref_l)}, right={len(ref_r)}")
        except Exception as e:
            out["success"] = False
            out["errors"]["ref_joints"] = str(e)
            return out

        # ---- IK using provided or fetched refs ----
        try:
            lj = self.ik_left_from_xyzabc(left_xyzabc, ref_joints=ref_l)
            rj = self.ik_right_from_xyzabc(right_xyzabc, ref_joints=ref_r)
            out["ik"]["left_joints_deg"] = lj
            out["ik"]["right_joints_deg"] = rj
        except Exception as e:
            out["success"] = False
            out["errors"]["ik"] = str(e)
            return out

        # ---- Move arms ----
        try:
            out["arms"]["left"] = self.arm_joint_positions(out["ik"]["left_joints_deg"], side="left")
            out["arms"]["right"] = self.arm_joint_positions(out["ik"]["right_joints_deg"], side="right")
        except Exception as e:
            out["success"] = False
            out["errors"]["arms"] = str(e)

        # ---- Move hands only if changed ----
        """
        try:
            if self._last_left_hand is None or abs(self._last_left_hand - left_hand) > 1e-9:
                out["hands"]["left"] = self.hand_move_6f("left", left_hand)
                self._last_left_hand = left_hand

            if self._last_right_hand is None or abs(self._last_right_hand - right_hand) > 1e-9:
                out["hands"]["right"] = self.hand_move_6f("right", right_hand)
                self._last_right_hand = right_hand
        except Exception as e:
            out["success"] = False
            out["errors"]["hands"] = str(e)
        """
        # 4) Move hands ONLY if changed (DISABLED TEMP)
        # ---- Move hands: only edge-trigger 0/1; dead-zone does nothing ----
        try:
            def _edge_cmd(v: float) -> Optional[float]:
                v = float(v)
                #if v > 0.9:
                    #return 1.0
                if v < 1.1:
                    return 0.0
                return None  # 0.1~0.9: do nothing

            cmd_l = _edge_cmd(left_hand)
            cmd_r = _edge_cmd(right_hand)

            # LEFT hand
            if cmd_l is None:
                out["hands"]["left"] = {
                    "success": True,
                    "side": "left",
                    "move": left_hand,
                    "skipped": True,
                    "message": "Hand skipped (dead-zone 0.1~0.9).",
                }
            else:
                if self._last_left_hand is None or abs(float(self._last_left_hand) - cmd_l) > 1e-9:
                    out["hands"]["left"] = self.hand_move_6f("left", cmd_l)
                    self._last_left_hand = cmd_l
                else:
                    out["hands"]["left"] = {
                        "success": True,
                        "side": "left",
                        "move": cmd_l,
                        "skipped": True,
                        "message": "Hand skipped (same edge state as last).",
                    }

            # RIGHT hand
            if cmd_r is None:
                out["hands"]["right"] = {
                    "success": True,
                    "side": "right",
                    "move": right_hand,
                    "skipped": True,
                    "message": "Hand skipped (dead-zone 0.1~0.9).",
                }
            else:
                if self._last_right_hand is None or abs(float(self._last_right_hand) - cmd_r) > 1e-9:
                    out["hands"]["right"] = self.hand_move_6f("right", cmd_r)
                    self._last_right_hand = cmd_r
                else:
                    out["hands"]["right"] = {
                        "success": True,
                        "side": "right",
                        "move": cmd_r,
                        "skipped": True,
                        "message": "Hand skipped (same edge state as last).",
                    }

        except Exception as e:
            out["success"] = False
            out["errors"]["hands"] = str(e)


        if not out["errors"]:
            out.pop("errors", None)
        return out


    # ---------------- move_path (ADDED from good_action) ----------------
    def move_path(self, path: List[Sequence[float]], n: int = 100, dt: float = 0.01) -> List[Dict[str, Any]]:
        """
        path: list of 14D actions: [Lxyzabc(6), Rxyzabc(6), Lhand, Rhand]
        For each segment (p[i] -> p[i+1]), generate n points and call self.action().
        Hands (last 2) are NOT interpolated (held constant from the segment start).
        """
        if not path or len(path) < 2:
            return []

        out: List[Dict[str, Any]] = []
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")

        for i in range(len(path) - 1):
            p0 = [float(x) for x in path[i]]
            p1 = [float(x) for x in path[i + 1]]
            if len(p0) != 14 or len(p1) != 14:
                raise ValueError("each waypoint must be length 14")

            # hands: hold from p0 (no interpolation)
            hL, hR = p0[12], p0[13]

            for k in range(1, n + 1):
                t = k / float(n)
                a = [0.0] * 14

                # interpolate ONLY first 12 dims (arms)
                for j in range(12):
                    a[j] = p0[j] + (p1[j] - p0[j]) * t

                a[12], a[13] = hL, hR

                out.append(self.action(a))
                if dt > 0:
                    time.sleep(dt)
        
            time.sleep(5)
        return out

    # ---------------- Step helpers (KEPT but FIXED robustly) ----------------
    def _hand_binary_from_state(self, side: str) -> float:
        """
        Robust version:
        - Reads available finger states (skips fingers whose endpoint fails)
        - Computes ratio over only the fingers that succeeded (denominator matches)
        - If no finger states can be read, falls back to last commanded hand value (if known)
        """
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        # Use only fingers 2..6 (matches your original intention; finger_1 may be constant)
        finger_ids = range(2, 7)  # 2,3,4,5,6

        # Max positions indexed by finger_id-1
        max_positions = [2648, 1168, 2536, 2748, 2881, 2803]

        def _extract_pos(st: Dict[str, Any]) -> Optional[float]:
            candidates = [
                ("feedback", "position"),
                ("feedback", "pos"),
                ("position",),
                ("pos",),
            ]
            for path in candidates:
                cur: Any = st
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok:
                    try:
                        return float(cur)
                    except Exception:
                        return None
            return None

        total = 0.0
        denom = 0.0
        got = 0

        for fid in finger_ids:
            try:
                st = self.hand_finger_state(side, fid)
            except Exception as e:
                # Warn once per (side, fid) to avoid spamming
                key = (side, int(fid))
                if key not in self._warned_finger_state:
                    self._warned_finger_state.add(key)
                    print(f"⚠️ hand_finger_state failed for {side} finger {fid}: {e} (skipping)")
                continue

            pos = _extract_pos(st)
            if pos is None:
                continue

            total += pos
            # denominator matches only included fingers
            denom += float(max_positions[fid - 1])
            got += 1

        if got == 0 or denom <= 1e-9:
            # Fallback: use last commanded hand if available
            last = self._last_left_hand if side == "left" else self._last_right_hand
            if last is None:
                return 0.0
            return 0.0 if float(last) < 0.2 else 1.0

        ratio = total / denom
        return 0.0 if ratio < 0.2 else 1.0

    def _xyzabc_left_world_from_state(self, st: Dict[str, Any]) -> List[float]:
        left_j = self._ref_joints_from_state(st, "left")
        T_left = self._kk.fk(robot_serial=0, joints=left_j)
        if not T_left:
            raise RuntimeError("FK failed for LEFT")
        xyzabc_left_local = self._kk.mat4x4_to_xyzabc(T_left)
        if not xyzabc_left_local:
            raise RuntimeError("mat4x4_to_xyzabc failed for LEFT")

        w = [float(v) for v in xyzabc_left_local]
        w[1] = -w[1]
        w[2] = -w[2]-56
        w[3] = -w[3]
        w[4] = 180.0 - w[4]
        w[5] = -w[5]
        return w

    def _xyzabc_right_world_from_state(self, st: Dict[str, Any]) -> List[float]:
        right_j = self._ref_joints_from_state(st, "right")
        T_right = self._kk.fk(robot_serial=1, joints=right_j)
        if not T_right:
            raise RuntimeError("FK failed for RIGHT")
        xyzabc_right_local = self._kk.mat4x4_to_xyzabc(T_right)
        if not xyzabc_right_local:
            raise RuntimeError("mat4x4_to_xyzabc failed for RIGHT")
        w = [float(v) for v in xyzabc_right_local]
        w[2] = w[2]+56
        print("wwwww")
        print(w)
        return w

    def get_obs(self) -> Dict[str, Any]:
        image = self.capture_image(timeout_ms=1000, save=False)

        st = self.robot_state()
        left_xyzabc_world = self._xyzabc_left_world_from_state(st)
        right_xyzabc_world = self._xyzabc_right_world_from_state(st)

        left_hand = self._hand_binary_from_state("left")
        right_hand = self._hand_binary_from_state("right")

        state14 = left_xyzabc_world + right_xyzabc_world + [left_hand, right_hand]
        print("14141414141")
        print(state14)
        return {"image_main": image, "state": state14}

    # ---------------- REAL step: execute a chunk of delta-actions ----------------
    def step(self, actions: Sequence[Sequence[float]], sleep_s: float = 0.0) -> Dict[str, Any]:
        if actions is None:
            raise ValueError("step(actions): actions is None")
        actions_list = list(actions)
        for i, a in enumerate(actions_list):
            if a is None or len(a) != 14:
                raise ValueError(f"step(actions): actions[{i}] must be length 14")

        obs_initial = self.get_obs()
        base_world = [float(x) for x in obs_initial["state"]]

        for i, delta in enumerate(actions_list):
            d = [float(x) for x in delta]

            if i == 0:
                left_delta_local = d[0:6]
                right_delta_local = d[6:12]

                left_delta_world = self.left_delta_local_to_world(left_delta_local)
                right_delta_world = self.right_delta_local_to_world(right_delta_local)

                d_world = left_delta_world + right_delta_world + d[12:14]
                target = [base_world[j] + d_world[j] for j in range(14)]
            else:
                # subsequent deltas assumed already in world
                target = [base_world[j] + d[j] for j in range(14)]

            # update base for next iteration
            base_world = target.copy()

            # clamp hands
            target[12] = 0.0 if target[12] < 0.0 else (1.0 if target[12] > 1.0 else target[12])
            target[13] = 0.0 if target[13] < 0.0 else (1.0 if target[13] > 1.0 else target[13])

            _ = self.action(target)

            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))

        return self.get_obs()


    def __del__(self):
        self.close_camera()


def pretty(x: Any) -> None:
    print(json.dumps(x, ensure_ascii=False, indent=2))


# ---------------- Example usage ----------------
def main() -> None:
    client = Gr00tClient(host="192.168.50.26", port=5555)
    robot = Robot(host="localhost", port=8080, timeout_s=5.0, robot_ip="192.168.10.190")

    pretty(robot.robot_connect())
    pretty(robot.arm_enable("left"))
    pretty(robot.arm_enable("right"))

    try:
        pretty(robot.arm_set_control_mode("left", 1))
        pretty(robot.arm_set_control_mode("right", 1))
    except Exception as e:
        print("⚠️ arm_set_control_mode failed (endpoint may differ):", e)

    pretty(robot.hand_connect("left", "marvin_arm_left"))
    pretty(robot.hand_connect("right", "marvin_arm_right"))

    pretty(robot._post("/arms/left/commands/limits",  {"speed_ratio": 5, "accel_ratio": 10}))
    pretty(robot._post("/arms/right/commands/limits", {"speed_ratio": 5, "accel_ratio": 10}))

    
    ##pretty(robot.arm_joint_positions([55.27,-46.14,-72.44,-105.35,52.97,14.87,-28.80], side="left"))
    #pretty(robot.arm_joint_positions([], side="right"))
    #time.sleep(10)
    #pretty(robot.hand_move_6f("left", 1.0))
    #pretty(robot.hand_move_6f("left", 0.0))
    #robot.get_obs()

    # Example: delta steps
    """
    deltas = [
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0],
        [10,0,10,0,-1,0, 10,0,-10,0,1,0, 0,0]
    ]

    out = robot.step(deltas, sleep_s=0.1)
    """
    #path = [
        #[600, 0, -300, 0, 90, 0,   600, 0, 300, 0, 90, 0,   0, 0],
       # [600, 0, -300, 0, 90, 0,   600, 0, 300, 0, 90, 0,   0, 0],
    #]
    #pretty(robot.move_path(path, n=10, dt=0.05))

    # Example: move_path (absolute waypoints)
    
    """
    path = [
        [0, 0, -900, 0, 180, 0,   600, 0, 300, 0, 135, 0,   0, 0],
        [0, 0, -900, 0, 180, 0,   600, 0, 300, 0, 135, 0,   0, 0],
    ]
    pretty(robot.move_path(path, n=1, dt=0.5))
    """
    '''
    pretty(robot.arm_joint_positions([0,-45,0,-45,180,0,0], side="left"))
    pretty(robot.arm_joint_positions([0,-45,0,-45,180,0,0], side="right"))
    '''
    pretty(robot.arm_joint_positions([0,-70,0,-20,0,0,0], side="left"))
    pretty(robot.arm_joint_positions([0,-60,0,-60,0,0,0], side="right"))
    
    #time.sleep(10)
    u = [350, 30, -166, 0, 60, 90,  350, 30, 166, 0, 120, 90,   0.0, 0.0]
    #pretty(robot.action(u))
    path = [
        [600, 0, -300, 0, 90, 0,      600, 0, 200, 0, 90, 0,   0, 0],
        [600, -100, -300, 0, 90, 0,   600, 0, 300, 0, 90, 0,   0, 0],
        [600, -200, -300, 0, 90, 0,   600, -100, 300, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      600, -100, 200, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      600, 0, 200, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      500, 0, 200, 0, 90, 0,   0, 0],
        [600, -100, -300, 0, 90, 0,   500, 0, 300, 0, 90, 0,   0, 0],
        [600, -200, -300, 0, 90, 0,   500, -100, 300, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      500, -100, 200, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      500, 0, 200, 0, 90, 0,   0, 0],
        # [600, -100, -300, 0, 90, 0,   500, 0, 300, 0, 90, 0,   0, 0],
        [600, 0, -300, 0, 90, 0,      600, 0, 200, 0, 90, 0,   0, 0],
    ]
    #pretty(robot.move_path(path, n=1000, dt=0.005))
    #pretty(robot.arm_joint_positions([18.41,-64.78,165.06,55.41,-164.6,28.53,4.43], side="right"))
if __name__ == "__main__":
    main()

