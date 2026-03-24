"""
제스처 감지 모듈 (MediaPipe Tasks API v0.10+)
PoseLandmarker로 손 들기 / 고개 끄덕 / 정지를 감지한다.
"""

import os
import time
from collections import deque
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import config

# 모델 파일 경로
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "models", "pose_landmarker_lite.task"
)

# Pose 랜드마크 인덱스 (Tasks API는 정수 인덱스 사용)
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24


class GestureDetector:
    def __init__(self):
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"모델 파일 없음: {_MODEL_PATH}\n"
                "assets/models/pose_landmarker_lite.task 를 다운받아 주세요."
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

        # 고개 끄덕 감지용 nose y 이력
        self._nose_y_history = deque(maxlen=60)
        self._last_nod_time = 0.0

        # 정지 감지용 pose 이력
        self._pose_history = deque(maxlen=10)
        self._still_start_time = None

    def detect(self, frame_rgb: np.ndarray) -> set:
        """
        RGB numpy 프레임을 받아 감지된 제스처 집합을 반환한다.
        반환값: set of str
          'one_hand', 'both_hands', 'nod', 'still'
        """
        gestures = set()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            self._still_start_time = None
            return gestures

        lm = result.pose_landmarks[0]  # 첫 번째 사람

        # ── 손 들기 감지 ──────────────────────────────────────
        l_shoulder_y = lm[_LEFT_SHOULDER].y
        r_shoulder_y = lm[_RIGHT_SHOULDER].y
        l_wrist_y = lm[_LEFT_WRIST].y
        r_wrist_y = lm[_RIGHT_WRIST].y

        thr = config.HAND_RAISE_THRESHOLD
        l_raised = l_wrist_y < l_shoulder_y - thr
        r_raised = r_wrist_y < r_shoulder_y - thr

        if l_raised and r_raised:
            gestures.add('both_hands')
        elif l_raised or r_raised:
            gestures.add('one_hand')

        # ── 고개 끄덕 감지 ────────────────────────────────────
        nose_y = lm[_NOSE].y
        self._nose_y_history.append(nose_y)
        if self._detect_nod():
            gestures.add('nod')

        # ── 정지 감지 ─────────────────────────────────────────
        key_points = np.array([
            [lm[_LEFT_WRIST].x,  lm[_LEFT_WRIST].y],
            [lm[_RIGHT_WRIST].x, lm[_RIGHT_WRIST].y],
            [lm[_NOSE].x,        lm[_NOSE].y],
            [lm[_LEFT_HIP].x,    lm[_LEFT_HIP].y],
            [lm[_RIGHT_HIP].x,   lm[_RIGHT_HIP].y],
        ])
        self._pose_history.append(key_points)

        if len(self._pose_history) >= 8:
            diffs = [
                np.mean(np.abs(self._pose_history[i] - self._pose_history[i - 1]))
                for i in range(1, len(self._pose_history))
            ]
            avg_movement = np.mean(diffs)

            if avg_movement < config.STILL_THRESHOLD:
                if self._still_start_time is None:
                    self._still_start_time = time.time()
                elif time.time() - self._still_start_time >= config.STILL_DURATION:
                    gestures.add('still')
            else:
                self._still_start_time = None

        return gestures

    def _detect_nod(self) -> bool:
        """nose y 이력으로 끄덕임 이벤트 감지"""
        if len(self._nose_y_history) < 10:
            return False

        recent = list(self._nose_y_history)[-20:]
        median_y = np.median(recent)
        thr = config.NOD_MOVEMENT_THRESHOLD

        directions = []
        for y in recent:
            if y > median_y + thr:
                directions.append('down')
            elif y < median_y - thr:
                directions.append('up')
            else:
                directions.append('neutral')

        filtered = [d for d in directions if d != 'neutral']
        if len(filtered) < 2:
            return False

        transitions = sum(
            1 for i in range(1, len(filtered))
            if filtered[i] != filtered[i - 1]
        )

        now = time.time()
        if transitions >= config.NOD_REQUIRED_CYCLES * 2:
            if now - self._last_nod_time > 1.5:
                self._last_nod_time = now
                self._nose_y_history.clear()
                return True

        return False

    def close(self):
        self._detector.close()
