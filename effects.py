"""
이펙트 모듈
플래시, 폭죽(파티클), 슬로우모션, 바람 블러 이펙트를 관리한다.
"""

import time
import random
from collections import deque

import cv2
import numpy as np

import config


class FlashEffect:
    """화면 가장자리 카메라 플래시 효과"""

    def __init__(self):
        self._alpha = 0.0       # 현재 밝기 (0~1)
        self._active = False
        self._start_time = 0.0

    def trigger(self):
        self._alpha = 1.0
        self._active = True
        self._start_time = time.time()

    def apply(self, frame):
        if not self._active:
            return frame

        elapsed = time.time() - self._start_time
        self._alpha = max(0.0, 1.0 - elapsed / config.FLASH_DURATION)

        if self._alpha <= 0:
            self._active = False
            return frame

        h, w = frame.shape[:2]
        border = int(min(h, w) * 0.12)  # 가장자리 두께

        overlay = frame.copy()

        # 상하좌우 가장자리에 흰색 직사각형 테두리 그리기
        color = (255, 255, 255)
        cv2.rectangle(overlay, (0, 0), (w, border), color, -1)
        cv2.rectangle(overlay, (0, h - border), (w, h), color, -1)
        cv2.rectangle(overlay, (0, 0), (border, h), color, -1)
        cv2.rectangle(overlay, (w - border, 0), (w, h), color, -1)

        return cv2.addWeighted(overlay, self._alpha, frame, 1 - self._alpha, 0)

    @property
    def is_active(self):
        return self._active


class Particle:
    """폭죽 파티클 하나"""

    def __init__(self, x, y):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(4, 14)
        self.x = float(x)
        self.y = float(y)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed - 3  # 약간 위쪽으로 초기 방향
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        self.radius = random.randint(3, 7)
        self.birth = time.time()
        self.lifetime = config.PARTICLE_LIFETIME * random.uniform(0.6, 1.0)

    def update(self):
        self.vy += 0.4  # 중력
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.97  # 공기 저항

    def alpha(self):
        """생존 시간에 따른 투명도 (1→0)"""
        age = time.time() - self.birth
        return max(0.0, 1.0 - age / self.lifetime)

    @property
    def is_dead(self):
        return time.time() - self.birth >= self.lifetime


class FireworksEffect:
    """폭죽 파티클 이펙트"""

    def __init__(self):
        self._particles: list[Particle] = []

    def trigger(self, frame_shape):
        """화면 여러 위치에서 폭죽 발사"""
        h, w = frame_shape[:2]
        # 3~5곳에서 폭죽 터트리기
        for _ in range(random.randint(3, 5)):
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 5, 2 * h // 3)
            for _ in range(random.randint(40, 70)):
                self._particles.append(Particle(cx, cy))

    def apply(self, frame):
        if not self._particles:
            return frame

        overlay = frame.copy()
        alive = []
        for p in self._particles:
            if not p.is_dead:
                p.update()
                a = p.alpha()
                cx, cy = int(p.x), int(p.y)
                # 화면 범위 내에서만 그리기
                if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                    cv2.circle(overlay, (cx, cy), p.radius, p.color, -1)
                alive.append(p)
        self._particles = alive

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    @property
    def is_active(self):
        return bool(self._particles)


class SlowMotionEffect:
    """
    슬로우모션 + 바람(수평 블러) 효과.
    정지 감지 시 최근 프레임 버퍼를 2배 느리게 재생하며 블러를 적용한다.
    """

    def __init__(self, buffer_size=60):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._playback: list = []       # 재생할 프레임 큐
        self._active = False
        self._end_time = 0.0

    def push_frame(self, frame):
        """매 프레임 버퍼에 저장"""
        self._buffer.append(frame.copy())

    def trigger(self):
        if self._active:
            return
        # 버퍼에 있는 프레임을 2배 느리게 재생 (각 프레임 2회 반복)
        frames = list(self._buffer)
        self._playback = [f for f in frames for _ in range(2)]
        self._active = True
        self._end_time = time.time() + config.SLOWMO_DURATION

    def apply(self, frame):
        if not self._active:
            return frame

        if time.time() > self._end_time or not self._playback:
            self._active = False
            self._playback.clear()
            return frame

        # 버퍼에서 꺼내 바람 블러 적용
        playback_frame = self._playback.pop(0)

        # 수평 motion blur (바람 효과)
        blurred = self._wind_blur(playback_frame)
        return blurred

    def _wind_blur(self, frame, shifts=5, strength=0.18):
        """여러 오프셋으로 shift 후 합산 → 수평 모션 블러"""
        result = frame.astype(np.float32) * (1 - strength * shifts)
        for i in range(1, shifts + 1):
            shifted = np.roll(frame, i * 2, axis=1).astype(np.float32)
            result += shifted * strength
        return np.clip(result, 0, 255).astype(np.uint8)

    @property
    def is_active(self):
        return self._active


class EffectManager:
    """모든 이펙트를 통합 관리"""

    def __init__(self):
        self.flash = FlashEffect()
        self.fireworks = FireworksEffect()
        self.slowmo = SlowMotionEffect()

        # 제스처 중복 트리거 방지용 쿨다운
        self._last_one_hand_time = 0.0
        self._last_both_hands_time = 0.0
        self._last_still_time = 0.0
        self._COOLDOWN = 1.5  # 초

    def update(self, gestures: set, frame):
        """제스처 집합을 받아 이펙트 트리거 및 현재 프레임에 이펙트 적용"""
        now = time.time()
        h, w = frame.shape[:2]

        # 슬로우모션 버퍼에 항상 프레임 저장
        if not self.slowmo.is_active:
            self.slowmo.push_frame(frame)

        # 제스처 → 이펙트 트리거
        if 'both_hands' in gestures and now - self._last_both_hands_time > self._COOLDOWN:
            self.flash.trigger()
            self.fireworks.trigger(frame.shape)
            self._last_both_hands_time = now

        elif 'one_hand' in gestures and now - self._last_one_hand_time > self._COOLDOWN:
            self.flash.trigger()
            self._last_one_hand_time = now

        if 'still' in gestures and now - self._last_still_time > self._COOLDOWN + config.STILL_DURATION:
            self.slowmo.trigger()
            self._last_still_time = now

        # 이펙트 적용 순서: 슬로우모션 → 폭죽 → 플래시
        result = self.slowmo.apply(frame)
        result = self.fireworks.apply(result)
        result = self.flash.apply(result)

        return result
