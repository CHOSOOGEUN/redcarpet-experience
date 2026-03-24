"""
사운드 관리 모듈
pygame.mixer로 환호/박수/팡파레 사운드를 재생한다.
사운드 파일이 없으면 numpy로 간단한 tone을 생성해 임시 사용.
"""

import os
import time
import numpy as np

try:
    import pygame
    _pygame_available = True
except ImportError:
    _pygame_available = False
    print("[sound] pygame 없음 — 사운드 비활성화")


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "sounds")


def _generate_tone(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    volume: float = 0.4,
) -> np.ndarray:
    """단순 사인파 tone 생성 (16bit PCM)"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    # 스테레오 (좌우 동일)
    stereo = np.column_stack([wave, wave])
    return stereo


def _generate_fanfare(sample_rate: int = 44100) -> np.ndarray:
    """간단한 팡파레 tone 시퀀스"""
    segments = [
        (523.25, 0.15),  # C5
        (659.25, 0.15),  # E5
        (783.99, 0.15),  # G5
        (1046.5, 0.4),   # C6
        (783.99, 0.1),   # G5
        (1046.5, 0.5),   # C6
    ]
    parts = []
    for freq, dur in segments:
        parts.append(_generate_tone(freq, dur, sample_rate))
        # 짧은 무음 간격
        silence = np.zeros((int(sample_rate * 0.03), 2), dtype=np.int16)
        parts.append(silence)
    return np.concatenate(parts, axis=0)


def _generate_applause(sample_rate: int = 44100, duration: float = 2.5) -> np.ndarray:
    """박수/환호 흉내 — 랜덤 노이즈 + 엔벨로프"""
    n = int(sample_rate * duration)
    noise = np.random.uniform(-1, 1, n)

    # 엔벨로프: 빠른 attack + 느린 decay
    env = np.ones(n)
    attack = int(sample_rate * 0.05)
    decay_start = int(sample_rate * 0.8)
    env[:attack] = np.linspace(0, 1, attack)
    env[decay_start:] = np.linspace(1, 0, n - decay_start)

    wave = (noise * env * 0.35 * 32767).astype(np.int16)
    return np.column_stack([wave, wave])


def _save_wav(path: str, data: np.ndarray, sample_rate: int = 44100):
    """numpy int16 스테레오 배열을 WAV 파일로 저장"""
    import wave
    import struct

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


class SoundManager:
    """사운드 트리거 및 재생 관리"""

    def __init__(self):
        self._enabled = _pygame_available
        self._sounds: dict = {}

        if not self._enabled:
            return

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        applause_path = os.path.join(ASSETS_DIR, "applause.wav")
        fanfare_path = os.path.join(ASSETS_DIR, "fanfare.wav")

        # 사운드 파일 없으면 생성
        if not os.path.exists(applause_path):
            print("[sound] applause.wav 생성 중...")
            _save_wav(applause_path, _generate_applause())

        if not os.path.exists(fanfare_path):
            print("[sound] fanfare.wav 생성 중...")
            _save_wav(fanfare_path, _generate_fanfare())

        try:
            self._sounds['applause'] = pygame.mixer.Sound(applause_path)
            self._sounds['fanfare'] = pygame.mixer.Sound(fanfare_path)
        except Exception as e:
            print(f"[sound] 로드 실패: {e}")
            self._enabled = False

        # 쿨다운 타이머
        self._last_applause = 0.0
        self._last_fanfare = 0.0
        self._APPLAUSE_COOLDOWN = 3.0
        self._FANFARE_COOLDOWN = 5.0

    def play(self, gestures: set):
        """제스처에 따라 사운드 재생"""
        if not self._enabled:
            return

        now = time.time()

        # 손 들기 → 박수
        if ('one_hand' in gestures or 'both_hands' in gestures) \
                and now - self._last_applause > self._APPLAUSE_COOLDOWN:
            self._play('applause')
            self._last_applause = now

    def _play(self, name: str):
        sound = self._sounds.get(name)
        if sound:
            sound.play()

    def stop_all(self):
        if self._enabled:
            pygame.mixer.stop()

    def quit(self):
        if self._enabled:
            pygame.mixer.quit()
