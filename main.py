"""
수근영화제 레드카펫 입장 체험 앱
실행: python main.py

제스처 → 이펙트 매핑:
  한 손 들기   → 화면 가장자리 플래시 번쩍
  양손 들기    → 플래시 + 폭죽 파티클
  고개 끄덕    → 수상 발표 자막 순서대로 등장
  3초 이상 정지 → 슬로우모션 + 바람 블러 효과

종료: q 키 또는 ESC
"""

import re
import subprocess
import sys
import cv2
import numpy as np

import config
from background import BackgroundCompositor
from gesture import GestureDetector
from effects import EffectManager
from subtitle import SubtitleManager
from sound_manager import SoundManager


def find_builtin_camera_index() -> int:
    """AVFoundation 디바이스 목록에서 내장 FaceTime 카메라 인덱스를 반환."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stderr.splitlines():
            # 예: [AVFoundation indev @ 0x...] [0] FaceTime HD Camera
            match = re.search(r'\[(\d+)\]\s+FaceTime', line, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                print(f"[정보] 내장 FaceTime 카메라 인덱스: {idx}")
                return idx
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    print("[경고] ffmpeg으로 카메라 목록을 확인할 수 없어 index 0 사용")
    return 0


def build_ui_overlay(frame: np.ndarray) -> np.ndarray:
    """디버그용 우측 상단 힌트 텍스트 표시 (영문만 OpenCV로)"""
    hints = [
        "Q / ESC : Quit",
        "N : Award (수상)",
        "F : Flash",
        "B : Fireworks",
        "One hand up : Flash",
        "Both hands : Fireworks",
        "Nod head : Award",
    ]
    y = 30
    for hint in hints:
        cv2.putText(
            frame, hint,
            (frame.shape[1] - 300, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        y += 22
    return frame


def main():
    # ── 웹캠 초기화 ───────────────────────────────────────────
    # macOS AVFoundation 백엔드로 내장 FaceTime 카메라 사용 (아이폰 Continuity Camera 제외)
    cam_idx = find_builtin_camera_index()
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다. 카메라 연결을 확인해 주세요.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # 카메라 웜업 (macOS는 첫 몇 프레임 읽기 실패 가능)
    for _ in range(10):
        cap.read()

    # 실제 해상도 확인
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[정보] 웹캠 해상도: {W}x{H}")

    # ── 모듈 초기화 ───────────────────────────────────────────
    print("[정보] 배경 초기화 중...")
    bg = BackgroundCompositor(W, H)

    print("[정보] 제스처 감지기 초기화 중...")
    gesture_detector = GestureDetector()

    print("[정보] 이펙트 초기화...")
    effect_mgr = EffectManager()

    print("[정보] 자막 초기화...")
    subtitle_mgr = SubtitleManager()

    print("[정보] 사운드 초기화...")
    sound_mgr = SoundManager()

    print("[준비 완료] 수근영화제 레드카펫에 오신 것을 환영합니다! 🎬")
    print("  종료: q 또는 ESC\n")

    window_name = "🎬 수근영화제 레드카펫"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, W, H)

    # ── 메인 루프 ─────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전 (거울 모드 — 더 자연스럽게)
        frame = cv2.flip(frame, 1)

        # 1. 배경 합성 (레드카펫 교체)
        frame = bg.apply(frame)

        # 2. 제스처 감지 (RGB로 변환 후 처리)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gestures = gesture_detector.detect(frame_rgb)

        # 3. 이펙트 적용 (플래시, 폭죽, 슬로우모션)
        frame = effect_mgr.update(gestures, frame)

        # 4. 자막 렌더링 (이름, 수상 발표)
        frame = subtitle_mgr.render(frame, gestures)

        # 5. 사운드 재생
        sound_mgr.play(gestures)

        # 6. UI 힌트 오버레이
        frame = build_ui_overlay(frame)

        # 7. 화면 출력
        cv2.imshow(window_name, frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # q, Q, ESC → 종료
            break
        elif key in (ord('n'), ord('N')):    # n → 수상 자막 강제 트리거
            gestures_manual = {'nod'}
            subtitle_mgr.render(frame, gestures_manual)
            subtitle_mgr.trigger_award()
            sound_mgr.play(gestures_manual)
        elif key in (ord('f'), ord('F')):    # f → 플래시 강제 트리거
            effect_mgr.flash.trigger()
        elif key in (ord('b'), ord('B')):    # b → 폭죽 강제 트리거
            effect_mgr.flash.trigger()
            effect_mgr.fireworks.trigger(frame.shape)

    # ── 정리 ──────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    gesture_detector.close()
    bg.close()
    sound_mgr.quit()
    print("[종료] 수근영화제 레드카펫을 마칩니다.")


if __name__ == "__main__":
    main()
