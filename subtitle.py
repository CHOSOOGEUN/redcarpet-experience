"""
자막 렌더링 모듈
Pillow로 한글 텍스트를 OpenCV 프레임 위에 합성한다.
"""

import time
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import config

# ── 폰트 경로 ─────────────────────────────────────────────────
# 수상 자막: SD Gothic Neo (모던/세련된 산세리프)
# 이름 자막: 명조 계열
_FONT_MODERN = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
_FONT_MYUNGJO = "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf"


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """폰트 파일 로드. 실패 시 기본 폰트 반환."""
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # 폴백 목록
    fallbacks = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
    ]
    for fb in fallbacks:
        if os.path.exists(fb) and fb != path:
            try:
                return ImageFont.truetype(fb, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_text_on_frame(
    frame_bgr: np.ndarray,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont,
    text_color=(255, 255, 255),
    bg_alpha=170,
    pad_x=36,
    pad_y=14,
    full_width_bg=False,   # True: 화면 전체 너비 반투명 띠
) -> np.ndarray:
    """프레임에 텍스트 + 반투명 배경 띠를 그린다 (항상 수평 중앙 정렬)."""
    h, w = frame_bgr.shape[:2]

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw_tmp = ImageDraw.Draw(pil_img)

    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = y

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)

    if full_width_bg:
        # 화면 전체 너비 띠
        ov.rectangle(
            [0, ty - pad_y, w, ty + th + pad_y],
            fill=(0, 0, 0, bg_alpha),
        )
    else:
        ov.rectangle(
            [max(0, tx - pad_x), max(0, ty - pad_y),
             min(w, tx + tw + pad_x), min(h, ty + th + pad_y)],
            fill=(0, 0, 0, bg_alpha),
        )

    pil_img = Image.alpha_composite(pil_img, overlay)
    draw2 = ImageDraw.Draw(pil_img)
    draw2.text((tx, ty), text, font=font, fill=(*text_color, 255))

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


class SubtitleManager:
    """자막 시스템 관리"""

    def __init__(self):
        # 이름 자막: 명조 (클래식한 영화 자막 느낌)
        self._font_name = _load_font(_FONT_MYUNGJO, 38)
        # 수상 발표 대제목: SD Gothic Neo Bold 느낌으로 크게
        self._font_award_big = _load_font(_FONT_MODERN, 62)
        # 수상자 소제목
        self._font_award_sub = _load_font(_FONT_MODERN, 32)

        self._award_index = 0
        self._current_award: str | None = None
        self._award_start_time = 0.0
        self._nod_cooldown = 0.0

    def trigger_award(self):
        """수상 자막을 다음 항목으로 진행"""
        now = time.time()
        if now < self._nod_cooldown:
            return
        if self._award_index < len(config.AWARDS):
            self._current_award = config.AWARDS[self._award_index]
            self._award_index += 1
            self._award_start_time = now
            self._nod_cooldown = now + 1.0

    def render(self, frame: np.ndarray, gestures: set) -> np.ndarray:
        now = time.time()
        h, w = frame.shape[:2]

        if 'nod' in gestures:
            self.trigger_award()

        award_active = (
            self._current_award is not None
            and now - self._award_start_time < config.AWARD_SUBTITLE_DURATION
        )

        # ── 하단 이름/소속 자막 (항상 표시) ──────────────────
        name_text = f"{config.NAME}  |  {config.AFFILIATION}"
        frame = _draw_text_on_frame(
            frame,
            name_text,
            y=h - 72,
            font=self._font_name,
            text_color=(240, 240, 240),
            bg_alpha=180,
            pad_x=44,
            pad_y=16,
            full_width_bg=True,
        )

        # ── 수상 자막 (고개 끄덕 / N 키) ─────────────────────
        if award_active:
            elapsed = now - self._award_start_time
            total = config.AWARD_SUBTITLE_DURATION

            # 페이드인(0.5s) / 유지 / 페이드아웃(0.6s)
            if elapsed < 0.5:
                a = elapsed / 0.5
            elif elapsed > total - 0.6:
                a = (total - elapsed) / 0.6
            else:
                a = 1.0
            a = max(0.0, min(1.0, a))
            alpha = int(255 * a)

            # 중앙 수상 대제목 (골드)
            center_y = h // 2 - 55
            frame = _draw_text_on_frame(
                frame,
                self._current_award,
                y=center_y,
                font=self._font_award_big,
                text_color=(210, 175, 80),       # 골드
                bg_alpha=min(200, alpha),
                pad_x=50,
                pad_y=18,
                full_width_bg=True,
            )

            # 수상자 이름 소제목 (흰색)
            frame = _draw_text_on_frame(
                frame,
                config.NAME,
                y=center_y + 82,
                font=self._font_award_sub,
                text_color=(220, 220, 220),
                bg_alpha=min(160, alpha),
                pad_x=36,
                pad_y=12,
                full_width_bg=True,
            )

        return frame
