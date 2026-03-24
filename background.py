"""
배경 합성 모듈 (MediaPipe Tasks API v0.10+)
ImageSegmenter로 사람을 추출하고 레드카펫 배경으로 합성한다.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "models", "selfie_segmenter.tflite"
)
_BG_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "redcarpet_bg.jpg"
)


def _find_korean_font(size: int) -> ImageFont.FreeTypeFont:
    """시스템에서 한글 지원 폰트를 탐색한다."""
    candidates = [
        "/System/Library/Fonts/AppleMyungjo.ttf",
        "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
        "/Library/Fonts/AppleMyungjo.ttf",
        os.path.expanduser("~/Library/Fonts/NanumMyeongjo.ttf"),
        os.path.expanduser("~/Library/Fonts/NotoSerifKR-Regular.otf"),
        os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"),
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _build_backdrop(width: int, height: int) -> np.ndarray:
    """
    레드카펫 배경 이미지 생성 (BGR numpy array).
      - 상단 45%: 검정 백드롭 + "제 1회 수근영화제" 골드 텍스트 반복 타일
      - 하단 55%: 빨간 레드카펫 + 골드 세로 스트라이프
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    backdrop_h = int(height * 0.45)

    # ── 레드카펫 영역 (하단) — PIL은 RGB 순서 ─────────────
    draw.rectangle([(0, backdrop_h), (width, height)], fill=(160, 0, 20))   # 진한 빨강

    # 골드 세로 스트라이프
    for x in range(0, width, 60):
        draw.rectangle([(x, backdrop_h), (x + 6, height)], fill=(210, 165, 30))  # 골드

    # 카펫 위/아래 골드 테두리
    draw.rectangle([(0, backdrop_h), (width, backdrop_h + 10)], fill=(210, 165, 30))
    draw.rectangle([(0, height - 10), (width, height)], fill=(210, 165, 30))

    # ── 백드롭 영역 (상단) ────────────────────────────────
    draw.rectangle([(0, 0), (width, backdrop_h)], fill=(10, 10, 10))

    title = "제 1회 수근영화제"
    font_size = max(22, width // 28)
    font = _find_korean_font(font_size)

    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tile_w = tw + 40
    tile_h = th + 30

    for row in range(0, backdrop_h, tile_h):
        offset_x = (row // tile_h % 2) * (tile_w // 2)
        for col in range(-tile_w, width + tile_w, tile_w):
            draw.text(
                (col + offset_x, row + 8),
                title,
                font=font,
                fill=(200, 175, 100),
            )

    # 백드롭 하단 골드 테두리
    draw.rectangle([(0, backdrop_h - 12), (width, backdrop_h)], fill=(210, 165, 30))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _overlay_title_on_image(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    레드카펫 배경 이미지 상단에 반투명 검정 띠 + "제 1회 수근영화제" 텍스트 타일을 합성한다.
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    banner_h = int(height * 0.18)  # 상단 18% 영역에 텍스트 배너

    # 반투명 검정 오버레이
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([(0, 0), (width, banner_h)], fill=(0, 0, 0, 160))
    pil_img = Image.alpha_composite(pil_img, overlay)
    draw = ImageDraw.Draw(pil_img)

    title = "제 1회 수근영화제"
    font_size = max(20, width // 32)
    font = _find_korean_font(font_size)

    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tile_w = tw + 50
    tile_h = th + 20

    for row in range(0, banner_h, tile_h):
        offset_x = (row // tile_h % 2) * (tile_w // 2)
        for col in range(-tile_w, width + tile_w, tile_w):
            draw.text(
                (col + offset_x, row + 6),
                title,
                font=font,
                fill=(210, 175, 80, 230),  # 골드
            )

    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


class BackgroundCompositor:
    """사람을 추출해 레드카펫 배경으로 합성"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"세그멘테이션 모델 없음: {_MODEL_PATH}"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,  # selfie_segmenter는 confidence mask 사용
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)

        # 배경 이미지 로드: 커스텀 이미지 우선, 없으면 자동 생성
        if os.path.exists(_BG_IMAGE_PATH):
            print(f"[배경] 커스텀 이미지 사용: {_BG_IMAGE_PATH}")
            bg_img = cv2.imread(_BG_IMAGE_PATH)
            self._backdrop = cv2.resize(bg_img, (width, height))
            # "제 1회 수근영화제" 텍스트 타일 오버레이
            self._backdrop = _overlay_title_on_image(self._backdrop, width, height)
        else:
            print("[배경] 자동 생성 배경 사용 (assets/redcarpet_bg.jpg 없음)")
            self._backdrop = _build_backdrop(width, height)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR 프레임의 배경을 레드카펫으로 교체해 반환한다."""
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._segmenter.segment(mp_image)

        # confidence_masks[1]: 사람 영역 확률 (0~1 float)
        # selfie_segmenter: index 0=배경, 1=사람
        if not result.confidence_masks:
            return frame_bgr

        masks = result.confidence_masks
        # selfie_segmenter: 마스크 1개 = foreground(사람) 확률
        # 2개인 경우 index 1이 사람, index 0이 배경
        if len(masks) > 1:
            person_mask = masks[1].numpy_view().astype(np.float32)
        else:
            person_mask = masks[0].numpy_view().astype(np.float32)

        # (H, W, 1) → (H, W) 로 squeeze
        if person_mask.ndim == 3:
            person_mask = person_mask.squeeze(-1)

        # 부드러운 경계 처리
        person_mask = cv2.GaussianBlur(person_mask, (15, 15), 0)
        mask_3ch = np.stack([person_mask] * 3, axis=-1)

        bg = self._backdrop.copy()
        composite = (frame_bgr * mask_3ch + bg * (1.0 - mask_3ch)).astype(np.uint8)
        return composite

    def close(self):
        self._segmenter.close()
