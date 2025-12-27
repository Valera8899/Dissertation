# yolo_proto/src/geometry_adapter.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple
import json

# Імпортуємо структури даних
from domain import Board, BoardMetadata, Component, Joint, save_board_json

# ===========================
# Налаштування
# ===========================

DEFAULT_PX_TO_MM: float = 0.03056

# Мапінг властивостей монтажу (Side, Type) згідно з інструкцією.
# Ключі у верхньому регістрі, бо код нормалізує вхідні дані до .upper().
MOUNTING_SPECS_MAP: Dict[str, Tuple[str, str]] = {
    "ELECTROLYT_CAP_VERT": ("TOP", "SMD"),
    "TO_263_5": ("TOP", "SMD"),
    "SMD_DIODE": ("TOP", "SMD"),
    "SMD_SMALL_2PIN": ("TOP", "SMD"),
    "TOROID_INDUCTOR_HORZ": ("TOP", "THT"),
    "TRIM_POTENTIOMETER": ("TOP", "THT"),
}


# ===========================
# Допоміжні функції
# ===========================

def _bbox_px_to_mm(bbox_px: Sequence[float], px_to_mm: float) -> Tuple[float, float, float, float]:
    """Конвертує bbox [x1, y1, x2, y2] з пікселів у мм."""
    x1, y1, x2, y2 = bbox_px
    return (
        x1 * px_to_mm,
        y1 * px_to_mm,
        x2 * px_to_mm,
        y2 * px_to_mm,
    )


def _estimate_joints_mm(cls_upper: str, bbox_mm: Tuple[float, float, float, float]) -> List[Joint]:
    """
    Визначає координати "робочих точок" (joints).
    """
    x1, y1, x2, y2 = bbox_mm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1

    if "ELECTROLYT" in cls_upper:
        j1 = Joint(x_mm=x1 + 0.25 * width, y_mm=cy)
        j2 = Joint(x_mm=x1 + 0.75 * width, y_mm=cy)
        return [j1, j2]

    return [Joint(x_mm=cx, y_mm=cy)]


def _resolve_mounting_specs(cls_canon: str) -> Tuple[str, str]:
    """
    Визначає (side, mount_type) на основі класу.
    Кидає помилку, якщо клас невідомий (вимога constraints).
    """
    if cls_canon not in MOUNTING_SPECS_MAP:
        raise ValueError(
            f"CRITICAL ERROR: Unknown component class '{cls_canon}'. "
            f"Please update MOUNTING_SPECS_MAP in geometry_adapter.py"
        )
    return MOUNTING_SPECS_MAP[cls_canon]


# ===========================
# Основна логіка
# ===========================

def build_board_from_combined_data(
        combined_detections: List[Dict[str, Any]],
        image_size_px: Tuple[int, int],
        image_path_str: str,
        board_id: str,
        px_to_mm: float = DEFAULT_PX_TO_MM,
        # default_side видалено, бо тепер це визначає мапінг
) -> Board:
    """
    Створює об'єкт Board з уже об'єднаного списку детекцій.
    """

    meta = BoardMetadata(
        board_id=board_id,
        image_path=image_path_str,
        image_size_px=image_size_px,
        px_to_mm=px_to_mm,
    )

    components: List[Component] = []
    per_class_counter: Dict[str, int] = {}

    for det in combined_detections:
        raw_cls = det.get("class_name", f"class_{det.get('class_id', '?')}")
        canon_cls = raw_cls.upper()

        # 1. Визначаємо специфікації монтажу (Side, SMD/THT)
        # Це впаде з помилкою, якщо клас не прописаний у словнику (fail explicitly)
        side, mount_type = _resolve_mounting_specs(canon_cls)

        # 2. Геометрія
        bbox_px = det["bbox_px"]
        bbox_mm = _bbox_px_to_mm(bbox_px, px_to_mm)

        # 3. ID / RefDes
        per_class_counter[canon_cls] = per_class_counter.get(canon_cls, 0) + 1
        idx = per_class_counter[canon_cls]

        comp_id = det.get("id", f"{canon_cls}_{idx}")
        refdes = f"{canon_cls}_{idx}"

        # 4. Точки
        joints = _estimate_joints_mm(canon_cls, bbox_mm)

        comp = Component(
            id=str(comp_id),
            refdes=refdes,
            cls=canon_cls,
            side=side,  # <--- Використовуємо з мапінгу
            mount_type=mount_type,  # <--- Використовуємо з мапінгу
            bbox_mm=bbox_mm,
            joints_mm=joints,
        )
        components.append(comp)

    return Board(meta=meta, components=components)