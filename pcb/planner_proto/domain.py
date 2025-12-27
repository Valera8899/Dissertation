# yolo_proto/src/domain.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json


# ==========
# Dataclasses
# ==========


@dataclass
class Joint:
    """Точка контакту (площадка) в міліметрах відносно системи координат плати."""
    x_mm: float
    y_mm: float


@dataclass
class Component:
    """
    Канонічний опис компонента для планувальника.
    """
    id: str
    refdes: str
    cls: str
    side: str  # "TOP" / "BOTTOM"
    mount_type: str  # "SMD" / "THT"  <--- НОВЕ ПОЛЕ
    bbox_mm: Tuple[float, float, float, float]
    joints_mm: List[Joint]


@dataclass
class BoardMetadata:
    board_id: str
    image_path: Optional[str]
    image_size_px: Tuple[int, int]
    px_to_mm: float


@dataclass
class Board:
    meta: BoardMetadata
    components: List[Component]


# ==========================
# JSON (де)серіалізація Board
# ==========================


def board_to_dict(board: Board) -> Dict[str, Any]:
    """Перетворює Board у словник, готовий до json.dump()."""
    return {
        "board_id": board.meta.board_id,
        "image_path": board.meta.image_path,
        "image_size_px": list(board.meta.image_size_px),
        "px_to_mm": board.meta.px_to_mm,
        "components": [
            {
                "id": comp.id,
                "refdes": comp.refdes,
                "cls": comp.cls,
                "side": comp.side,
                "mount_type": comp.mount_type,  # <--- ЗБЕРІГАЄМО
                "bbox_mm": list(comp.bbox_mm),
                "joints_mm": [
                    {"x_mm": j.x_mm, "y_mm": j.y_mm} for j in comp.joints_mm
                ],
            }
            for comp in board.components
        ],
    }


def board_from_dict(data: Dict[str, Any]) -> Board:
    """Парсить словник (після json.load) у Board."""
    meta = BoardMetadata(
        board_id=data["board_id"],
        image_path=data.get("image_path"),
        image_size_px=tuple(data["image_size_px"]),
        px_to_mm=float(data["px_to_mm"]),
    )

    components: List[Component] = []
    for c in data.get("components", []):
        bbox_list = c["bbox_mm"]
        bbox_mm: Tuple[float, float, float, float] = (
            float(bbox_list[0]),
            float(bbox_list[1]),
            float(bbox_list[2]),
            float(bbox_list[3]),
        )
        joints_raw = c.get("joints_mm", [])
        joints_mm = [
            Joint(x_mm=float(j["x_mm"]), y_mm=float(j["y_mm"]))
            for j in joints_raw
        ]

        comp = Component(
            id=str(c["id"]),
            refdes=str(c.get("refdes", c["id"])),
            cls=str(c["cls"]),
            side=str(c.get("side", "UNKNOWN")),
            mount_type=str(c.get("mount_type", "UNKNOWN")),  # <--- ЧИТАЄМО
            bbox_mm=bbox_mm,
            joints_mm=joints_mm,
        )
        components.append(comp)

    return Board(meta=meta, components=components)


def save_board_json(path: Path | str, board: Board) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = board_to_dict(board)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_board_json(path: Path | str) -> Board:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return board_from_dict(data)