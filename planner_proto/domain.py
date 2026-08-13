# domain.py
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

    Поля:
      - id: унікальний id всередині плати (може співпадати з refdes, але не обов'язково).
      - refdes: "C1", "R5", "U3" або штучний типу "ELECTROLYT_CAP_VERT_1".
      - cls: канонічне ім'я класу, з яким працюємо в планувальнику
             (наприклад, "ELECTROLYT_CAP_VERT", "SMD_DIODE", "TOROID_INDUCTOR").
      - side: "TOP" або "BOTTOM" (або "UNKNOWN", якщо не знаємо).
      - bbox_mm: (x1_mm, y1_mm, x2_mm, y2_mm) — прямокутник корпусу в мм
                 у координатах плати.
      - joints_mm: список точок Joint, де треба прогрівати / братися пінцетом.
    """
    id: str
    refdes: str
    cls: str
    side: str
    bbox_mm: Tuple[float, float, float, float]
    joints_mm: List[Joint]


@dataclass
class BoardMetadata:
    """
    Метадані про плату.

    - board_id: логічний ідентифікатор ("XL4015_demo").
    - image_path: шлях до вихідного зображення (відносно кореня проекту).
    - image_size_px: (width_px, height_px).
    - px_to_mm: масштаб, скільки мм відповідає одному пікселю.
    """
    board_id: str
    image_path: Optional[str]
    image_size_px: Tuple[int, int]
    px_to_mm: float


@dataclass
class Board:
    """
    Повний опис плати для планувальника:
      - meta: BoardMetadata з загальною інфою.
      - components: список Component у мм.
    """
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
            bbox_mm=bbox_mm,
            joints_mm=joints_mm,
        )
        components.append(comp)

    return Board(meta=meta, components=components)


def save_board_json(path: Path | str, board: Board) -> None:
    """Зберігає Board у JSON-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = board_to_dict(board)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_board_json(path: Path | str) -> Board:
    """Завантажує Board з JSON-файлу."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return board_from_dict(data)

