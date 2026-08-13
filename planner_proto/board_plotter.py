#рендер 2D-схеми для демо
from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt

from domain import Board, Component, load_board_json


# Типи для зручності
BoardLike = Union[Board, Mapping[str, Any]]
PathLike = Union[str, Path]


# Просте мапування класів компонентів → форма для відмалювання
CLASS_SHAPES: Dict[str, str] = {
    "electrolyt_cap_vert": "circle",        # електроліт: коло
    "toroid_inductor_horz": "ring",         # тороїд: кільце
    "TO_263_5": "rect",                     # силовий IC: прямокутник
    "smd_diode": "rect",                    # діод: прямокутник
    "smd_small_2pin": "rect",               # дрібний SMD: прямокутник
    "trim_potentiometer": "rect",           # потенціометр: прямокутник
}


def _as_board_dict(board: BoardLike) -> Dict[str, Any]:
    """
    Нормалізує Board / dict до чистого dict.
    """
    if isinstance(board, Mapping):
        return dict(board)

    if isinstance(board, Board) or is_dataclass(board):
        # Перетворюємо dataclass Board → dict рекурсивно
        from dataclasses import asdict

        return asdict(board)

    raise TypeError(f"Непідтримуваний тип board: {type(board)!r}")


def _extract_components(board_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Дістає список компонентів у вигляді dict.
    Очікується поле `components` як список або подібна структура.
    """
    comps = board_dict.get("components", [])
    out: List[Dict[str, Any]] = []

    for c in comps:
        if isinstance(c, Mapping):
            out.append(dict(c))
        elif is_dataclass(c):
            from dataclasses import asdict

            out.append(asdict(c))
        else:
            raise TypeError(f"Непідтримуваний тип component: {type(c)!r}")

    return out


def _get_board_size_mm(board_dict: Dict[str, Any]) -> Tuple[float, float]:
    """
    Повертає (width_mm, height_mm) на основі:
    - size_mm, якщо є,
    - або image_size_px + px_to_mm.
    """
    if "size_mm" in board_dict and board_dict["size_mm"]:
        w_mm, h_mm = board_dict["size_mm"]
        return float(w_mm), float(h_mm)

    img_w_px = img_h_px = None
    if "image_size_px" in board_dict and board_dict["image_size_px"]:
        img_w_px, img_h_px = board_dict["image_size_px"]

    px_to_mm = board_dict.get("px_to_mm")
    if img_w_px is not None and img_h_px is not None and px_to_mm:
        w_mm = float(img_w_px) * float(px_to_mm)
        h_mm = float(img_h_px) * float(px_to_mm)
        return w_mm, h_mm

    # fallback: щось адекватне за замовчуванням
    return 100.0, 50.0


def _extract_bbox_and_center_mm(comp: Dict[str, Any]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    """
    Дістає bbox_mm та center_mm з компонента.
    Якщо center_mm немає — рахує як центроїд bbox.
    """
    bbox_mm_raw = comp.get("bbox_mm")
    if not bbox_mm_raw or len(bbox_mm_raw) != 4:
        raise ValueError(f"Компонент не має коректного bbox_mm: {comp!r}")

    x1, y1, x2, y2 = map(float, bbox_mm_raw)
    cx = cy = None

    center_mm_raw = comp.get("center_mm")
    if center_mm_raw and len(center_mm_raw) == 2:
        cx, cy = map(float, center_mm_raw)
    else:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

    return (x1, y1, x2, y2), (cx, cy)


def plot_board(
    board: BoardLike,
    out_path: PathLike,
    dpi: int = 200,
    title: Optional[str] = None,
) -> Path:
    """
    Малює 2D-схему плати на основі canonical board JSON / Board.

    - Прямокутник плати (у мм).
    - Фігури компонентів:
        * електроліт — коло
        * тороїд — кільце
        * решта — прямокутники
    - Центри компонентів + підписи (id).

    :param board: Board або dict (як з load_board_json)
    :param out_path: шлях до PNG/SVG для збереження
    :param dpi: роздільна здатність зображення
    :param title: заголовок графіка (опціонально)
    :return: фактичний Path до збереженого файлу
    """
    board_dict = _as_board_dict(board)
    components = _extract_components(board_dict)
    w_mm, h_mm = _get_board_size_mm(board_dict)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(w_mm / 10.0, h_mm / 10.0), dpi=dpi)

    # Прямокутник плати
    board_rect = plt.Rectangle((0, 0), w_mm, h_mm, fill=False, linewidth=1.5)
    ax.add_patch(board_rect)

    # Малюємо компоненти
    for comp in components:
        comp_id = str(comp.get("id", ""))
        cls_name = str(comp.get("class_name", "unknown"))
        side = str(comp.get("side", "TOP")).upper()

        bbox_mm, center_mm = _extract_bbox_and_center_mm(comp)
        x1, y1, x2, y2 = bbox_mm
        cx, cy = center_mm
        width = x2 - x1
        height = y2 - y1

        shape = CLASS_SHAPES.get(cls_name, "rect")

        # базовий стиль
        edgecolor = "black"
        linewidth = 1.0
        linestyle = "-" if side == "TOP" else "--"

        if shape == "circle":
            radius = max(width, height) / 2.0
            circle = plt.Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            ax.add_patch(circle)

        elif shape == "ring":
            # зовнішнє коло
            outer_r = max(width, height) / 2.0
            inner_r = outer_r * 0.6
            outer = plt.Circle(
                (cx, cy),
                outer_r,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            inner = plt.Circle(
                (cx, cy),
                inner_r,
                fill=False,
                edgecolor=edgecolor,
                linewidth=0.7,
                linestyle=linestyle,
            )
            ax.add_patch(outer)
            ax.add_patch(inner)

        else:
            # прямокутник
            rect = plt.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            ax.add_patch(rect)

        # центр компонента
        ax.scatter([cx], [cy], s=8)

        # підпис
        label = comp_id or cls_name
        ax.text(
            cx,
            cy,
            label,
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)  # інвертуємо вісь Y, щоб збігалось з образом плати
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("мм по X")
    ax.set_ylabel("мм по Y")
    if title is None:
        title = str(board_dict.get("board_id", "PCB board layout"))
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def plot_board_from_json(
    json_path: PathLike,
    out_path: Optional[PathLike] = None,
    dpi: int = 200,
) -> Path:
    """
    Шорткат: прочитати canonical board JSON і одразу намалювати PNG.

    :param json_path: шлях до board.json
    :param out_path: куди зберегти PNG; якщо None — поруч із JSON
    :param dpi: роздільна здатність
    :return: Path до збереженого файлу
    """
    json_path = Path(json_path)
    board = load_board_json(json_path)

    if out_path is None:
        out_path = json_path.with_suffix("").with_name(json_path.stem + "_layout.png")

    return plot_board(board, out_path=out_path, dpi=dpi)


if __name__ == "__main__":
    # Простий ручний запуск для дебагу:
    # python board_plotter.py path/to/board.json
    import sys

    if len(sys.argv) < 2:
        print("Використання: python board_plotter.py path/to/board.json [out.png]")
        sys.exit(1)

    in_json = sys.argv[1]
    out_img = sys.argv[2] if len(sys.argv) >= 3 else None

    result_path = plot_board_from_json(in_json, out_img)
    print(f"Збережено візуалізацію плати до: {result_path}")
