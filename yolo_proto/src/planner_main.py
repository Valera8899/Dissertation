"""
yolo_proto/src/planner_main.py
"""

from pathlib import Path
import sys
import csv

# --- ІМПОРТИ ДЛЯ ВІЗУАЛІЗАЦІЇ ---
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --------------------------------

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.append(str(SRC_DIR))

import detect_components
import yolo_io
import geometry_adapter
import ops_builder
import planning_core
from domain import save_board_json

# ================= НАЛАШТУВАННЯ =================
RUN_NAME = "final_demo_v1"

DATA_DIR = PROJECT_ROOT / "yolo_proto" / "data" / "caps_xl_test"
INPUT_IMAGE = DATA_DIR / "xl_test_1.jpg"
CLASSES_TXT = DATA_DIR / "classes.txt"

STEP_1_OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / RUN_NAME
STEP_1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_1_RESULT_IMG = STEP_1_OUTPUT_DIR / INPUT_IMAGE.name
FINAL_COMBINED_IMG = STEP_1_OUTPUT_DIR / "final_combined_result.jpg"
FINAL_BOARD_JSON = STEP_1_OUTPUT_DIR / "board.json"
FINAL_STEPS_CSV = STEP_1_OUTPUT_DIR / "steps.csv"
FINAL_ROUTE_IMG = STEP_1_OUTPUT_DIR / "route_trajectory.png"  # <--- Новий файл


def get_sim_environment():
    cfg = planning_core.MachineConfig(
        travel_speed_mm_s=200.0,
        z_raise_time_s=0.5,
        settle_time_s=0.2,
        side_switch_time_s=5.0
    )
    nozzles = {
        "TIP_M": planning_core.Nozzle(planning_core.ToolModule.HOT_TIP, "TIP_M", 2.0, 2.0,
                                      (planning_core.Side.TOP, planning_core.Side.BOTTOM)),
        "HOT_AIR_4": planning_core.Nozzle(planning_core.ToolModule.HOT_AIR_NOZZLE, "HOT_AIR_4", 4.0, 10.0,
                                          (planning_core.Side.TOP, planning_core.Side.BOTTOM)),
        "GRIPPER": planning_core.Nozzle(planning_core.ToolModule.MECH_GRIPPER, "GRIPPER", 3.0, 15.0,
                                        (planning_core.Side.TOP,))
    }
    return cfg, nozzles


# --- ФУНКЦІЯ 1: Показати фінальне фото плати ---
def show_final_result(image_path: Path):
    if not image_path.exists():
        return
    try:
        img = mpimg.imread(str(image_path))
        plt.figure("Final Detection", figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        #plt.title(f"DETECTION RESULT: {image_path.name}")
        print("\n[INFO] Вікно з фото плати відкрито. Закрийте його для продовження...")
        plt.show()
    except Exception as e:
        print(f"[ERROR] Viz error: {e}")


# --- ФУНКЦІЯ 2: Намалювати, зберегти і показати маршрут ---
def visualize_route(result: planning_core.PlanResult, path_png: Path):
    """
    Малює траєкторію руху інструмента.
    """
    # 1. Фільтруємо точки (ігноруємо RESCAN, бо це віртуальна подія)
    steps_to_draw = [s for s in result.steps if not s.op_id.startswith("RESCAN_")]

    if not steps_to_draw:
        print("Немає кроків для візуалізації.")
        return

    # Отримуємо координати
    pts = [(s.visit_point[0], s.visit_point[1]) for s in steps_to_draw]
    xs, ys = zip(*pts)

    # 2. Налаштовуємо графік
    plt.figure("Route Trajectory", figsize=(8, 7))

    # Малюємо лінію і точки
    plt.plot(xs, ys, marker="o", linestyle="-", color="steelblue", alpha=0.7, label="Шлях маніпулятора")

    # Нумеруємо кроки
    for i, (x, y) in enumerate(pts):
        plt.text(x, y, str(i + 1), fontsize=9, color="red", fontweight="bold")

    # 3. Маркери баз (беремо константи з planning_core)
    bases = [
        (planning_core.BASE_TOP, "BASE_TOP", "green"),
        (planning_core.BASE_BOTTOM, "BASE_BOTTOM", "green"),
        (planning_core.TOOL_STATION, "TOOL_STATION", "orange")
    ]

    for (bx, by), label, color in bases:
        plt.scatter([bx], [by], c=color, s=150, marker="s", label=label, edgecolors="black")
        plt.text(bx, by - 5, label, fontsize=8, ha='center')

    # Оформлення
    plt.title("Розрахований маршрут (Setup & Side-Switch враховані)")
    plt.xlabel("X, мм")
    plt.ylabel("Y, мм")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 4. Зберігаємо
    plt.savefig(path_png, dpi=140)
    print(f"✅ Графік маршруту збережено: {path_png}")

    # 5. Показуємо
    #print("[INFO] Вікно з графіком маршруту відкрито. Закрийте його для завершення...")
    #plt.show()


def main():
    print(f"=== ЗАПУСК ПАЙПЛАЙНУ: {RUN_NAME} ===")

    # 1. AI
    print("--- 1. YOLO ---")
    payload_yolo = detect_components.run_single_image(
        image_path=INPUT_IMAGE,
        vis_subdir=RUN_NAME,
        save_vis=True,
        conf=0.15
    )

    # 2. Manual
    print("--- 2. Manual ---")
    payload_manual = yolo_io.get_single_sample(INPUT_IMAGE, DATA_DIR, CLASSES_TXT)

    # Viz Combined
    all_detections = payload_yolo["detections"] + payload_manual["detections"]
    yolo_io.visualize_detections({"image_path": str(INPUT_IMAGE), "detections": all_detections}, FINAL_COMBINED_IMG,
                                 None)

    # 3. Geometry
    print("--- 3. Geometry ---")
    w, h = payload_yolo["image_size_px"]
    board_model = geometry_adapter.build_board_from_combined_data(all_detections, (w, h), str(INPUT_IMAGE), "XL_Run",
                                                                  0.03056)
    save_board_json(FINAL_BOARD_JSON, board_model)

    # 4. Ops Builder
    print("--- 4. Ops Builder ---")
    ops = ops_builder.ops_from_canonical_json(FINAL_BOARD_JSON)
    print(f"Сформовано {len(ops)} композитних задач.")

    # 5. Planning
    print("--- 5. Planning ---")
    cfg, nozzles = get_sim_environment()
    plan_result = planning_core.plan_full_job(ops, cfg, nozzles, max_passes=1)

    print(f"План готовий. Час: {plan_result.total_time_s:.2f}с")

    # 6. CSV Export
    print("--- 6. Exporting CSV ---")
    with open(FINAL_STEPS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["StepID", "OpID", "Component", "Nozzle", "Action", "Side", "X_mm", "Y_mm", "Duration_s"])

        step_counter = 1
        for step in plan_result.steps:
            # Службові кроки
            if step.op_id.startswith("MOVE") or step.op_id.startswith("SETUP"):
                writer.writerow(
                    [step_counter, step.op_id, "-", step.nozzle, "MOVE/SETUP", "-", f"{step.visit_point[0]:.1f}",
                     f"{step.visit_point[1]:.1f}", f"{step.travel_time_s + step.setup_time_s:.1f}"])
                step_counter += 1
                continue

            # Розпарсинг ID
            if "::" in step.op_id:
                cls_name, comp_id = step.op_id.split("::", 1)
            else:
                cls_name, comp_id = "UNKNOWN", step.op_id

            x_str = f"{step.visit_point[0]:.1f}"
            y_str = f"{step.visit_point[1]:.1f}"

            # Розділення нагріву/витягування
            if step.nozzle in ["TIP_M", "HOT_AIR_4"]:
                heat_time = 3.0
                action_name = "HEAT_JOINT" if step.nozzle == "TIP_M" else "HOT_AIR_REFLOW"

                writer.writerow([
                    step_counter, f"HEAT_{comp_id}", cls_name, step.nozzle, action_name,
                    "BOTTOM" if "TOROID" in cls_name or "POTENTIOMETER" in cls_name else "TOP",
                    x_str, y_str, heat_time
                ])
                step_counter += 1

                extract_time = step.op_time_s - heat_time
                if extract_time < 0: extract_time = 1.0

                writer.writerow([
                    step_counter, f"EXTRACT_{comp_id}", cls_name, "GRIPPER", "PULL_COMPONENT",
                    "TOP", x_str, y_str, f"{extract_time:.1f}"
                ])
                step_counter += 1
            else:
                writer.writerow([step_counter, step.op_id, cls_name, step.nozzle, "GENERIC_ACTION", "TOP", x_str, y_str,
                                 f"{step.op_time_s:.1f}"])
                step_counter += 1

    print(f"✅ CSV збережено: {FINAL_STEPS_CSV}")

    # --- 7. ВІЗУАЛІЗАЦІЯ ---
    print("\n--- 7. Visualization ---")

    # 1. Спочатку показуємо результат розпізнавання (Фото)
    show_final_result(FINAL_COMBINED_IMG)

    # 2. Потім показуємо графік маршруту (Схема)
    visualize_route(plan_result, FINAL_ROUTE_IMG)

    print("\n=== ПАЙПЛАЙН ЗАВЕРШЕНО ===")


if __name__ == "__main__":
    main()