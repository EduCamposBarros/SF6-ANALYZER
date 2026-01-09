import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(ROOT, "output", "tuning_report.json")
OUT = os.path.join(ROOT, "vision", "tuned_state_config.py")

def main():
    """
    Lê `output/tuning_report.json` e persiste a melhor configuração em
    `vision/tuned_state_config.py` para uso posterior pelo detector.
    """

    if not os.path.exists(REPORT):
        print("No tuning report found at", REPORT)
        return
    with open(REPORT, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results in tuning report")
        return

    top = results[0].get("config", {})
    # Map keys to StateDetectorConfig fields
    mapping = {
        "area_attack_threshold": top.get("area_t"),
        "area_attack_fallback": top.get("area_fb"),
        "mean_color_block_threshold": top.get("mean_c"),
        "jump_cy_delta": top.get("jump_delta"),
        "drive_cx_delta_factor": top.get("drive_factor"),
        "motion_thresh": top.get("motion_thresh"),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Auto-generated tuned state detector config\n")
        f.write("from vision.state_detection import StateDetectorConfig\n\n")
        f.write("def get_default_config():\n")
        f.write("    return StateDetectorConfig(\\n")
        for k, v in mapping.items():
            if v is None:
                continue
            # write numeric value (float or int)
            if isinstance(v, float):
                f.write(f"        {k}={v},\\n")
            else:
                f.write(f"        {k}={v},\\n")
        f.write("    )\\n")

    print("Wrote tuned config to", OUT)


if __name__ == '__main__':
    main()
