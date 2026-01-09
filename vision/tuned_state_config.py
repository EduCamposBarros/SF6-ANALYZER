# Auto-generated tuned state detector config
from vision.state_detection import StateDetectorConfig

def get_default_config():
    return StateDetectorConfig(
        area_attack_threshold=15000,
        area_attack_fallback=2500,
        mean_color_block_threshold=30,
        jump_cy_delta=0.04,
        drive_cx_delta_factor=0.08,
        motion_thresh=1.5,
    )