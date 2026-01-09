from analysis.events import detect_events
from models.structures import FrameData, Event


def make_frame(frame_id=0, life_p1=1000, life_p2=1000, p1_state="neutral", p2_state="neutral"):
    return FrameData(
        frame_id=frame_id,
        timestamp=float(frame_id) / 60.0,
        p1_state=p1_state,
        p2_state=p2_state,
        p1_can_act=True,
        p2_can_act=True,
        p1_bbox=(0, 0, 10, 10),
        p2_bbox=(100, 0, 110, 10),
        life_p1=life_p1,
        life_p2=life_p2,
    )


def test_hit_p2_and_drive_impact():
    prev = make_frame(frame_id=1, life_p1=1000, life_p2=1000, p1_state="drive")
    cur = make_frame(frame_id=2, life_p1=1000, life_p2=950, p1_state="drive")

    ev = detect_events(cur, prev)
    types = [e.type for e in ev]

    assert "hit" in types
    assert "drive_impact" in types


def test_hit_p1_detected_independently():
    prev = make_frame(frame_id=10, life_p1=900, life_p2=1000, p2_state="neutral")
    cur = make_frame(frame_id=11, life_p1=850, life_p2=1000, p2_state="neutral")

    ev = detect_events(cur, prev)
    assert any(e.type == "hit" and e.attacker == "P2" for e in ev)


def test_jump_start_and_land_always_detected():
    prev = make_frame(frame_id=20, p1_state="neutral", p2_state="neutral")
    cur = make_frame(frame_id=21, p1_state="jump", p2_state="neutral")
    ev = detect_events(cur, prev)
    assert any(e.type == "jump_start" and e.attacker == "P1" for e in ev)

    prev2 = make_frame(frame_id=22, p1_state="jump", p2_state="neutral")
    cur2 = make_frame(frame_id=23, p1_state="neutral", p2_state="neutral")
    ev2 = detect_events(cur2, prev2)
    assert any(e.type == "jump_land" and e.attacker == "P1" for e in ev2)


def test_attack_start_and_block():
    prev = make_frame(frame_id=30, p1_state="neutral", p2_state="neutral")
    cur = make_frame(frame_id=31, p1_state="attack_active", p2_state="block")
    ev = detect_events(cur, prev)
    types = [e.type for e in ev]
    assert "attack_start" in types
    assert "block" in types
