from src.filters import HaltDetector, StateUniformDetector


def test_halt_detector_triggers_after_exact_window() -> None:
    detector = HaltDetector(window=3)
    snapshot = ((0, 0, 0, 1), (1, 1, 1, 2))

    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is True


def test_halt_detector_resets_after_change() -> None:
    detector = HaltDetector(window=2)
    snap_a = ((0, 0, 0, 1),)
    snap_b = ((0, 1, 0, 1),)

    assert detector.observe(snap_a) is False
    assert detector.observe(snap_a) is False
    assert detector.observe(snap_b) is False
    assert detector.observe(snap_b) is False
    assert detector.observe(snap_b) is True


def test_halt_detector_window_one() -> None:
    detector = HaltDetector(window=1)
    snapshot = ((0, 0, 0, 1),)

    assert detector.observe(snapshot) is False
    assert detector.observe(snapshot) is True


def test_state_uniform_detector_only_full_uniform() -> None:
    detector = StateUniformDetector()
    assert detector.observe([1, 1, 1, 1]) is True
    assert detector.observe([1, 1, 2, 1]) is False


def test_state_uniform_detector_empty_is_false() -> None:
    detector = StateUniformDetector()
    assert detector.observe([]) is False
