"""
Tests proving the patch detector (a) flags a synthetic adversarial patch and
(b) stays quiet on a clean scene.
Run from the project root:  python -m pytest   (or)   python tests/test_detector.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detector
import synth


def test_clean_scene_not_flagged():
    img = synth.make_scene(size=256, seed=0)
    result, _ = detector.detect(img)
    # a clean scene should not trip the frequency/saturation detector
    assert result["verdict"] == "no patch detected", result


def test_adversarial_patch_flagged():
    img = synth.make_scene(size=256, seed=0)
    patched, _bbox = synth.add_patch(img, size=44, seed=1)
    result, _ = detector.detect(patched)
    # the high-frequency, high-saturation patch should be caught
    assert result["num_flagged_windows"] >= 1, result
    assert result["verdict"] == "ADVERSARIAL PATCH LIKELY", result


if __name__ == "__main__":
    test_clean_scene_not_flagged()
    test_adversarial_patch_flagged()
    print("all tests passed")
