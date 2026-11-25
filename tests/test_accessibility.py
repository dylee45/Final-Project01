import unittest

from src.accessibility import evaluate_accessibility


def make_detection(class_name, bbox):
    return {
        "class_name": class_name,
        "confidence": 0.95,
        "bbox": bbox,
    }


class TestAccessibility(unittest.TestCase):

    def setUp(self):
        self.ramp = make_detection(
            "ramp", [100, 100, 400, 400]
        )

        self.slope_limit = 5.0

        # 장애물 탐지를 지원하는 가상의 모델 정보
        self.model_classes = {
            "ramp_supported": True,
            "stairs_supported": True,
            "obstacle_supported": True,
            "classes": ["ramp", "stairs", "obstacle"]
        }

    def evaluate(self, detections, angle):
        return evaluate_accessibility(
            detections=detections,
            measurements=[{"angle": angle}],
            slope_limit=self.slope_limit,
            model_classes=self.model_classes,
        )

    # 1. 장애물 없는 경사로
    def test_clear_ramp(self):
        result = self.evaluate([self.ramp], 3.0)

        self.assertEqual(
            result["status"], "경사 기준 이내"
        )

        self.assertEqual(result["verdict"], "V")

    # 2. 경사로 위 장애물
    def test_obstacle_on_ramp(self):
        obstacle = make_detection(
            "obstacle", [150, 150, 250, 250]
        )

        result = self.evaluate(
            [self.ramp, obstacle], 3.0
        )

        self.assertEqual(
            result["status"], "장애물 확인 필요"
        )

        self.assertEqual(
            result["conflicts"][0]["overlap"], 1.0
        )

        self.assertEqual(result["verdict"], "X")

    # 3. 경사로 옆 장애물
    def test_obstacle_beside_ramp(self):
        obstacle = make_detection(
            "obstacle", [450, 100, 550, 200]
        )

        result = self.evaluate(
            [self.ramp, obstacle], 3.0
        )

        self.assertEqual(
            result["status"], "경사 기준 이내"
        )

        self.assertEqual(
            result["conflicts"][0]["overlap"], 0.0
        )

        self.assertEqual(result["verdict"], "V")

    # 4. 장애물이 일부 겹치는 경우
    def test_partial_overlap(self):
        obstacle = make_detection(
            "obstacle", [390, 150, 490, 250]
        )

        result = self.evaluate(
            [self.ramp, obstacle], 3.0
        )

        self.assertEqual(
            result["status"], "장애물 확인 필요"
        )

        self.assertEqual(
            result["conflicts"][0]["overlap"], 0.1
        )

        self.assertEqual(result["verdict"], "?")

    # 5. 경사각 기준 초과
    def test_steep_ramp(self):
        result = self.evaluate([self.ramp], 12.0)

        self.assertEqual(
            result["status"], "경사 기준 초과"
        )

        self.assertEqual(result["verdict"], "X")

    # 6. 경사각 추정 실패
    def test_angle_unavailable(self):
        result = self.evaluate([self.ramp], None)

        self.assertEqual(
            result["status"], "확인 필요"
        )

        self.assertEqual(result["verdict"], "?")

    # 7. 계단만 탐지
    def test_stairs_only(self):
        stairs = make_detection(
            "stairs", [100, 100, 400, 400]
    )

        result = evaluate_accessibility(
            detections=[stairs],
            measurements=[],
            slope_limit=self.slope_limit,
            model_classes=self.model_classes,
        )

        self.assertEqual(
            result["status"], "통행 기준 미충족"
        )

        self.assertEqual(result["verdict"], "X")

    # 8. 객체 미탐지
    def test_no_detection(self):
        result = evaluate_accessibility(
            detections=[],
            measurements=[],
            slope_limit=self.slope_limit,
            model_classes=self.model_classes,
        )

        self.assertEqual(
            result["status"], "확인 필요"
        )

        self.assertEqual(result["verdict"], "?")

    # 9. 경사각이 기준과 동일한 경우
    def test_angle_at_limit(self):
        result = self.evaluate([self.ramp], 5.0)

        self.assertEqual(
            result["status"], "경사 기준 이내"
        )

        self.assertEqual(result["verdict"], "V")

    # 10. 장애물 탐지를 지원하지 않는 모델
    def test_obstacle_unsupported(self):
        model_classes = {
            "ramp_supported": True,
            "stairs_supported": True,
            "obstacle_supported": False,
            "classes": ["ramp", "stairs"]
        }

        result = evaluate_accessibility(
            detections=[self.ramp],
            measurements=[{"angle": 3.0}],
            slope_limit=self.slope_limit,
            model_classes=model_classes,
        )

        self.assertEqual(
            result["status"], "확인 필요"
        )

        self.assertEqual(result["verdict"], "?")
        self.assertTrue(result["warnings"])

    # 11. 모델 클래스 정보가 없는 경우
    def test_model_classes_unknown(self):
        result = evaluate_accessibility(
            detections=[self.ramp],
            measurements=[{"angle": 3.0}],
            slope_limit=self.slope_limit,
            model_classes=None,
        )

        self.assertEqual(
            result["status"], "확인 필요"
        )

        self.assertEqual(result["verdict"], "?")
        self.assertTrue(result["warnings"])

    # 12. 장애물 겹침 비율이 정확히 30%인 경우
    def test_overlap_at_threshold(self):
        obstacle = make_detection(
            "obstacle", [370, 150, 470, 250]
        )

        result = self.evaluate(
            [self.ramp, obstacle], 3.0
        )

        self.assertEqual(
            result["conflicts"][0]["overlap"], 0.3
        )

        self.assertEqual(result["verdict"], "X")

    # 13. 경사각 판단 기준이 없는 경우
    def test_slope_limit_unavailable(self):
        result = evaluate_accessibility(
            detections=[self.ramp],
            measurements=[{"angle": 3.0}],
            slope_limit=None,
            model_classes=self.model_classes,
        )

        self.assertEqual(
            result["status"], "확인 필요"
        )

        self.assertEqual(result["verdict"], "?")


if __name__ == "__main__":
    unittest.main()
