"""
EcoBrake AI — Automated Unit & Integration Test Suite
=====================================================
Validates all dataset generator outputs, physical energy calculations,
automotive rule heuristics, edge cases, and graceful fallbacks.
"""

import os
import csv
import math
import unittest
from generate_dataset import generate_dataset_records, validate_records, import_and_validate_public_dataset
from physics_engine import (
    calculate_kinetic_energy_joules,
    calculate_energy_recovery_wh,
    fallback_rule_predict,
    predict_regen_level,
    get_engineering_diagnosis
)


class TestEcoBrakeAI(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.base_dir, "ev_regenerative_braking.csv")

    def test_01_dataset_generator_5000_rows(self):
        """Verify generator creates exactly 5,000 rows with complete schema."""
        records = generate_dataset_records(n_samples=5000, seed=42)
        self.assertEqual(len(records), 5000, "Dataset must contain exactly 5,000 records")

        # Validate schema
        expected_keys = {'speed_kmh', 'battery_soc', 'road_condition', 'brake_demand', 'regen_level'}
        for r in records[:50]:
            self.assertEqual(set(r.keys()), expected_keys)

        validation = validate_records(records)
        self.assertEqual(validation['total_rows'], 5000)
        self.assertIn('Dry', validation['road_conditions'])
        self.assertIn('Wet', validation['road_conditions'])
        self.assertIn('Slippery', validation['road_conditions'])

        # Verify all 3 target classes are generated
        dist = validation['target_distribution']
        self.assertIn('Low', dist)
        self.assertIn('Medium', dist)
        self.assertIn('High', dist)
        self.assertGreater(dist['High'], 0, "High regen must be represented")
        self.assertGreater(dist['Medium'], 0, "Medium regen must be represented")
        self.assertGreater(dist['Low'], 0, "Low regen must be represented")

    def test_02_dataset_csv_file_on_disk(self):
        """Verify existing ev_regenerative_braking.csv file integrity."""
        self.assertTrue(os.path.exists(self.csv_path), "ev_regenerative_braking.csv must exist")
        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 5000, "CSV on disk must contain 5,000 records")

    def test_03_zero_speed_and_negative_energy_prevention(self):
        """Verify that at 0 speed, energy is 0 and energy is never negative."""
        # Zero speed
        e_k = calculate_kinetic_energy_joules(mass_kg=1800, speed_kmh=0.0)
        self.assertEqual(e_k, 0.0)

        rec = calculate_energy_recovery_wh(mass_kg=1800, speed_kmh=0.0, regen_level="High")
        self.assertEqual(rec['recovered_energy_wh'], 0.0)
        self.assertEqual(rec['total_kinetic_energy_wh'], 0.0)

        # Negative inputs must be clamped to non-negative
        e_k_neg = calculate_kinetic_energy_joules(mass_kg=-1000, speed_kmh=-50)
        self.assertGreaterEqual(e_k_neg, 0.0)
        rec_neg = calculate_energy_recovery_wh(mass_kg=-1000, speed_kmh=-50, regen_level="Low")
        self.assertGreaterEqual(rec_neg['recovered_energy_wh'], 0.0)

    def test_04_high_battery_soc_overcharge_protection(self):
        """Verify that high battery SOC (>= 95%) forces Low regen."""
        pred = fallback_rule_predict(speed=80.0, soc=98.0, road="Dry", brake_demand=0.8)
        self.assertEqual(pred, "Low", "At SOC >= 95%, regen must be throttled to Low to prevent overcharge")

        pred_boundary = fallback_rule_predict(speed=60.0, soc=95.0, road="Dry", brake_demand=0.7)
        self.assertEqual(pred_boundary, "Low", "Boundary SOC = 95% must throttle to Low")

    def test_05_road_condition_traction_limits(self):
        """Verify traction limits across all three road conditions."""
        # Slippery surface -> Low regen regardless of brake demand
        pred_ice = fallback_rule_predict(speed=70.0, soc=50.0, road="Slippery", brake_demand=0.9)
        self.assertEqual(pred_ice, "Low", "Slippery roads must restrict regen to Low for stability control")

        # Wet surface -> capped at Medium even with high demand
        pred_wet_high_demand = fallback_rule_predict(speed=70.0, soc=50.0, road="Wet", brake_demand=0.85)
        self.assertEqual(pred_wet_high_demand, "Medium", "Wet roads must cap regen at Medium")

        # Wet surface with low demand -> Low
        pred_wet_low_demand = fallback_rule_predict(speed=70.0, soc=50.0, road="Wet", brake_demand=0.20)
        self.assertEqual(pred_wet_low_demand, "Low")

        # Dry surface with high demand and adequate speed -> High
        pred_dry_high = fallback_rule_predict(speed=80.0, soc=50.0, road="Dry", brake_demand=0.85)
        self.assertEqual(pred_dry_high, "High")

    def test_06_low_speed_cutoff(self):
        """Verify that speeds < 10 km/h result in Low regen."""
        pred_crawl = fallback_rule_predict(speed=5.0, soc=50.0, road="Dry", brake_demand=0.8)
        self.assertEqual(pred_crawl, "Low", "Speeds < 10 km/h must command Low regen due to low back-EMF")

    def test_07_energy_recovery_monotonicity(self):
        """Verify that energy recovery increases with higher speed and higher regen level."""
        mass = 1800
        speed = 80.0
        rec_high = calculate_energy_recovery_wh(mass, speed, "High")["recovered_energy_wh"]
        rec_med = calculate_energy_recovery_wh(mass, speed, "Medium")["recovered_energy_wh"]
        rec_low = calculate_energy_recovery_wh(mass, speed, "Low")["recovered_energy_wh"]

        self.assertGreater(rec_high, rec_med, "High regen must recover more energy than Medium")
        self.assertGreater(rec_med, rec_low, "Medium regen must recover more energy than Low")

    def test_08_missing_model_graceful_handling(self):
        """Verify that predict_regen_level handles None pipeline gracefully without throwing."""
        pred, probs, src = predict_regen_level(
            pipeline=None,
            speed=70.0,
            soc=50.0,
            road="Dry",
            brake_demand=0.85
        )
        self.assertEqual(pred, "High")
        self.assertIsNone(probs)
        self.assertIn("Simulated Engineering Heuristic", src)

    def test_09_public_dataset_validator(self):
        """Verify the public dataset validator handles both valid files and missing files."""
        success, msg, diag = import_and_validate_public_dataset("non_existent_file.csv")
        self.assertFalse(success)

        # Test against our generated dataset
        success_valid, msg_valid, diag_valid = import_and_validate_public_dataset(self.csv_path)
        self.assertTrue(success_valid)
        self.assertTrue(diag_valid['has_all_expected_inputs'])
        self.assertTrue(diag_valid['has_target'])
        self.assertEqual(diag_valid['total_rows'], 5000)


if __name__ == "__main__":
    unittest.main()
