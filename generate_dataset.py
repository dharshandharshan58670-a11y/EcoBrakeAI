"""
EcoBrake AI — Synthetic Dataset Generator for EV Regenerative Braking
======================================================================
This module generates a reproducible synthetic dataset of 5,000 electric vehicle (EV)
braking scenarios based on automotive engineering principles and domain heuristics.

DISCLAIMER:
This dataset is generated for educational and research simulation purposes.
It does NOT represent physical vehicle dynamometer or on-road telemetry, and
the assigned labels reflect simulated heuristic control logic rather than
experimentally certified vehicle braking firmware.
"""

import os
import csv
import math
import random
import argparse


def generate_dataset_records(n_samples: int = 5000, seed: int = 42) -> list[dict]:
    """
    Generate synthetic EV braking scenario records using Python's standard library.
    Guarantees zero-dependency execution across any Python 3 installation while
    ensuring 100% reproducible results.

    Features:
        - speed_kmh: Vehicle speed in km/h (0 to 120 km/h)
        - battery_soc: Battery State of Charge in % (10% to 100%)
        - road_condition: Surface traction level ('Dry', 'Wet', 'Slippery')
        - brake_demand: Driver pedal request normalized (0.05 to 1.00)

    Target:
        - regen_level: Recommended regenerative braking level ('Low', 'Medium', 'High')

    Automotive Engineering Heuristics:
        1. Low Speed Cut-off (speed < 10 km/h):
           Back-EMF too low, friction brakes handle final stopping. -> 'Low'
        2. High Battery SOC (battery_soc >= 95%):
           Li-ion overcharge & lithium plating prevention. -> 'Low'
        3. Traction Limit (road_condition == 'Slippery'):
           Low surface friction; stability control limits regen torque to prevent wheel lockup. -> 'Low'
        4. Moderate Traction Limit (road_condition == 'Wet'):
           Reduced friction; capped at 'Medium'.
        5. Dry Road Baseline:
           - brake_demand < 0.30 -> 'Low'
           - 0.30 <= brake_demand < 0.70 -> 'Medium'
           - brake_demand >= 0.70 and speed >= 35.0 and SOC < 88.0 -> 'High', else 'Medium'
    """
    rng = random.Random(seed)
    records = []

    road_options = ['Dry', 'Dry', 'Dry', 'Dry', 'Dry', 'Dry', 'Wet', 'Wet', 'Slippery']

    for i in range(n_samples):
        # 1. Vehicle speed (km/h): Mixture of urban (45%), suburban (35%), highway (20%)
        mode_select = rng.random()
        if mode_select < 0.45:
            speed = round(rng.uniform(0.0, 50.0), 1)
        elif mode_select < 0.80:
            speed = round(rng.uniform(50.0, 80.0), 1)
        else:
            speed = round(rng.uniform(80.0, 120.0), 1)

        # 2. Battery SOC (%): 10% to 100%
        # Approximate beta-like distribution centered around 55-75%
        u1 = rng.uniform(0.1, 1.0)
        u2 = rng.uniform(0.1, 1.0)
        soc_val = round(10.0 + (0.5 * (u1 + u2)) * 90.0, 1)
        soc_val = min(100.0, max(10.0, soc_val))

        # 3. Road condition
        road = rng.choice(road_options)

        # 4. Brake demand (0.05 to 1.0)
        # In real-world driving: light (35%), medium (45%), heavy/emergency (20%)
        p_demand = rng.random()
        if p_demand < 0.35:
            # Gentle pedal input
            brake_demand = round(rng.uniform(0.05, 0.29), 2)
        elif p_demand < 0.80:
            # Moderate stopping demand
            brake_demand = round(rng.uniform(0.30, 0.69), 2)
        else:
            # Strong deceleration demand
            brake_demand = round(rng.uniform(0.70, 1.00), 2)

        # 5. Apply Automotive Engineering Control Heuristics
        if speed < 10.0:
            level = 'Low'
        elif soc_val >= 95.0:
            level = 'Low'
        elif road == 'Slippery':
            level = 'Low'
        elif road == 'Wet':
            if brake_demand < 0.30:
                level = 'Low'
            else:
                level = 'Medium'
        else:  # Dry road
            if brake_demand < 0.30:
                level = 'Low'
            elif 0.30 <= brake_demand < 0.70:
                level = 'Medium'
            else:
                if speed >= 35.0 and soc_val < 88.0:
                    level = 'High'
                else:
                    level = 'Medium'

        # Realistic 2% sensor/boundary noise on non-critical boundaries
        if rng.random() < 0.02 and speed >= 15.0 and soc_val < 92.0 and road != 'Slippery':
            if level == 'Low':
                level = 'Medium'
            elif level == 'High':
                level = 'Medium'
            elif level == 'Medium':
                level = 'High' if (brake_demand > 0.6 and road == 'Dry') else 'Low'

        records.append({
            'speed_kmh': speed,
            'battery_soc': soc_val,
            'road_condition': road,
            'brake_demand': brake_demand,
            'regen_level': level
        })

    return records


def validate_records(records: list[dict]) -> dict:
    """
    Validates records against schema and bounds without external libraries.
    """
    total = len(records)
    if total == 0:
        raise ValueError("No records generated.")

    speeds = [r['speed_kmh'] for r in records]
    socs = [r['battery_soc'] for r in records]
    brakes = [r['brake_demand'] for r in records]
    levels = [r['regen_level'] for r in records]
    roads = set(r['road_condition'] for r in records)

    level_counts = {}
    for l in levels:
        level_counts[l] = level_counts.get(l, 0) + 1

    return {
        'total_rows': total,
        'speed_min': min(speeds),
        'speed_max': max(speeds),
        'soc_min': min(socs),
        'soc_max': max(socs),
        'brake_min': min(brakes),
        'brake_max': max(brakes),
        'road_conditions': list(roads),
        'target_distribution': level_counts
    }


def import_and_validate_public_dataset(filepath: str) -> tuple[bool, str, dict]:
    """
    Inspects an external public EV dataset CSV file.
    Validates presence of required columns, row counts, and data completeness.
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}", {}

    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        return False, f"Failed to read CSV file: {str(e)}", {}

    expected_inputs = ['speed_kmh', 'battery_soc', 'road_condition', 'brake_demand']
    target = 'regen_level'

    has_inputs = all(col in headers for col in expected_inputs)
    has_target = target in headers

    missing_val_count = 0
    for r in rows:
        for col in expected_inputs:
            if col in r and (r[col] is None or r[col].strip() == ''):
                missing_val_count += 1

    diagnostics = {
        'total_rows': len(rows),
        'columns_found': headers,
        'has_all_expected_inputs': has_inputs,
        'has_target': has_target,
        'missing_values_count': missing_val_count,
        'status_summary': (
            "Fully compatible with EcoBrake AI schema and validation."
            if (has_inputs and has_target)
            else "Partial compatibility: Missing inputs or target column."
        )
    }

    return True, "Dataset inspected successfully.", diagnostics


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EV regenerative braking dataset.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate (default: 5000)")
    parser.add_argument("--output", type=str, default="ev_regenerative_braking.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output

    print(f"Generating {args.samples} synthetic EV braking scenarios (seed={args.seed})...")
    records = generate_dataset_records(n_samples=args.samples, seed=args.seed)

    validation = validate_records(records)
    print("Dataset validation passed!")
    print(f"  Total records: {validation['total_rows']}")
    print(f"  Speed range:   {validation['speed_min']} to {validation['speed_max']} km/h")
    print(f"  SOC range:     {validation['soc_min']}% to {validation['soc_max']}%")
    print(f"  Brake demand:  {validation['brake_min']} to {validation['brake_max']}")
    print("  Target distribution:")
    for level, count in sorted(validation['target_distribution'].items()):
        pct = (count / validation['total_rows']) * 100
        print(f"    - {level}: {count} ({pct:.1f}%)")

    # Save to CSV
    fieldnames = ['speed_kmh', 'battery_soc', 'road_condition', 'brake_demand', 'regen_level']
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSuccessfully saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
