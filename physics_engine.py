"""
EcoBrake AI — Core Automotive Physics & Heuristic Engine
========================================================
Pure Python implementation of EV kinetic energy physics, energy conversion,
powertrain efficiency chain modeling, and automotive rule heuristics.
Zero third-party dependencies required.
"""


def calculate_kinetic_energy_joules(mass_kg: float, speed_kmh: float) -> float:
    """
    Calculates vehicle kinetic energy:
        E_k = 0.5 * m * v^2 [Joules]
    where v is velocity in m/s (speed_kmh / 3.6).
    Guaranteed non-negative.
    """
    m = max(0.0, float(mass_kg))
    v = max(0.0, float(speed_kmh)) / 3.6
    e_k = 0.5 * m * (v ** 2)
    return round(max(0.0, e_k), 2)


def calculate_motor_rpm(
    speed_kmh: float,
    gear_ratio: float = 9.0,
    wheel_radius_m: float = 0.32
) -> float:
    """
    Computes electric motor RPM from vehicle linear speed, reduction gear ratio, and wheel radius:
        v_ms = speed_kmh / 3.6
        wheel_circumference = 2 * pi * wheel_radius_m
        motor_rpm = (v_ms / wheel_circumference) * gear_ratio * 60
    Guaranteed non-negative.
    """
    import math
    s = max(0.0, float(speed_kmh))
    if s <= 0.0:
        return 0.0
    v_ms = s / 3.6
    r = max(0.01, float(wheel_radius_m))
    g = max(0.1, float(gear_ratio))
    wheel_circ = 2.0 * math.pi * r
    rpm = (v_ms / wheel_circ) * g * 60.0
    return round(max(0.0, rpm), 1)


def calculate_energy_recovery_wh(
    mass_kg: float,
    speed_kmh: float,
    regen_level: str,
    eta_powertrain: float = 0.77,
    gear_ratio: float = 9.0,
    wheel_radius_m: float = 0.32,
    brake_demand: float = 0.55
) -> dict:
    """
    Computes illustrative kinetic energy recovery in Watt-hours (Wh) and Joules,
    as well as motor rotational kinematics: motor RPM, total motor revolutions during stop,
    and electrical energy recovered per motor revolution in Joules per revolution (J/rev).

    Conversion: 1 Wh = 3600 Joules -> E_Wh = E_Joules / 3600

    Assumptions & Fractions:
        - High Regen: ~65% of deceleration braking force allocated to electric motor
        - Medium Regen: ~45% allocated to motor
        - Low Regen: ~20% allocated to motor
        - Powertrain efficiency chain:
            * Motor Generator Efficiency (eta_motor) ~ 90%
            * Inverter AC/DC Conversion (eta_inverter) ~ 95%
            * Battery Electrochemical Charge Acceptance (eta_battery) ~ 90%
            * Combined powertrain efficiency: 0.90 * 0.95 * 0.90 = ~77% (0.77)
        - Powertrain rotational parameters:
            * Single-speed reduction gear ratio (gear_ratio): default 9.0:1
            * Wheel rolling radius (wheel_radius_m): default 0.32 m (~2.01 m circumference)

    Returns dictionary with full physical breakdown. Guaranteed non-negative.
    """
    import math

    e_joules = calculate_kinetic_energy_joules(mass_kg, speed_kmh)
    e_wh_total = e_joules / 3600.0

    fraction_map = {
        "High": 0.65,
        "Medium": 0.45,
        "Low": 0.20
    }
    regen_fraction = fraction_map.get(regen_level, 0.20)
    friction_fraction = 1.0 - regen_fraction

    # Recovered energy stored in battery
    recovered_wh = e_wh_total * regen_fraction * max(0.0, min(1.0, eta_powertrain))
    recovered_joules = recovered_wh * 3600.0

    # Thermal dissipation in brake pads & discs
    friction_dissipated_wh = e_wh_total * friction_fraction

    # Electrical and inverter heat loss
    electrical_loss_wh = (e_wh_total * regen_fraction) * (1.0 - eta_powertrain)

    # Motor rotational speed (RPM)
    s_kmh = max(0.0, float(speed_kmh))
    v_ms = s_kmh / 3.6
    r_w = max(0.01, float(wheel_radius_m))
    g_ratio = max(0.1, float(gear_ratio))
    wheel_circ = 2.0 * math.pi * r_w
    motor_rpm = (v_ms / wheel_circ) * g_ratio * 60.0 if v_ms > 0 else 0.0

    # Braking stop dynamics & motor revolutions
    bd = max(0.05, min(1.0, float(brake_demand)))
    a_decel = max(1.0, bd * 7.5)  # m/s^2 deceleration
    stopping_dist_m = (v_ms ** 2) / (2.0 * a_decel) if v_ms > 0 else 0.0
    stopping_time_s = v_ms / a_decel if v_ms > 0 else 0.0
    motor_revs = (stopping_dist_m / wheel_circ) * g_ratio if v_ms > 0 else 0.0

    # Electrical energy recovered per motor revolution (J/rev)
    joules_per_rev = (recovered_joules / motor_revs) if motor_revs > 0 else 0.0

    return {
        "total_kinetic_energy_wh": round(max(0.0, e_wh_total), 2),
        "total_kinetic_energy_j": round(max(0.0, e_joules), 1),
        "recovered_energy_wh": round(max(0.0, recovered_wh), 2),
        "recovered_energy_j": round(max(0.0, recovered_joules), 1),
        "friction_dissipated_wh": round(max(0.0, friction_dissipated_wh), 2),
        "electrical_loss_wh": round(max(0.0, electrical_loss_wh), 2),
        "regen_split_percent": int(round(regen_fraction * 100)),
        "friction_split_percent": int(round(friction_fraction * 100)),
        "powertrain_efficiency_percent": int(round(eta_powertrain * 100)),
        "motor_rpm": round(max(0.0, motor_rpm), 1),
        "stopping_distance_m": round(max(0.0, stopping_dist_m), 2),
        "stopping_time_s": round(max(0.0, stopping_time_s), 2),
        "motor_revolutions": round(max(0.0, motor_revs), 1),
        "energy_recovered_per_rev_j": round(max(0.0, joules_per_rev), 2),
        "gear_ratio": round(g_ratio, 2),
        "wheel_radius_m": round(r_w, 3)
    }



def fallback_rule_predict(speed: float, soc: float, road: str, brake_demand: float) -> str:
    """
    Baseline automotive domain logic fallback used when the trained joblib model
    is not yet generated, guaranteeing zero app crashes and continuous usability.
    """
    s = max(0.0, float(speed))
    b_soc = max(0.0, min(100.0, float(soc)))
    bd = max(0.0, min(1.0, float(brake_demand)))

    # Low speed cutoff: back-EMF too low, mechanical brakes finish stop
    if s < 10.0:
        return "Low"

    # Battery overcharge protection: BMS prevents high charging currents
    if b_soc >= 95.0:
        return "Low"

    # Low friction surface: ESC/ABS priority to prevent wheel lockup
    if road == "Slippery":
        return "Low"

    # Moderate friction wet surface: capped at Medium
    if road == "Wet":
        return "Low" if bd < 0.30 else "Medium"

    # Normal dry road conditions
    if bd < 0.30:
        return "Low"
    elif bd < 0.70:
        return "Medium"
    else:
        return "High" if (s >= 35.0 and b_soc < 88.0) else "Medium"


def predict_regen_level(pipeline, speed: float, soc: float, road: str, brake_demand: float):
    """
    Executes prediction using the scikit-learn Pipeline if available,
    otherwise gracefully falls back to baseline engineering rules.
    """
    if pipeline is not None:
        try:
            import pandas as pd
            input_df = pd.DataFrame([{
                'speed_kmh': float(speed),
                'battery_soc': float(soc),
                'road_condition': str(road),
                'brake_demand': float(brake_demand)
            }])
            pred = pipeline.predict(input_df)[0]
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(input_df)[0]
                classes = pipeline.classes_
                prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
            else:
                prob_dict = None
            return pred, prob_dict, "ML Pipeline (Random Forest)"
        except Exception as e:
            fallback = fallback_rule_predict(speed, soc, road, brake_demand)
            return fallback, None, f"Fallback Heuristic (Model Error: {str(e)})"
    else:
        fallback = fallback_rule_predict(speed, soc, road, brake_demand)
        return fallback, None, "Simulated Engineering Heuristic (Model not trained yet)"


def get_engineering_diagnosis(speed: float, soc: float, road: str, brake_demand: float, regen_level: str) -> str:
    """
    Generates an automotive engineering explanation of why the predicted level was selected.
    """
    notes = []
    if speed < 10.0:
        notes.append("• **Low Speed Threshold (< 10 km/h):** Motor back-EMF is too low to produce sufficient regenerative counter-torque. Mechanical friction brakes handle the final stop without stall.")
    if soc >= 95.0:
        notes.append("• **Battery Overcharge Protection (SOC ≥ 95%):** Battery cells are near maximum charge cut-off. BMS curtails high charging currents to prevent lithium plating and thermal runaway.")
    if road == "Slippery":
        notes.append("• **Traction Control Limit (Slippery):** On ice/snow, high motor regenerative torque causes sudden wheel slip and yaw instability. ABS/ESC commands Low regen to preserve lateral tire grip.")
    elif road == "Wet":
        notes.append("• **Traction Moderation (Wet):** Reduced friction coefficient (μ ≈ 0.5) caps regenerative torque to Medium to prevent front/rear wheel lockup.")

    if not notes:
        if regen_level == "High":
            notes.append("• **Optimal Deceleration Recuperation:** Dry road with adequate traction, vehicle speed ≥ 35 km/h, and high driver brake demand allows maximum safe electric motor energy recovery.")
        elif regen_level == "Medium":
            notes.append("• **Balanced Deceleration:** Moderate driver braking request captured predominantly by the electric drive unit with minimal friction brake engagement.")
        else:
            notes.append("• **Gentle Deceleration:** Light brake demand matches one-pedal coasting or gentle deceleration profile.")

    return "\n".join(notes)
