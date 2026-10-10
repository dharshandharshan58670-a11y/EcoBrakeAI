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


def calculate_energy_recovery_wh(
    mass_kg: float,
    speed_kmh: float,
    regen_level: str,
    eta_powertrain: float = 0.77
) -> dict:
    """
    Computes illustrative kinetic energy recovery in Watt-hours (Wh).

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

    Returns dictionary with full physical breakdown. Guaranteed non-negative.
    """
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

    # Thermal dissipation in brake pads & discs
    friction_dissipated_wh = e_wh_total * friction_fraction

    # Electrical and inverter heat loss
    electrical_loss_wh = (e_wh_total * regen_fraction) * (1.0 - eta_powertrain)

    return {
        "total_kinetic_energy_wh": round(max(0.0, e_wh_total), 2),
        "total_kinetic_energy_j": round(max(0.0, e_joules), 1),
        "recovered_energy_wh": round(max(0.0, recovered_wh), 2),
        "friction_dissipated_wh": round(max(0.0, friction_dissipated_wh), 2),
        "electrical_loss_wh": round(max(0.0, electrical_loss_wh), 2),
        "regen_split_percent": int(round(regen_fraction * 100)),
        "friction_split_percent": int(round(friction_fraction * 100)),
        "powertrain_efficiency_percent": int(round(eta_powertrain * 100))
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
