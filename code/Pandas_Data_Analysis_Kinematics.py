import re
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = "Pasted text(2).txt"
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

MEASUREMENT_UNCERTAINTY = {
    "displacement": 0.02,
    "velocity": 0.07,
    "acceleration": 0.07,
}

ROUGHNESS_LENGTH = {
    0.10: 2.91,
    0.12: 3.08,
    0.15: 3.75,
    0.18: 3.42,
    0.22: 3.18,
    0.31: 2.67,
}

def clean_number(x):
    x = str(x).replace("−", "-").strip()
    if x == "" or x.lower() == "nan":
        return np.nan
    return float(x)

def parse_markdown_table(lines):
    rows = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if all(set(p) <= {"-", " "} for p in parts):
            continue
        rows.append(parts)

    if len(rows) < 2:
        return None

    header = rows[0]
    data = rows[1:]

    cleaned_rows = []
    for row in data:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        cleaned_rows.append(row[:len(header)])

    return header, cleaned_rows

def table_to_long_df(porosity, table_type, header, rows):
    output = []

    for row in rows:
        try:
            time = clean_number(row[0])
        except ValueError:
            continue

        values = [v for v in row[1:] if str(v).strip() != ""]

        if len(values) < 10:
            continue

        first_five = values[:5]
        second_five = values[5:10]

        if table_type == "displacement":
            components = ("Sx", "Sy")
        elif table_type == "velocity":
            components = ("vx", "vy")
        elif table_type == "acceleration":
            components = ("ax", "ay")
        else:
            continue

        for i, val in enumerate(first_five, start=1):
            output.append({
                "Porosity": porosity,
                "Data_Type": table_type,
                "Time_s": time,
                "Trial": i,
                "Component": components[0],
                "Value": clean_number(val),
            })

        for i, val in enumerate(second_five, start=1):
            output.append({
                "Porosity": porosity,
                "Data_Type": table_type,
                "Time_s": time,
                "Trial": i,
                "Component": components[1],
                "Value": clean_number(val),
            })

    return pd.DataFrame(output)

def detect_table_type(context):
    context = context.lower()

    if "displacement" in context or "maximum height" in context or "range" in context:
        return "displacement"
    if "velocity" in context or "velocities" in context:
        return "velocity"
    if "acceleration" in context or "accelerations" in context:
        return "acceleration"

    return None

text = Path(INPUT_FILE).read_text(encoding="utf-8")

sections = re.split(r"## POROSITY\s+([0-9.]+)", text)

all_data = []

for i in range(1, len(sections), 2):
    porosity = float(sections[i])
    body = sections[i + 1]

    lines = body.splitlines()

    current_context = ""
    table_lines = []

    for line in lines + ["END"]:
        if line.startswith("###") or line.startswith("##"):
            current_context += " " + line

        if line.strip().startswith("|"):
            table_lines.append(line)
        else:
            if table_lines:
                table_type = detect_table_type(current_context)
                parsed = parse_markdown_table(table_lines)

                if parsed and table_type:
                    header, rows = parsed
                    df_long = table_to_long_df(porosity, table_type, header, rows)
                    if not df_long.empty:
                        all_data.append(df_long)

                table_lines = []

raw_df = pd.concat(all_data, ignore_index=True)

raw_df.to_csv(OUTPUT_DIR / "long_format_trial_data.csv", index=False)

summary = (
    raw_df
    .groupby(["Porosity", "Data_Type", "Component", "Time_s"])
    .agg(
        Mean=("Value", "mean"),
        SD=("Value", "std"),
        SEM=("Value", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        N=("Value", "count"),
    )
    .reset_index()
)

summary["Instrument_Uncertainty"] = summary["Data_Type"].map(MEASUREMENT_UNCERTAINTY)
summary["Combined_Uncertainty"] = np.sqrt(
    summary["SEM"].fillna(0) ** 2 + summary["Instrument_Uncertainty"] ** 2
)

summary.to_csv(OUTPUT_DIR / "time_resolved_summary.csv", index=False)

peak_rows = []

for porosity in sorted(raw_df["Porosity"].unique()):
    p_df = raw_df[raw_df["Porosity"] == porosity]

    for trial in sorted(p_df["Trial"].unique()):
        t_df = p_df[p_df["Trial"] == trial]

        def get_component(dtype, component):
            return t_df[
                (t_df["Data_Type"] == dtype) &
                (t_df["Component"] == component)
            ]["Value"]

        vx = get_component("velocity", "vx")
        vy = get_component("velocity", "vy")
        ax = get_component("acceleration", "ax")
        ay = get_component("acceleration", "ay")

        peak_rows.append({
            "Porosity": porosity,
            "Trial": trial,
            "Peak_Horizontal_Velocity_ms": vx.max() if not vx.empty else np.nan,
            "Peak_Vertical_Velocity_ms": vy.max() if not vy.empty else np.nan,
            "Peak_Horizontal_Acceleration_ms2": ax.min() if not ax.empty else np.nan,
            "Peak_Vertical_Acceleration_ms2": ay.min() if not ay.empty else np.nan,
        })

peak_df = pd.DataFrame(peak_rows)
peak_df.to_csv(OUTPUT_DIR / "trial_peak_values.csv", index=False)

peak_summary = (
    peak_df
    .groupby("Porosity")
    .agg(
        Peak_Horizontal_Velocity_Mean=("Peak_Horizontal_Velocity_ms", "mean"),
        Peak_Horizontal_Velocity_SD=("Peak_Horizontal_Velocity_ms", "std"),
        Peak_Vertical_Velocity_Mean=("Peak_Vertical_Velocity_ms", "mean"),
        Peak_Vertical_Velocity_SD=("Peak_Vertical_Velocity_ms", "std"),
        Peak_Horizontal_Acceleration_Mean=("Peak_Horizontal_Acceleration_ms2", "mean"),
        Peak_Horizontal_Acceleration_SD=("Peak_Horizontal_Acceleration_ms2", "std"),
        Peak_Vertical_Acceleration_Mean=("Peak_Vertical_Acceleration_ms2", "mean"),
        Peak_Vertical_Acceleration_SD=("Peak_Vertical_Acceleration_ms2", "std"),
        N=("Trial", "count"),
    )
    .reset_index()
)

peak_summary["Aerodynamic_Roughness_Length_m"] = peak_summary["Porosity"].map(ROUGHNESS_LENGTH)

peak_summary.to_csv(OUTPUT_DIR / "processed_summary_with_uncertainty.csv", index=False)

print("Files created in processed_data/")
print("- long_format_trial_data.csv")
print("- time_resolved_summary.csv")
print("- trial_peak_values.csv")
print("- processed_summary_with_uncertainty.csv")