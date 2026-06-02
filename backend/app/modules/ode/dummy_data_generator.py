import numpy as np
import pandas as pd


class DummyDataGenerator:
    def __init__(self, treatment_days=1, number_of_patients=1, seed=42):
        self.treatment_days = treatment_days
        self.number_of_patients = number_of_patients
        self.rng = np.random.default_rng(seed)

    def calculate_bsa(self, weight, height):
        return round(np.sqrt((weight * height) / 3600.0), 2)

    def get_weight_for_age(self, age):
        if age <= 3:
            return self.rng.uniform(8, 14)
        if age <= 6:
            return self.rng.uniform(14, 22)
        if age <= 10:
            return self.rng.uniform(20, 32)
        if age <= 13:
            return self.rng.uniform(28, 45)
        if age <= 15:
            return self.rng.uniform(40, 70)
        return self.rng.uniform(50, 80)

    def get_height_for_age(self, age):
        if age <= 3:
            return self.rng.uniform(70, 95)
        if age <= 6:
            return self.rng.uniform(95, 115)
        if age <= 10:
            return self.rng.uniform(110, 135)
        if age <= 13:
            return self.rng.uniform(130, 155)
        if age <= 15:
            return self.rng.uniform(150, 175)
        return self.rng.uniform(160, 185)

    def get_dummy_data(self):
        rows = []
        for pid in range(1, self.number_of_patients + 1):
            age = int(self.rng.integers(1, 18))
            weight = round(self.get_weight_for_age(age), 1)
            height = round(self.get_height_for_age(age), 1)
            bsa = self.calculate_bsa(weight, height)
            rows.append({
                "Patient_ID": pid,
                "Day": 0,
                "6MP_Daily_Dose_mg": 50.0,
                "MTX_Weekly_Dose_mg": 20.0,
                "VCR_28day_Dose_mg": 1.2,
                "WBC": round(self.rng.uniform(3.0, 3.4), 2),
                "ANC": round(self.rng.uniform(1.0, 1.4), 2),
                "Weight_kg": weight,
                "Height_cm": height,
                "BSA": bsa,
                "Age": age,
                "TPMT": int(self.rng.choice([0, 1])),
                "Diet": round(self.rng.uniform(0.9, 1.2), 4),
                "Exercise": float(self.rng.choice([0.5, 1.0])),
                "Vitamin_D": round(self.rng.uniform(20, 40), 1),
                "Under3_Sensitivity": 1.05 if age < 3 else 1.0,
            })
        return pd.DataFrame(rows)
