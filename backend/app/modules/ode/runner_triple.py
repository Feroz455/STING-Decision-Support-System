import os
import pandas as pd

from dummy_data_generator import DummyDataGenerator
from equations_daily import EquationSystem
from genetic_algorithms import TripleDoseOptimizer
from plot_functions import TriplePlotter


class TripleRunner:
    def __init__(self, out_dir="outputs_triple", generations=14, pop_size=12):
        self.out_dir = out_dir
        self.generations = generations
        self.pop_size = pop_size
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        patient_df = DummyDataGenerator(number_of_patients=1, seed=42).get_dummy_data()
        print(patient_df.to_string(index=False))
        eq = EquationSystem(patient_df)
        optimizer = TripleDoseOptimizer(eq, n_generations=self.generations, pop_size=self.pop_size, elite_size=3, seed=123)
        best_plan, best_score, best_metrics, best_out, history = optimizer.optimize()

        patient_dir = os.path.join(self.out_dir, "patient_01")
        os.makedirs(patient_dir, exist_ok=True)

        TriplePlotter().plot_wbc(best_out, os.path.join(patient_dir, "wbc_dose_plot.png"))
        TriplePlotter().plot_anc(best_out, os.path.join(patient_dir, "anc_dose_plot.png"))
        TriplePlotter().plot_vipn(best_out, os.path.join(patient_dir, "vipn_dose_plot.png"))

        pd.DataFrame(history).to_excel(os.path.join(patient_dir, "ga_history.xlsx"), index=False)
        pd.DataFrame({
            "t": best_out["t"],
            "WBC": best_out["WBC"],
            "ANC": best_out["ANC"],
            "VIPN": best_out["VIPN"],
        }).to_excel(os.path.join(patient_dir, "time_series.xlsx"), index=False)
        patient_df.to_excel(os.path.join(patient_dir, "patient_info.xlsx"), index=False)

        print("\nFinal en iyi plan:")
        print("6MP =", best_plan["6mp"])
        print("MTX =", best_plan["mtx"])
        print("VCR =", best_plan["vcr"])
        print(f"WBC min={best_metrics['wbc_min']:.3f}, max={best_metrics['wbc_max']:.3f}, hedef={best_metrics['wbc_target_frac']:.2%}")
        print(f"ANC min={best_metrics['anc_min']:.3f}, hedef={best_metrics['anc_target_frac']:.2%}")
        print(f"VIPN min={best_metrics['vipn_min']:.3f}")
        print(f"Skor={best_score:.4f}")
        return best_plan, history, best_out


if __name__ == "__main__":
    TripleRunner().run()
