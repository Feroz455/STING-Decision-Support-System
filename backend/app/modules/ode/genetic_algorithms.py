"""
genetic_algorithms.py — STING DSS GA Doz Optimizasyonu (v2)
============================================================
Yeni: ode_simulator.SimulationConfig ile çalışır (5 ilaç, 250 gün, 4 faz).
      TripleDoseOptimizer → FiveDrugOptimizer olarak güncellendi.
      Geri uyumluluk için TripleDoseOptimizer adı korundu.

Optimize edilen ilaçlar:
  6-MP  → haftalık çizelge (idame/konsolidasyon fazlarında)
  MTX   → haftalık çizelge (idame/konsolidasyon fazlarında)
  VCR   → 28 günlük döngü çizelgesi (tüm fazlar)
  DNR   → sabit (G1,8,15,22 + G84,91) — protokol gereği optimize edilmez

PEG-ASP → sabit COG/BFM protokolü — optimize edilmez
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class TripleDoseOptimizer:
    """
    5 ilaçlı ALL tedavisinde 6-MP / MTX / VCR doz optimizasyonu.
    DNR ve PEG-ASP protokol sabit olduğu için optimize edilmez,
    SimulationConfig'e doğrudan geçirilir.
    """

    def __init__(
        self,
        equation_system=None,       # geriye uyumluluk — artık kullanılmıyor
        n_generations: int = 14,
        pop_size:      int = 12,
        elite_size:    int = 3,
        seed:          int = 123,
        # Hasta parametreleri (SimulationConfig için)
        patient_config: Optional[Dict] = None,
    ):
        self.eq            = equation_system   # eski API için saklandı
        self.n_generations = int(n_generations)
        self.pop_size      = int(pop_size)
        self.elite_size    = int(elite_size)
        self.rng           = np.random.default_rng(seed)
        self.patient_config = patient_config or {}

        # 250 gün / 4 faz protokolü
        # İdame fazı: G140–250 → ~16 hafta = 16 haftalık çizelge
        # Konsolidasyon: G29–84 → ~8 hafta
        # Toplam aktif 6-MP/MTX fazı: ~24 hafta
        self.n_weeks   = 24    # 6-MP ve MTX haftalık çizelge uzunluğu
        self.n_vcr     = 9     # VCR 28 günlük döngü sayısı (250/28 ≈ 9)

        # Doz aralıkları
        self.bounds_6mp = (35.0, 75.0)
        self.bounds_mtx = (8.0,  22.0)
        self.bounds_vcr = (0.8,   1.5)
        self.bounds_dnr = (15.0, 30.0)  # sabit protokol, optimize edilmez

        # Aktif ilaçlar
        self.active_drugs = {"6mp", "mtx", "vcr"}

    def _sim(self, dose_6mp: float, dose_mtx: float, dose_vcr: float) -> Dict:
        """Yeni ode_simulator.SimulationConfig ile tek simülasyon çalıştır."""
        from app.modules.ode.ode_simulator import SimulationConfig, PhaseDefinition, run_simulation

        pc = self.patient_config
        active = list(self.active_drugs)

        cfg = SimulationConfig(
            weight_kg      = pc.get("weight_kg", 30.0),
            height_cm      = pc.get("height_cm", 135.0),
            tpmt           = pc.get("tpmt", 1.0),
            vitamin_d      = pc.get("vitamin_d", 30.0),
            diet           = pc.get("diet", 1.0),
            exercise       = pc.get("exercise", 0.4),
            wbc0           = pc.get("wbc0", 5.0),
            anc0           = pc.get("anc0", 1.6),
            active_drugs   = active,
            dose_6mp_mg    = dose_6mp if "6mp" in self.active_drugs else 0.0,
            dose_mtx_mg    = dose_mtx if "mtx" in self.active_drugs else 0.0,
            dose_vcr_mg    = dose_vcr if "vcr" in self.active_drugs else 0.0,
            # Yeni ilaçlar — sabit protokol dozlarıyla (GA bu ilaçları henüz optimize etmiyor)
            dose_ster_mg_m2  = self.patient_config.get("dose_ster_mg_m2", 40.0)
                               if "corticosteroid" in self.active_drugs else 0.0,
            dose_arac_mg_m2  = self.patient_config.get("dose_arac_mg_m2", 75.0)
                               if "cytarabine" in self.active_drugs else 0.0,
            dose_cpm_mg_m2   = self.patient_config.get("dose_cpm_mg_m2", 1000.0)
                               if "cyclophosphamide" in self.active_drugs else 0.0,
            dose_6tg_mg_m2   = self.patient_config.get("dose_6tg_mg_m2", 60.0)
                               if "6tg" in self.active_drugs else 0.0,
            dose_cop_mg      = self.patient_config.get("dose_cop_mg", 60.0)
                               if "copanlisib" in self.active_drugs else 0.0,
            dose_nov_mg_kg   = self.patient_config.get("dose_nov_mg_kg", 10.0)
                               if "novobiocin" in self.active_drugs else 0.0,
            custom_phases    = [
                PhaseDefinition(
                    name=ph.get("name","Phase"),
                    duration_days=ph.get("duration_days",28),
                    drugs=ph.get("drugs",[]),
                    doses=ph.get("doses",{}),
                )
                for ph in self.patient_config.get("custom_phases",[])
            ] if self.patient_config.get("custom_phases") else [],
            dose_dnr_mg_m2 = pc.get("dose_dnr_mg_m2", 25.0),
            peg_dose_per_m2= pc.get("peg_dose_per_m2", 2500.0),
            peg_dose_days  = pc.get("peg_dose_days", [4, 36, 57, 91]),
            t_end          = pc.get("t_end", 250.0),
            dt             = 0.5,  # daha hızlı optimizasyon için
            wbc_target_low = 1.5,
            wbc_target_high= 3.0,
            anc_target_low = 0.5,
            anc_target_high= 2.0,
        )
        return run_simulation(cfg)

    def _random_walk(self, lo, hi, n, start, sigma):
        arr = [float(np.clip(start, lo, hi))]
        for _ in range(1, n):
            arr.append(float(np.clip(
                arr[-1] + self.rng.normal(0, sigma), lo, hi
            )))
        return arr

    def make_individual(self):
        return {
            "6mp": self._random_walk(*self.bounds_6mp, self.n_weeks, start=58.0, sigma=5.0)
                   if "6mp" in self.active_drugs else [0.0] * self.n_weeks,
            "mtx": self._random_walk(*self.bounds_mtx, self.n_weeks, start=18.0, sigma=2.5)
                   if "mtx" in self.active_drugs else [0.0] * self.n_weeks,
            "vcr": self._random_walk(*self.bounds_vcr, self.n_vcr,   start=1.2,  sigma=0.10)
                   if "vcr" in self.active_drugs else [0.0] * self.n_vcr,
        }

    @staticmethod
    def target_penalty(series, lo, hi):
        below = np.clip(lo - series, 0, None)
        above = np.clip(series - hi, 0, None)
        return float(np.sum(below**2) + np.sum(above**2))

    @staticmethod
    def target_fraction(series, lo, hi):
        return float(np.mean((series >= lo) & (series <= hi)))

    @staticmethod
    def smoothness_penalty(arr):
        arr = np.asarray(arr, dtype=float)
        return float(np.sum(np.diff(arr)**2))

    def fitness(self, ind) -> Tuple[float, Dict, Dict]:
        """Bireyi değerlendir — yeni ode_simulator kullanır."""
        # Haftalık → günlük ortalama doz (simülasyon için yaklaşık)
        dose_6mp = float(np.mean(ind["6mp"])) if "6mp" in self.active_drugs else 0.0
        dose_mtx = float(np.mean(ind["mtx"])) if "mtx" in self.active_drugs else 0.0
        dose_vcr = float(np.mean(ind["vcr"])) if "vcr" in self.active_drugs else 1.2

        result = self._sim(dose_6mp, dose_mtx, dose_vcr)

        if not result.get("success"):
            return 1e9, {}, {"t": [], "WBC": [], "ANC": [], "VIPN": []}

        ts   = result["timeseries"]
        wbc  = np.array(ts.get("wbc", []))
        anc  = np.array(ts.get("anc", []))
        vipn = np.array(ts.get("vipn", []))

        if len(wbc) == 0:
            return 1e9, {}, result

        score = 0.0
        # Hedef aralık penaltisi
        score += 140.0 * self.target_penalty(wbc,  1.5, 3.0)
        score += 25.0  * self.target_penalty(anc,  0.5, 2.0)
        # Aşırı baskılama penaltisi
        score += 700.0 * max(0.0, float(np.max(wbc)) - 3.6) ** 2
        score += 350.0 * max(0.0, 1.2 - float(np.min(wbc))) ** 2
        score += 220.0 * max(0.0, 0.4 - float(np.min(anc)))  ** 2
        # VIPN
        if len(vipn) > 0:
            score += 70.0  * max(0.0, 0.75 - float(np.min(vipn))) ** 2
        # Ani düşüş
        if len(wbc) > 1:
            score += 35.0 * float(np.sum(np.clip(-np.diff(wbc) - 1.0, 0, None)**2))
        if len(anc) > 1:
            score += 20.0 * float(np.sum(np.clip(-np.diff(anc) - 0.45, 0, None)**2))
        # Düzgünlük
        score += 0.10 * self.smoothness_penalty(ind["6mp"])
        score += 0.10 * self.smoothness_penalty(ind["mtx"])
        score += 0.20 * self.smoothness_penalty(ind["vcr"])

        metrics = {
            "wbc_target_frac": self.target_fraction(wbc, 1.5, 3.0),
            "anc_target_frac": self.target_fraction(anc, 0.5, 2.0),
            "wbc_min": float(np.min(wbc)),
            "wbc_max": float(np.max(wbc)),
            "anc_min": float(np.min(anc)),
            "vipn_min": float(np.min(vipn)) if len(vipn) > 0 else 1.0,
        }

        # timeseries'i GA formatına çevir (daily arrays)
        t_arr = np.array(ts.get("t", []))
        n_days = int(self.patient_config.get("t_end", 250)) + 1

        def interp_daily(arr):
            if len(arr) == 0 or len(t_arr) == 0:
                return np.zeros(n_days)
            days = np.arange(n_days)
            return np.interp(days, t_arr, arr)

        d6 = np.zeros(n_days)
        dm = np.zeros(n_days)
        dv = np.zeros(n_days)

        if "6mp" in self.active_drugs:
            for w_i, dose in enumerate(ind["6mp"]):
                start = w_i * 7
                end   = min(start + 7, n_days)
                d6[start:end] = dose

        if "mtx" in self.active_drugs:
            for w_i, dose in enumerate(ind["mtx"]):
                day = w_i * 7
                if day < n_days:
                    dm[day] = dose

        if "vcr" in self.active_drugs:
            from app.modules.ode.ode_simulator import T_IND_DEFAULT, T_CONS_DEFAULT, T_REIND_DEFAULT
            # Custom fazlar varsa faz sınırlarını dinamik hesapla
            if self.patient_config.get("custom_phases"):
                cph = self.patient_config["custom_phases"]
                t = 0.0
                _pb = []
                for ph in cph:
                    _pb.append((t, t + ph.get("duration_days", 28), ph))
                    t += ph.get("duration_days", 28)
                T_IND   = _pb[0][1] if len(_pb) > 0 else T_IND_DEFAULT
                T_CONS  = _pb[1][1] if len(_pb) > 1 else T_CONS_DEFAULT
                T_REIND = _pb[2][1] if len(_pb) > 2 else T_REIND_DEFAULT
            else:
                T_IND, T_CONS, T_REIND = T_IND_DEFAULT, T_CONS_DEFAULT, T_REIND_DEFAULT

            vcr_schedule = []
            for d2 in [1,8,15,22]: vcr_schedule.append(d2)
            for d2 in [36,64]:     vcr_schedule.append(d2)
            for d2 in [T_CONS+1, T_CONS+8, T_CONS+15, T_CONS+22]: vcr_schedule.append(int(d2))
            d2 = T_REIND
            while d2 < n_days:
                vcr_schedule.append(int(d2)); d2 += 28
            for cycle_i, day_i in enumerate(vcr_schedule):
                if day_i < n_days and cycle_i < len(ind["vcr"]):
                    dv[day_i] = ind["vcr"][cycle_i]

        # WBC/ANC/VIPN → orijinal t_arr ile birlikte tut (501 nokta)
        # daily_* → gün bazlı array (251 nokta: 0..250)
        # _make_plots bunu bilmeli: days ekseni için ayrı array kullan
        out = {
            "t":         t_arr,
            "WBC":       np.array(ts.get("wbc", [])),
            "ANC":       np.array(ts.get("anc", [])),
            "VIPN":      np.array(ts.get("vipn", [])),
            "daily_6mp": d6,
            "daily_mtx": dm,
            "daily_vcr": dv,
        }

        return float(score), metrics, out

    def crossover(self, p1, p2):
        child = {}
        for key in ["6mp", "mtx", "vcr"]:
            a1, a2 = p1[key], p2[key]
            child[key] = [
                float(self.rng.uniform(0.35, 0.65) * x + (1 - self.rng.uniform(0.35, 0.65)) * y)
                for x, y in zip(a1, a2)
            ]
        return child

    def mutate_array(self, arr, lo, hi, sigma):
        out = []
        prev = None
        for x in arr:
            val = float(np.clip(x + self.rng.normal(0, sigma), lo, hi))
            if prev is not None:
                max_step = 6.0 if hi > 10 else 0.18
                val = float(np.clip(val, prev - max_step, prev + max_step))
                val = float(np.clip(val, lo, hi))
            out.append(val)
            prev = val
        return out

    def mutate(self, ind):
        return {
            "6mp": self.mutate_array(ind["6mp"], *self.bounds_6mp, sigma=2.2)
                   if "6mp" in self.active_drugs else ind["6mp"],
            "mtx": self.mutate_array(ind["mtx"], *self.bounds_mtx, sigma=1.0)
                   if "mtx" in self.active_drugs else ind["mtx"],
            "vcr": self.mutate_array(ind["vcr"], *self.bounds_vcr, sigma=0.06)
                   if "vcr" in self.active_drugs else ind["vcr"],
        }

    def optimize(self):
        population = [self.make_individual() for _ in range(self.pop_size)]
        best_score = np.inf
        best_ind   = None
        best_out   = None
        best_metrics = {}
        history    = []

        for gen in range(1, self.n_generations + 1):
            scored = []
            for ind in population:
                score, metrics, out = self.fitness(ind)
                scored.append((score, metrics, ind, out))
            scored.sort(key=lambda x: x[0])

            if scored[0][0] < best_score:
                best_score   = scored[0][0]
                best_metrics = scored[0][1]
                best_ind     = {
                    "6mp": [round(float(x), 3) for x in scored[0][2]["6mp"]],
                    "mtx": [round(float(x), 3) for x in scored[0][2]["mtx"]],
                    "vcr": [round(float(x), 3) for x in scored[0][2]["vcr"]],
                }
                best_out = scored[0][3]

            history.append({
                "generation":      gen,
                "best_score":      float(best_score),
                **{k: float(v) for k, v in best_metrics.items()},
            })
            print(
                f"Nesil {gen:02d} | Skor={best_score:.3f} | "
                f"WBC={best_metrics.get('wbc_target_frac',0):.1%} | "
                f"ANC={best_metrics.get('anc_target_frac',0):.1%}"
            )

            elites = [x[2] for x in scored[:self.elite_size]]
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = elites[self.rng.integers(0, len(elites))]
                p2 = elites[self.rng.integers(0, len(elites))]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            population = new_pop

        return best_ind, float(best_score), best_metrics, best_out, history
