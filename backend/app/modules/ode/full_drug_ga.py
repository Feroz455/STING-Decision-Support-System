# full_drug_ga.py
# -*- coding: utf-8 -*-
"""
Protocol-constrained real-coded genetic algorithm for the TEN-DRUG engine.

Optimizes DOSES for EIGHT agents:
  6-MP, MTX, VCR, DNR, PEG-ASP, corticosteroid (Pred+Dex),
  Copanlisib [REPOSITIONING], Novobiocin [REPOSITIONING]

Cyclophosphamide and cytarabine stay at calibrated nominal doses (fixed).

GENE LAYOUT — 18 genes total (was 16):

    6-MP        : consolidation level, maintenance level             (2)
    MTX         : consolidation weekly, maintenance weekly           (2)
    VCR         : induction, reinduction, maintenance                (3)
    DNR         : induction, reinduction                             (2)
    PEG-ASP     : G4, G36, G57, G91 (per dose)                      (4)
    corticost.  : prednisolone, dex-reinduction, dex-maintenance     (3)
    Copanlisib  : re-induction dose (mg, IV weekly)                  (1)  ← NEW
    Novobiocin  : maintenance dose (mg/day, oral)                    (1)  ← NEW
                                                              total = 18

Copanlisib and Novobiocin are only optimized when include_repositioning=True
is passed to FullDrugALLModel. When the flag is False the two genes are
carried in the chromosome but chromosome_to_dose_plan() zeroes them out,
so backwards-compatibility with the 16-gene runs is preserved.

Fitness: constrained-optimization framing identical to the 16-gene version,
with two additional SOFT penalty terms for the repositioning candidates:
  - cop_pen : Copanlisib efficacy contribution (Lt reduction at re-ind end)
  - nov_pen : Novobiocin maintenance-phase Lt control

Academic / in-silico only; not clinical dosing advice.
"""

import numpy as np


class FullDrugAdaptiveGA:
    def __init__(self, model, patient, seed=42):
        self.model   = model
        self.patient = patient
        self.rng     = np.random.default_rng(seed)
        self.t       = model.t
        self.dt      = model.dt
        self.BSA     = model.BSA

        # Repositioning flag — inherited from model
        self.include_repositioning = bool(model.include_repositioning)

        # Fixed protocol days — read FROM the engine so GA and ODE agree
        self.vcr_days = np.asarray(model.VCR_DAYS,  dtype=float)
        self.dnr_days = np.asarray(model.DNR_DAYS,  dtype=float)
        self.mtx_days = np.asarray(model.MTX_DAYS,  dtype=float)
        self.peg_days = np.asarray(model.peg_params["dose_days"], dtype=float)
        self.T_IND    = model.T_IND
        self.T_CONS   = model.T_CONS
        self.T_REIND  = model.T_REIND

        # Phase masks
        self.vcr_phase = self._phase_index(self.vcr_days, max_index=2)
        self.dnr_phase = self._phase_index(self.dnr_days, max_index=1)

        # Gene layout: 16 backbone + 2 repositioning = 18
        self.n_peg = len(self.peg_days)
        self.gene_names = (
            ["6mp_cons", "6mp_maint",
             "mtx_cons", "mtx_maint",
             "vcr_ind",  "vcr_reind", "vcr_maint",
             "dnr_ind",  "dnr_reind"]
            + [f"peg_{int(d)}" for d in self.peg_days]
            + ["pred", "dex_reind", "dex_maint"]
            + ["cop_dose", "nov_dose"]          # NEW — repositioning genes
        )
        self.chromosome_length = len(self.gene_names)   # 18
        self.bounds = self._build_bounds()

        # ── Fitness weights ─────────────────────────────────────────────────
        # Hard constraints (large weights)
        self.W_VIPN      = 4000.0
        self.W_BRR       = 6000.0
        self.W_MRD       = 200.0
        self.W_DNR_CARD  = 0.01
        self.W_PEG       = 0.02
        self.W_ASN       = 0.10
        # Soft objective
        self.W_MAINT_WBC = 12.0
        self.W_MAINT_ANC = 12.0
        self.W_DOSE      = 1.0
        # Repositioning soft terms
        self.W_COP       = 5.0    # reward Copanlisib Lt reduction at re-ind end
        self.W_NOV       = 3.0    # reward Novobiocin Lt control in maintenance

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _phase_index(self, days, max_index=2):
        idx = np.zeros(len(days), dtype=int)
        for i, d in enumerate(days):
            if d < self.T_IND:
                idx[i] = 0
            elif self.T_CONS <= d < self.T_REIND:
                idx[i] = 1
            else:
                idx[i] = 2
        return np.clip(idx, 0, max_index)

    def _build_bounds(self):
        b = [
            (25.0, 50.0), (25.0, 50.0),              # 6-MP cons, maint (mg/day)
            (8.0,  20.0), (8.0,  20.0),              # MTX cons, maint (mg/week)
            (0.60, 1.50), (0.60, 1.50), (0.60, 1.50), # VCR ind/reind/maint (mg)
            (12.5, 25.0), (12.5, 25.0),              # DNR ind/reind (mg/m²)
        ]
        b += [(2000.0, 2500.0)] * self.n_peg          # PEG per dose (IU/m²)
        b += [(45.0, 60.0), (7.5, 10.0), (4.5, 6.0)] # steroids (mg/m²/day)
        # Repositioning bounds — clinically referenced ranges
        b += [(30.0,  60.0)]   # Copanlisib: 30–60 mg IV (weekly re-ind; Dreyling 2017)
        b += [(250.0, 500.0)]  # Novobiocin: 250–500 mg/day oral (Albrecht 2000; Liu 2020)
        return np.array(b, dtype=float)

    def reference_chromosome(self):
        """Nominal protocol → reproduces the simulation exactly."""
        ref = [50.0, 50.0,           # 6-MP
               20.0, 20.0,           # MTX
               1.50, 1.50, 1.50,     # VCR
               25.0, 25.0]           # DNR (mg/m²)
        ref += [2500.0] * self.n_peg # PEG
        ref += [60.0, 10.0, 6.0]     # steroids
        ref += [0.8 * self.model.weight, 500.0]   # COP (mg), NOV (mg/day) — nominal
        return np.array(ref, dtype=float)

    def clip_to_bounds(self, ind):
        return np.clip(ind, self.bounds[:, 0], self.bounds[:, 1])

    def initialize_population(self, population_size):
        ref  = self.reference_chromosome()
        pop  = [ref.copy()]
        span = self.bounds[:, 1] - self.bounds[:, 0]
        for k in range(population_size - 1):
            ind = ref + self.rng.normal(0.0, 0.18 * span)
            # Bias early individuals toward lower VCR (relieves VIPN violation)
            if k < (population_size - 1) // 2:
                ind[4:7] = (self.bounds[4:7, 0]
                            + self.rng.uniform(0, 0.4, 3)
                            * (self.bounds[4:7, 1] - self.bounds[4:7, 0]))
            pop.append(self.clip_to_bounds(ind))
        return np.array(pop)

    # ── Chromosome → dose_plan ───────────────────────────────────────────────

    def chromosome_to_dose_plan(self, chromosome):
        c = self.clip_to_bounds(chromosome)
        (sixmp_cons, sixmp_maint,
         mtx_cons, mtx_maint,
         vcr_ind, vcr_reind, vcr_maint,
         dnr_ind, dnr_reind) = c[:9]
        peg          = c[9 : 9 + self.n_peg]
        pred_m2, dex_ri_m2, dex_m_m2 = c[9 + self.n_peg : 9 + self.n_peg + 3]
        cop_dose_mg  = float(c[-2])
        nov_dose_mg  = float(c[-1])

        # 6-MP daily array
        six_mp_daily = np.zeros_like(self.t, dtype=float)
        six_mp_daily[(self.t >= self.T_IND)   & (self.t < self.T_CONS)]  = sixmp_cons
        six_mp_daily[(self.t >= self.T_REIND) & (self.t <= self.model.total_days)] = sixmp_maint

        # MTX per-event
        mtx_doses = np.where(self.mtx_days < self.T_CONS, mtx_cons, mtx_maint)

        # VCR per-event by phase
        vcr_levels = np.array([vcr_ind, vcr_reind, vcr_maint])
        vcr_doses  = vcr_levels[self.vcr_phase]

        # DNR per-event by phase (mg/m² → absolute mg)
        dnr_levels_m2 = np.array([dnr_ind, dnr_reind])
        dnr_doses     = dnr_levels_m2[self.dnr_phase] * self.BSA

        plan = {
            "include_repositioning": self.include_repositioning,
            "six_mp_daily":    six_mp_daily,
            "mtx_doses":       mtx_doses.copy(),
            "vcr_doses":       vcr_doses.copy(),
            "dnr_doses":       dnr_doses.copy(),
            "peg_doses":       peg.copy(),
            "pred_dose":       float(pred_m2   * self.BSA),
            "dex_reind_dose":  float(dex_ri_m2 * self.BSA),
            "dex_maint_dose":  float(dex_m_m2  * self.BSA),
        }

        # Repositioning doses — only written when flag is on;
        # engine's u_cop/u_nb already return 0.0 when include_repositioning=False
        if self.include_repositioning:
            # Copanlisib: override model's nominal dose for COP_DAYS events
            n_cop = len(self.model.COP_DAYS)
            plan["cop_doses"] = np.full(n_cop, cop_dose_mg) if n_cop > 0 else np.array([])
            # Novobiocin: scalar daily dose (engine reads via _scalar("nb_dose", ...))
            plan["nb_dose"] = nov_dose_mg

        return plan

    # ── Fitness ─────────────────────────────────────────────────────────────

    def calculate_score(self, result, dose_plan):
        t   = result["t"]
        WBC = result["WBC"]
        ANC = result["ANC"]
        PEG = result["PEG_A"]
        ASN = result["ASN"]
        Lt  = result["Lt"]
        p   = self.patient

        maint = t >= self.T_REIND
        reind = (t >= self.T_CONS) & (t < self.T_REIND)

        wbc_pen = (np.mean(np.maximum(0.0, p.wbc_low  - WBC[maint]) ** 2
                           + np.maximum(0.0, WBC[maint] - p.wbc_high) ** 2)
                   if maint.any() else 0.0)
        anc_pen = (np.mean(np.maximum(0.0, p.anc_low  - ANC[maint]) ** 2
                           + np.maximum(0.0, ANC[maint] - p.anc_high) ** 2)
                   if maint.any() else 0.0)

        vipn_pen     = max(0.0, p.vipn_threshold - result["VIPN_min"]) ** 2

        first_peg    = float(self.peg_days[0])
        pmask        = (t >= first_peg) & (t <= 120.0)
        peg_pen      = (np.mean(np.maximum(0.0, p.peg_activity_threshold - PEG[pmask]) ** 2)
                        if pmask.any() else 0.0)
        asn_pen      = (np.mean(np.maximum(0.0, ASN[pmask] - p.asn_control_high) ** 2)
                        if pmask.any() else 0.0)

        dnr_card_pen = max(0.0, result["cum_DNR_final"] - p.dnr_cum_threshold_ped) ** 2

        brr_pen      = max(0.0, p.brr_d8_target - result["BRR_d8"]) ** 2
        mrd          = max(result["EOI_MRD"], 1e-15)
        mrd_pen      = max(0.0, np.log10(mrd) - np.log10(p.mrd_d29_target)) ** 2

        dose_pen = (
            0.0007   * np.mean(dose_plan["six_mp_daily"] ** 2)
            + 0.001  * np.mean(dose_plan["mtx_doses"] ** 2)
            + 0.08   * np.mean(dose_plan["vcr_doses"] ** 2)
            + 0.0008 * np.mean((dose_plan["dnr_doses"] / max(self.BSA, 1e-9)) ** 2)
            + 0.0000003 * np.mean(dose_plan["peg_doses"] ** 2)
        )

        # ── Repositioning soft terms ─────────────────────────────────────────
        # Reward: lower Lt at end of re-induction (Copanlisib window)
        cop_pen = 0.0
        nov_pen = 0.0
        if self.include_repositioning:
            if reind.any():
                Lt_reind_end = float(Lt[reind][-1])
                # Penalize if Lt at re-induction end > 1% of initial burden
                cop_pen = max(0.0, Lt_reind_end - 0.01) ** 2
            if maint.any():
                # Reward keeping Lt suppressed in maintenance
                lt_maint_mean = float(np.mean(Lt[maint]))
                nov_pen = max(0.0, lt_maint_mean - 0.005) ** 2

        return (self.W_VIPN     * vipn_pen
                + self.W_BRR   * brr_pen
                + self.W_MRD   * mrd_pen
                + self.W_DNR_CARD * dnr_card_pen
                + self.W_PEG   * peg_pen
                + self.W_ASN   * asn_pen
                + self.W_MAINT_WBC * wbc_pen
                + self.W_MAINT_ANC * anc_pen
                + self.W_DOSE  * dose_pen
                + self.W_COP   * cop_pen
                + self.W_NOV   * nov_pen)

    # ── GA operators ────────────────────────────────────────────────────────

    def tournament_selection(self, population, scores, k=3):
        ids = self.rng.choice(len(population), size=k, replace=False)
        return population[ids[np.argmin(scores[ids])]].copy()

    def crossover(self, p1, p2, rate=0.85):
        if self.rng.random() > rate:
            return p1.copy(), p2.copy()
        a  = self.rng.uniform(0.25, 0.75, size=self.chromosome_length)
        return (self.clip_to_bounds(a * p1 + (1 - a) * p2),
                self.clip_to_bounds(a * p2 + (1 - a) * p1))

    def mutate(self, ind, rate=0.20):
        out  = ind.copy()
        span = self.bounds[:, 1] - self.bounds[:, 0]
        for i in range(self.chromosome_length):
            if self.rng.random() < rate:
                out[i] += self.rng.normal(0.0, 0.12 * span[i])
        return self.clip_to_bounds(out)

    # ── Main loop ────────────────────────────────────────────────────────────

    def optimize_six_doses_ga(self, generations=15, population_size=20,
                              elite_count=3, mutation_rate=0.20,
                              crossover_rate=0.85, verbose=True):
        pop        = self.initialize_population(population_size)
        best_score = np.inf
        best_ind   = None
        best_result = None
        history    = []

        for gen in range(1, generations + 1):
            scores  = np.empty(population_size)
            results = [None] * population_size
            for i in range(population_size):
                dp = self.chromosome_to_dose_plan(pop[i])
                try:
                    res       = self.model.simulate_all(dp)
                    scores[i] = self.calculate_score(res, dp)
                    results[i] = res
                except Exception:
                    scores[i] = 1e9

            order = np.argsort(scores)
            if scores[order[0]] < best_score:
                best_score  = scores[order[0]]
                best_ind    = pop[order[0]].copy()
                best_result = results[order[0]]
            history.append(best_score)

            if verbose and best_result is not None:
                r = best_result
                repo_str = ""
                if self.include_repositioning:
                    repo_str = (f" | Lt_reind={r['Lt'][(self.t >= self.T_CONS) & (self.t < self.T_REIND)][-1]:.4f}"
                                f" Lt_maint={float(np.mean(r['Lt'][self.t >= self.T_REIND])):.4f}")
                print(f"Nesil {gen:02d} | skor={best_score:.4f} | "
                      f"BRR_d8={r['BRR_d8']*100:.2f}% MRD={r['EOI_MRD']:.2e} | "
                      f"VIPNmin={r['VIPN_min']:.3f} | "
                      f"WBCnad={r['WBC_min']:.2f} ANCnad={r['ANC_min']:.2f} | "
                      f"idame WBC%{r['WBC_in_target_maint']:.0f}/ANC%{r['ANC_in_target_maint']:.0f}"
                      + repo_str,
                      flush=True)

            new_pop = [pop[order[e]].copy() for e in range(min(elite_count, population_size))]
            while len(new_pop) < population_size:
                p1      = self.tournament_selection(pop, scores)
                p2      = self.tournament_selection(pop, scores)
                c1, c2  = self.crossover(p1, p2, crossover_rate)
                new_pop.append(self.mutate(c1, mutation_rate))
                if len(new_pop) < population_size:
                    new_pop.append(self.mutate(c2, mutation_rate))
            pop = np.array(new_pop)

        dose_plan = self.chromosome_to_dose_plan(best_ind)
        result    = self.model.simulate_all(dose_plan)
        score     = self.calculate_score(result, dose_plan)
        self.best_chromosome = best_ind
        self.history         = history
        return dose_plan, result, score

    def summarize(self, chromosome=None):
        """Human-readable phase-level dose table for the best (or given) plan."""
        if chromosome is None:
            chromosome = getattr(self, "best_chromosome", None)
        if chromosome is None:
            raise ValueError(
                "summarize(): no chromosome available. Either run "
                "optimize_six_doses_ga() first or pass summarize(chromosome=...).")
        c = self.clip_to_bounds(chromosome)
        summary = {self.gene_names[i]: round(float(c[i]), 3)
                   for i in range(self.chromosome_length)}
        summary["_repositioning_active"] = self.include_repositioning
        return summary
