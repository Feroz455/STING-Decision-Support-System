import numpy as np
import matplotlib.pyplot as plt


class TriplePlotter:
    def __init__(self):
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def _setup(self, title):
        fig = plt.figure(figsize=(14, 8.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.25], hspace=0.10)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
        fig.suptitle(title, y=0.98)
        return fig, ax_top, ax_bot

    def _dose_panel_all(self, ax, d6, dm, dv):
        days = np.arange(len(d6))
        ax.step(days, d6, where="post", linewidth=2.0, label="6-MP")
        ax.stem(days, dm, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="MTX")
        ax.stem(days, dv, linefmt="C2-", markerfmt="C2s", basefmt=" ", label="VCR")
        ax.set_ylabel("Doz")
        ax.set_xlabel("Gün")
        ax.legend(loc="upper right", ncol=3, frameon=True)

    def _dose_panel_vcr_only(self, ax, dv):
        days = np.arange(len(dv))
        ax.stem(days, dv, linefmt="C2-", markerfmt="C2s", basefmt=" ", label="VCR")
        ax.set_ylabel("VCR dozu")
        ax.set_xlabel("Gün")
        ax.legend(loc="upper right", frameon=True)

    def plot_wbc(self, result, save_path=None):
        fig, ax1, ax2 = self._setup("WBC ve Doz Planı")
        t = result["t"]
        wbc = result["WBC"]
        d6, dm, dv = result["daily_6mp"], result["daily_mtx"], result["daily_vcr"]
        ax1.axhspan(1.5, 3.0, color="gray", alpha=0.22, label="Hedef aralık")
        ax1.plot(t, wbc, linewidth=3.0, label="WBC")
        ax1.set_ylabel("WBC")
        ax1.set_ylim(1.0, max(5.5, float(np.max(wbc)) + 0.4))
        hit = 100.0 * np.mean((wbc >= 1.5) & (wbc <= 3.0))
        txt = f"WBC min = {np.min(wbc):.3f}\nWBC max = {np.max(wbc):.3f}\nHedefte kalma = {hit:.1f}%"
        ax1.text(0.02, 0.96, txt, transform=ax1.transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9))
        ax1.legend(loc="upper right", frameon=True)
        self._dose_panel_all(ax2, d6, dm, dv)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_anc(self, result, save_path=None):
        fig, ax1, ax2 = self._setup("ANC ve Doz Planı")
        t = result["t"]
        anc = result["ANC"]
        d6, dm, dv = result["daily_6mp"], result["daily_mtx"], result["daily_vcr"]
        ax1.axhspan(0.5, 1.5, color="gray", alpha=0.22, label="Hedef aralık")
        ax1.plot(t, anc, linewidth=3.0, label="ANC")
        ax1.set_ylabel("ANC")
        ax1.set_ylim(0.2, max(2.0, float(np.max(anc)) + 0.3))
        hit = 100.0 * np.mean((anc >= 0.5) & (anc <= 1.5))
        txt = f"ANC min = {np.min(anc):.3f}\nANC max = {np.max(anc):.3f}\nHedefte kalma = {hit:.1f}%"
        ax1.text(0.02, 0.96, txt, transform=ax1.transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9))
        ax1.legend(loc="upper right", frameon=True)
        self._dose_panel_all(ax2, d6, dm, dv)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_vipn(self, result, save_path=None):
        fig, ax1, ax2 = self._setup("VIPN ve VCR Dozu")
        t = result["t"]
        vipn = result["VIPN"]
        dv = result["daily_vcr"]
        ax1.plot(t, vipn, linewidth=3.0, label="VIPN")
        ax1.axhline(0.78, linestyle="--", linewidth=2.0, color="C1", label="VIPN eşik")
        ax1.set_ylabel("VIPN")
        ax1.set_ylim(min(0.75, float(np.min(vipn)) - 0.02), 1.02)
        txt = f"VIPN min = {np.min(vipn):.3f}\n(Bu grafikte sadece VCR dozu gösterilir)"
        ax1.text(0.02, 0.96, txt, transform=ax1.transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.35", alpha=0.9))
        ax1.legend(loc="lower left", frameon=True)
        self._dose_panel_vcr_only(ax2, dv)
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
