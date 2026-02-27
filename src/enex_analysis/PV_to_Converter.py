"""
PV-to-Converter system model for energy, entropy, and exergy analysis.

Models a four-stage photovoltaic power conversion chain:
    PV Cell → Controller → Battery → DC/AC Converter

Each stage computes a full energy / entropy / exergy balance,
propagating the electrical output through efficiency losses and
irreversibility generation at each component.
"""

#%%
from dataclasses import dataclass
from . import calc_util as cu
from .constants import k_D, k_d

@dataclass
class PV_to_Converter:
    """Four-stage PV power conversion model with exergy accounting.

    Computes energy, entropy, and exergy balances for each conversion
    stage (PV cell, charge controller, battery, DC/AC inverter).
    All temperature inputs are in Kelvin [K] for thermodynamic
    consistency; Celsius values are converted internally via ``calc_util``.
    """

    # ═══ Default parameters (set in __post_init__) ═════════════════════════

    def __post_init__(self):
        # --- Installation and environmental conditions ---
        self.A_pv = 5.0         # Panel surface area [m²]
        self.alp_pv = 0.9       # Surface absorptivity [-]
        self.I_DN = 500.0       # Direct normal irradiance [W/m²]
        self.I_dH = 150.0       # Diffuse horizontal irradiance [W/m²]
        self.h_o = 15.0         # Outdoor heat transfer coefficient [W/(m²·K)]

        # --- Component efficiencies ---
        self.eta_pv = 0.20      # PV panel efficiency [-]  (typical 17–25 %)
        self.eta_ctrl = 0.95    # Controller efficiency [-] (typical 98–99.5 %)
        self.eta_batt = 0.90    # Battery round-trip efficiency [-] (typical 90–98 %)
        self.eta_DC_AC = 0.95   # DC/AC inverter efficiency [-] (typical 95–99 %)

        # --- Component operating temperatures ---
        self.T0_C      = 20     # Ambient / dead-state temperature [°C]
        self.T_ctrl_C  = 35     # Controller surface temperature [°C]
        self.T_batt_C  = 40     # Battery surface temperature [°C]
        self.T_DC_AC_C = 40     # DC/AC inverter surface temperature [°C]

        # --- Internal Kelvin conversions ---
        self.T0      = cu.C2K(self.T0_C)
        self.T_ctrl  = cu.C2K(self.T_ctrl_C)
        self.T_batt  = cu.C2K(self.T_batt_C)
        self.T_DC_AC = cu.C2K(self.T_DC_AC_C)

    # ═══ Core calculation ══════════════════════════════════════════════════

    def system_update(self):
        """Compute energy, entropy, and exergy balances for all four stages.

        Updates instance attributes for every stage:
            - ``E_pv0`` … ``E_pv3``: electrical output at each stage [W]
            - ``Q_l_*``: heat loss at each component [W]
            - ``S_*``, ``S_g_*``: entropy flows and generation [W/K]
            - ``X_*``, ``X_c_*``: exergy flows and consumption [W]
        """

        # --- Stage 0: Solar input ---
        self.I_sol = self.I_DN + self.I_dH

        # --- Stage 1: PV Cell ---
        # PV surface temperature derived from steady-state energy balance:
        #   A·α·I = E_pv0 + Q_loss  →  T_pv = T0 + I·(α − η) / (2·h_o)
        self.T_pv = self.T0 + (self.I_sol * (self.alp_pv - self.eta_pv)) / (2 * self.h_o)

        # Energy balance (PV)
        self.E_pv0 = self.A_pv * self.eta_pv * self.I_sol
        self.Q_l_pv = 2 * self.A_pv * self.h_o * (self.T_pv - self.T0)

        # Entropy balance (PV)
        self.s_DN = k_D * self.I_DN ** 0.9
        self.s_dH = k_d * self.I_dH ** 0.9
        self.s_sol = self.s_DN + self.s_dH

        self.S_sol = self.A_pv * self.alp_pv * self.s_sol
        self.S_pv0 = (1 / float('inf')) * self.E_pv0
        self.S_l_pv = (1 / self.T_pv) * self.Q_l_pv
        self.S_g_pv = self.S_pv0 + self.S_l_pv - self.S_sol

        # Exergy balance (PV)
        self.X_sol = self.A_pv * self.alp_pv * (self.I_sol - self.s_sol * self.T0)
        self.X_pv0 = self.E_pv0
        self.X_l_pv = (1 - self.T0 / self.T_pv) * self.Q_l_pv
        self.X_c_pv = self.S_g_pv * self.T0

        # --- Stage 2: Controller ---
        # Energy balance (Controller)
        self.E_pv1 = self.eta_ctrl * self.E_pv0
        self.Q_l_ctrl = (1 - self.eta_ctrl) * self.E_pv0

        # Entropy balance (Controller)
        self.S_pv1 = (1 / float('inf')) * self.E_pv1
        self.S_l_ctrl = (1 / self.T_ctrl) * self.Q_l_ctrl
        self.S_g_ctrl = self.S_pv1 + self.S_l_ctrl - self.S_pv0

        # Exergy balance (Controller)
        self.X_pv1 = self.E_pv1 - self.S_pv1 * self.T0
        self.X_l_ctrl = self.Q_l_ctrl - self.S_l_ctrl * self.T0
        self.X_c_ctrl = self.S_g_ctrl * self.T0

        # --- Stage 3: Battery ---
        # Energy balance (Battery)
        self.E_pv2 = self.eta_batt * self.E_pv1
        self.Q_l_batt = (1 - self.eta_batt) * self.E_pv1

        # Entropy balance (Battery)
        self.S_pv2 = (1 / float('inf')) * self.E_pv2
        self.S_l_batt = (1 / self.T_batt) * self.Q_l_batt
        self.S_g_batt = self.S_pv2 + self.S_l_batt - self.S_pv1

        # Exergy balance (Battery)
        self.X_pv2 = self.E_pv2 - self.S_pv2 * self.T0
        self.X_l_batt = self.Q_l_batt - self.S_l_batt * self.T0
        self.X_c_batt = self.S_g_batt * self.T0

        # --- Stage 4: DC/AC Converter ---
        # Energy balance (DC/AC)
        self.E_pv3 = self.eta_DC_AC * self.E_pv2
        self.Q_l_DC_AC = (1 - self.eta_DC_AC) * self.E_pv2

        # Entropy balance (DC/AC)
        self.S_pv3 = (1 / float('inf')) * self.E_pv3
        self.S_l_DC_AC = (1 / self.T_DC_AC) * self.Q_l_DC_AC
        self.S_g_DC_AC = self.S_pv3 + self.S_l_DC_AC - self.S_pv2

        # Exergy balance (DC/AC)
        self.X_pv3 = self.E_pv3 - self.S_pv3 * self.T0
        self.X_l_DC_AC = self.Q_l_DC_AC - self.S_l_DC_AC * self.T0
        self.X_c_DC_AC = self.S_g_DC_AC * self.T0

# %%
