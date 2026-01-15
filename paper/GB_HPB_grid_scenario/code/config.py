"""
Configuration parameters for DHW system analysis.

This module contains default parameters used in the original paper
and provides dataclasses for scenario configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TankConfig:
    """Tank geometry and thermal properties configuration."""
    r0: float = 0.2  # Tank radius [m]
    H: float = 0.8  # Tank height [m]
    x_shell: float = 0.01  # Shell thickness [m]
    x_ins: float = 0.10  # Insulation thickness [m]
    k_shell: float = 25.0  # Shell thermal conductivity [W/mK]
    k_ins: float = 0.03  # Insulation thermal conductivity [W/mK]
    h_o: float = 15.0  # Overall heat transfer coefficient [W/m²K]


@dataclass
class TemperatureConfig:
    """Temperature configuration for DHW system."""
    T_w_tank: float = 60.0  # Tank water temperature [°C]
    T_w_sup: float = 10.0  # Supply water temperature [°C]
    T_w_serv: float = 45.0  # Service water temperature [°C]
    T0: float = 0.0  # Reference temperature [°C]
    T_exh: Optional[float] = 70.0  # Exhaust gas temperature [°C] (for GB only)


@dataclass
class LoadConfig:
    """Load configuration for DHW system."""
    dV_w_serv: float = 1.2  # Service water flow rate [L/min]


@dataclass
class ElectricBoilerConfig:
    """Configuration for Electric Boiler."""
    tank: TankConfig = None
    temperature: TemperatureConfig = None
    load: LoadConfig = None
    
    def __post_init__(self):
        if self.tank is None:
            self.tank = TankConfig()
        if self.temperature is None:
            self.temperature = TemperatureConfig()
        if self.load is None:
            self.load = LoadConfig()


@dataclass
class GasBoilerConfig:
    """Configuration for Gas Boiler."""
    eta_comb: float = 0.9  # Combustion efficiency [-]
    tank: TankConfig = None
    temperature: TemperatureConfig = None
    load: LoadConfig = None
    
    def __post_init__(self):
        if self.tank is None:
            self.tank = TankConfig()
        if self.temperature is None:
            self.temperature = TemperatureConfig()
        if self.load is None:
            self.load = LoadConfig()


@dataclass
class HeatPumpBoilerConfig:
    """Configuration for Heat Pump Boiler."""
    eta_fan: float = 0.6  # Fan efficiency [-]
    COP: float = 2.5  # Coefficient of Performance [-]
    dP: float = 200.0  # Pressure difference [Pa]
    T_a_ext_out: Optional[float] = None  # External air outlet temperature [°C]
    T_r_ext: Optional[float] = None  # External refrigerant temperature [°C]
    T_r_tank: Optional[float] = None  # Tank refrigerant temperature [°C]
    tank: TankConfig = None
    temperature: TemperatureConfig = None
    load: LoadConfig = None
    
    def __post_init__(self):
        if self.tank is None:
            self.tank = TankConfig()
        if self.temperature is None:
            self.temperature = TemperatureConfig()
        if self.load is None:
            self.load = LoadConfig()
        
        # Set default values based on T0
        if self.T_a_ext_out is None:
            self.T_a_ext_out = self.temperature.T0 - 5.0
        if self.T_r_ext is None:
            self.T_r_ext = self.temperature.T0 - 10.0
        if self.T_r_tank is None:
            self.T_r_tank = self.temperature.T_w_tank + 5.0


@dataclass
class PaperScenarioConfig:
    """Complete scenario configuration matching the original paper."""
    eb_config: ElectricBoilerConfig = None
    gb_config: GasBoilerConfig = None
    hpb_config: HeatPumpBoilerConfig = None
    
    def __post_init__(self):
        if self.eb_config is None:
            self.eb_config = ElectricBoilerConfig()
        if self.gb_config is None:
            self.gb_config = GasBoilerConfig()
        if self.hpb_config is None:
            self.hpb_config = HeatPumpBoilerConfig()


# Default paper scenario (can be adjusted based on actual paper values)
PAPER_DEFAULT_SCENARIO = PaperScenarioConfig()

