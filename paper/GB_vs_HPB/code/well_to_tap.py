"""
Well-to-tap analysis module for DHW systems.

This module models the complete energy flow from power generation/ fuel extraction
through transmission/distribution to building and DHW system consumption.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class PowerPlantParams:
    """Parameters for power generation plant."""
    thermal_efficiency: float  # Thermal efficiency [-]
    exergy_efficiency: float  # Exergy efficiency [-]
    transmission_loss: float = 0.05  # Transmission loss fraction [-]
    distribution_loss: float = 0.03  # Distribution loss fraction [-]
    
    @property
    def total_grid_loss(self) -> float:
        """Total grid loss including transmission and distribution."""
        return 1.0 - (1.0 - self.transmission_loss) * (1.0 - self.distribution_loss)


@dataclass
class FuelSupplyParams:
    """Parameters for fuel supply chain (for gas boiler)."""
    extraction_efficiency: float = 0.95  # Fuel extraction efficiency [-]
    transport_efficiency: float = 0.98  # Fuel transport efficiency [-]
    distribution_efficiency: float = 0.99  # Fuel distribution efficiency [-]
    
    @property
    def total_supply_efficiency(self) -> float:
        """Total fuel supply chain efficiency."""
        return (self.extraction_efficiency * 
                self.transport_efficiency * 
                self.distribution_efficiency)


# Default parameters for different generation technologies
# Validation: Efficiency parameters verified against IPCC AR6, IEA, and official sources
# Note: These parameters represent thermal/exergy efficiency only. For CO2 emission factors and 
#       primary energy factors, see grid_mix_analysis.py DEFAULT_TECH_PARAMS
DEFAULT_PLANT_PARAMS = {
    # Source: Ministry of Heavy Industries India (AUSC technology), IPCC AR6 WG3
    # Reference: https://heavyindustries.gov.in/en/advanced-ultra-supercritical-adv-usc-technology-thermal-power-plants
    # Validation: ✓ 일치 - 현대 초초임계 석탄 발전 효율(37~45% 범위) 및 엑서지 효율과 부합
    # Note: AUSC 기술 기반 화력발전소는 약 46% 효율을 달성하며, 일반 초초임계 발전은 41-42%, 아임계 발전은 약 38% 효율입니다.
    #       코드의 40%는 보수적인 값으로 설정되었습니다. 엑서지 효율 35%도 보고된 석탄 발전의 에너지-엑서지 효율(약 36% 전후)과 일치합니다.
    'coal': PowerPlantParams(
        thermal_efficiency=0.40,
        exergy_efficiency=0.35,
        transmission_loss=0.05,
        distribution_loss=0.03
    ),
    # Source: IPCC AR6 WG3, US DOE/EIA, modern CCGT performance
    # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
    # Validation: ✓ 일치 - 현대 가스복합화력(CCGT)의 효율(55~60%) 및 엑서지 효율과 부합
    # Note: 코드의 열효율 50%는 보수적인 값이며, 실제 현대 CCGT는 55~60% 효율을 달성합니다.
    #       엑서지 효율 45%는 가스 터빈의 카르노 효율(약 60%대) 대비 손실를 감안하면 합리적인 수치입니다.
    'gas': PowerPlantParams(
        thermal_efficiency=0.50,
        exergy_efficiency=0.45,
        transmission_loss=0.05,
        distribution_loss=0.03
    ),
    # Source: IPCC AR6 WG3, WNA (World Nuclear Association), PWR nuclear efficiency
    # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
    # Validation: ✓ 일치 - 경수로 원전의 열효율 ~33%에 대응하며, 엑서지 효율 30%도 증기터빈 사이클의 이용가능에너지 효율에 부합
    # Note: 원자력 발전의 열효율은 증기터빈 사이클의 열역학적 한계로 인해 일반적으로 33% 내외입니다.
    'nuclear': PowerPlantParams(
        thermal_efficiency=0.33,
        exergy_efficiency=0.30,
        transmission_loss=0.05,
        distribution_loss=0.03
    ),
    # Source: IPCC AR6 WG3, renewable energy direct conversion
    # Validation: ✓ 일치 - 재생에너지는 연료 투입 없이 자연에너지를 직접 전력으로 변환하므로 효율 1.0으로 처리
    # Note: 재생에너지(수력, 풍력, 태양광 등)는 연료 연소 과정이 없으므로 열효율과 엑서지 효율 모두 1.0으로 설정됩니다.
    #       실제 변환 효율(예: 풍력 터빈의 Betz 한계, 태양광 모듈 효율 등)은 grid_mix_analysis.py의 재생에너지 하위 기술별 파라미터에 반영됩니다.
    'renewable': PowerPlantParams(
        thermal_efficiency=1.0,  # Direct conversion
        exergy_efficiency=1.0,
        transmission_loss=0.05,
        distribution_loss=0.03
    )
}

# Default natural gas supply chain parameters
DEFAULT_NG_SUPPLY_PARAMS = FuelSupplyParams(
    extraction_efficiency=0.95,
    transport_efficiency=0.98,
    distribution_efficiency=0.99
)


class WellToTapAnalyzer:
    """
    Analyzer for well-to-tap energy and exergy flows.
    
    This class calculates energy, exergy, and CO2 flows through the entire
    chain from fuel extraction/power generation to DHW service delivery.
    """
    
    def __init__(
        self,
        plant_params: Dict[str, PowerPlantParams] = None,
        ng_supply_params: FuelSupplyParams = None
    ):
        """
        Initialize well-to-tap analyzer.
        
        Parameters:
        -----------
        plant_params : dict, optional
            Dictionary of power plant parameters by technology type.
            If None, uses DEFAULT_PLANT_PARAMS.
        ng_supply_params : FuelSupplyParams, optional
            Natural gas supply chain parameters.
            If None, uses DEFAULT_NG_SUPPLY_PARAMS.
        """
        self.plant_params = plant_params or DEFAULT_PLANT_PARAMS
        self.ng_supply_params = ng_supply_params or DEFAULT_NG_SUPPLY_PARAMS
    
    def analyze_electric_system(
        self,
        building_electricity_demand: float,
        generation_mix: Dict[str, float],
        tech_co2_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze well-to-tap for electric system (EB or HPB).
        
        Parameters:
        -----------
        building_electricity_demand : float
            Electricity demand at building [kWh]
        generation_mix : dict
            Dictionary of generation mix fractions {tech: fraction}
        tech_co2_factors : dict
            Dictionary of CO2 emission factors {tech: factor [kg CO2/kWh]}
        
        Returns:
        --------
        dict
            Dictionary containing:
            - primary_energy_input: Total primary energy input [kWh]
            - exergy_consumption_generation: Exergy consumption in generation [kWh]
            - exergy_consumption_grid: Exergy consumption in grid [kWh]
            - exergy_consumption_total: Total exergy consumption [kWh]
            - co2_emissions: Total CO2 emissions [kg CO2]
            - stage_breakdown: Breakdown by stage
        """
        # Calculate weighted average plant parameters
        weighted_efficiency = sum(
            mix * self.plant_params[tech].thermal_efficiency
            for tech, mix in generation_mix.items()
        )
        weighted_exergy_efficiency = sum(
            mix * self.plant_params[tech].exergy_efficiency
            for tech, mix in generation_mix.items()
        )
        weighted_grid_loss = sum(
            mix * self.plant_params[tech].total_grid_loss
            for tech, mix in generation_mix.items()
        )
        
        # Calculate electricity at generation output
        electricity_at_gen = building_electricity_demand / (1.0 - weighted_grid_loss)
        
        # Calculate primary energy input
        primary_energy_input = electricity_at_gen / weighted_efficiency
        
        # Calculate exergy consumption in generation
        exergy_consumption_generation = (
            primary_energy_input - 
            electricity_at_gen * weighted_exergy_efficiency
        )
        
        # Calculate exergy consumption in grid
        exergy_consumption_grid = (
            electricity_at_gen * weighted_exergy_efficiency - 
            building_electricity_demand
        )
        
        # Calculate CO2 emissions
        co2_emissions = sum(
            mix * tech_co2_factors[tech] * electricity_at_gen
            for tech, mix in generation_mix.items()
        )
        
        return {
            'primary_energy_input': primary_energy_input,
            'exergy_consumption_generation': exergy_consumption_generation,
            'exergy_consumption_grid': exergy_consumption_grid,
            'exergy_consumption_total': (
                exergy_consumption_generation + exergy_consumption_grid
            ),
            'co2_emissions': co2_emissions,
            'stage_breakdown': {
                'generation': {
                    'primary_energy': primary_energy_input,
                    'exergy_consumption': exergy_consumption_generation,
                    'output': electricity_at_gen
                },
                'grid': {
                    'input': electricity_at_gen,
                    'exergy_consumption': exergy_consumption_grid,
                    'output': building_electricity_demand
                }
            }
        }
    
    def analyze_gas_system(
        self,
        building_gas_demand: float,
        ng_hhv: float = 50.0,  # Natural gas HHV [MJ/kg]
        ng_co2_factor: float = 2.75  # CO2 emission factor [kg CO2/kg NG]
    ) -> Dict[str, float]:
        """
        Analyze well-to-tap for natural gas system (GB).
        
        Parameters:
        -----------
        building_gas_demand : float
            Natural gas demand at building [kWh]
        ng_hhv : float
            Natural gas higher heating value [MJ/kg]
        ng_co2_factor : float
            CO2 emission factor [kg CO2/kg NG]
        
        Returns:
        --------
        dict
            Dictionary containing:
            - primary_energy_input: Total primary energy input [kWh]
            - exergy_consumption_supply: Exergy consumption in supply chain [kWh]
            - exergy_consumption_total: Total exergy consumption [kWh]
            - co2_emissions: Total CO2 emissions [kg CO2]
            - stage_breakdown: Breakdown by stage
        """
        # Convert kWh to MJ
        building_demand_mj = building_gas_demand * 3.6
        
        # Calculate natural gas mass required at building
        ng_mass_at_building = building_demand_mj / ng_hhv
        
        # Calculate natural gas mass at extraction (accounting for supply losses)
        ng_mass_at_extraction = (
            ng_mass_at_building / self.ng_supply_params.total_supply_efficiency
        )
        
        # Calculate primary energy input
        primary_energy_input = ng_mass_at_extraction * ng_hhv / 3.6  # Convert to kWh
        
        # Calculate exergy consumption in supply chain
        # Assuming exergy content is approximately 0.93 of HHV for natural gas
        ng_exergy_factor = 0.93
        exergy_at_extraction = primary_energy_input * ng_exergy_factor
        exergy_at_building = building_gas_demand * ng_exergy_factor
        exergy_consumption_supply = exergy_at_extraction - exergy_at_building
        
        # Calculate CO2 emissions
        co2_emissions = ng_mass_at_building * ng_co2_factor
        
        return {
            'primary_energy_input': primary_energy_input,
            'exergy_consumption_supply': exergy_consumption_supply,
            'exergy_consumption_total': exergy_consumption_supply,
            'co2_emissions': co2_emissions,
            'stage_breakdown': {
                'extraction': {
                    'primary_energy': primary_energy_input,
                    'output': ng_mass_at_extraction * ng_hhv / 3.6
                },
                'transport': {
                    'input': ng_mass_at_extraction * ng_hhv / 3.6,
                    'output': ng_mass_at_extraction * ng_hhv / 3.6 * 
                             self.ng_supply_params.transport_efficiency
                },
                'distribution': {
                    'input': ng_mass_at_extraction * ng_hhv / 3.6 * 
                            self.ng_supply_params.transport_efficiency,
                    'output': building_gas_demand
                }
            }
        }
    
    def compare_systems(
        self,
        eb_results: Dict[str, float],
        gb_results: Dict[str, float],
        hpb_results: Dict[str, float],
        generation_mix: Dict[str, float],
        tech_co2_factors: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare well-to-tap results for all three systems.
        
        Parameters:
        -----------
        eb_results : dict
            ElectricBoiler results from system_update()
        gb_results : dict
            GasBoiler results from system_update()
        hpb_results : dict
            HeatPumpBoiler results from system_update()
        generation_mix : dict
            Generation mix fractions
        tech_co2_factors : dict
            CO2 emission factors by technology
        
        Returns:
        --------
        dict
            Dictionary with well-to-tap analysis for each system
        """
        # Analyze electric systems
        eb_electricity = eb_results.get('E_heater', 0.0) / 1000.0  # Convert W to kW
        # Support both E_fan and E_fan_ou for compatibility
        hpb_fan_power = hpb_results.get('E_fan_ou', hpb_results.get('E_fan', 0.0))
        hpb_electricity = (
            (hpb_results.get('E_cmp', 0.0) + hpb_fan_power) / 1000.0
        )
        
        eb_wtt = self.analyze_electric_system(
            eb_electricity, generation_mix, tech_co2_factors
        )
        hpb_wtt = self.analyze_electric_system(
            hpb_electricity, generation_mix, tech_co2_factors
        )
        
        # Analyze gas system
        gb_gas = gb_results.get('E_NG', 0.0) / 1000.0  # Convert W to kW
        gb_wtt = self.analyze_gas_system(gb_gas)
        
        return {
            'EB': eb_wtt,
            'GB': gb_wtt,
            'HPB': hpb_wtt
        }



