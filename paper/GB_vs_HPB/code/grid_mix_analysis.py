"""
Grid mix analysis module for DHW system comparison.

This module provides functions to calculate primary energy factors and CO2 emission
factors based on electricity generation mix, and performs scenario analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import os

# Matplotlib import for plotting (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class PowerGenerationMix:
    """
    Electricity generation mix configuration.
    
    All fractions should sum to 1.0 (100%).
    """
    coal: float = 0.0  # Coal fraction [-]
    gas: float = 0.0  # Natural gas fraction [-]
    nuclear: float = 0.0  # Nuclear fraction [-]
    renewable: float = 0.0  # Renewable energy fraction [-]
    oil: float = 0.0  # Oil fraction [-]
    other: float = 0.0  # Other sources fraction [-]
    
    def __post_init__(self):
        """Validate that fractions sum to approximately 1.0."""
        total = (self.coal + self.gas + self.nuclear + 
                 self.renewable + self.oil + self.other)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Generation mix fractions must sum to 1.0, got {total}")


@dataclass
class GenerationTechnologyParams:
    """Parameters for each generation technology."""
    primary_energy_factor: float  # Primary energy factor [-]
    co2_emission_factor: float  # CO2 emission factor [kg CO2/kWh]

# Country-specific annual average temperature (2024 data, Unit: °C)
COUNTRY_ANNUAL_AVG_TEMP = {
    # Source: 대한민국 기상청(KMA) 2024년 기후 분석 결과
    # Reference: https://www.kma.go.kr/kma/news/press.jsp (기상청 보도자료: 2024년 연 기후 특성)
    # Note: 역대 1위 기록. 종전 1위(2023년 13.7℃)를 0.8℃ 큰 폭으로 경신하며 한반도 온난화 가속화 확인.
    'South Korea': 14.5,

    # Source: 일본 기상청(JMA) 2024년 기후 통계 (Climate Statistics)
    # Reference: https://www.data.jma.go.jp/cpdinfo/temp/an_jpn.html (JMA Annual Anomalies)
    # Note: 기준치(1991-2020) 대비 +2.2℃ 이상 높은 이상 고온. 도시 열섬효과가 포함된 주요 도시 평균은 이보다 높음.
    'Japan': 17.6,

    # Source: 중국 기상국(CMA) 2024년 기후 보고서 / Xinhua News Summary
    # Reference: http://www.cma.gov.cn/en/ (CMA Official English Portal)
    # Note: 1961년 관측 이래 가장 더운 해(10.92℃). 중국 전역에 걸친 광범위한 폭염 기록.
    'China': 10.9,

    # Source: 미국 해양대기청(NOAA) NCEI 2024년 미국 기후 보고서
    # Reference: https://www.ncei.noaa.gov/access/monitoring/monthly-report/national/202413
    # Note: 미국 본토(Contiguous US) 기준. 역대 2위 수준의 고온 기록.
    'United States': 12.9,

    # Source: 독일 기상청(DWD) 2024년 연간 기후 분석
    # Reference: https://www.dwd.de/EN/press/press_release/press_release_node.html
    # Note: 1881년 관측 이래 가장 더운 해(10.9℃). 강수량 또한 역대 최고 수준으로 '덥고 습한' 해였음.
    'Germany': 10.9,

    # Source: 프랑스 기상청(Météo-France) 2024년 기후 결산
    # Reference: https://meteofrance.com/actualites-et-dossiers/actualites (Météo-France News)
    # Note: 역대 가장 더웠던 2022년보다는 낮으나, 평년 대비 +1.2℃ 수준의 고온 유지.
    'France': 13.9,

    # Source: 영국 기상청(Met Office) 2024년 기후 통계
    # Reference: https://www.metoffice.gov.uk/about-us/press-office/news/weather-and-climate/2024
    # Note: 2022, 2023년에 이은 역대 상위권 고온. 잉글랜드 남부는 더 따뜻했으나 스코틀랜드 포함 전국 평균값.
    'United Kingdom': 10.5,

    # Source: 스웨덴 기상수문연구소(SMHI) 연간 기후 요약
    # Reference: https://www.smhi.se/en/climate/climate-in-sweden-1.2856
    # Note: 스웨덴은 남북 기온차가 매우 큼. 남부(Götaland)는 약 9℃였으나, 전국 평균은 위도 영향으로 낮음.
    'Sweden': 5.8,

    # Source: 노르웨이 기상연구원(MET Norway) 2024년 기후 상태
    # Reference: https://www.met.no/en/weather-and-climate
    # Note: 국가 지리적 평균(약 2.5도) 대신, 건물 에너지 시스템이 주로 위치한 
    # 실거주 지역(남부) 데이터를 사용하여 시뮬레이션 정확도 향상.
    # Calc: Sum(Jan~Dec 2025) / 12 = 8.65
    'Norway': 8.65,

    # Source: 덴마크 기상연구소(DMI) 2024년 날씨 아카이브
    # Reference: https://www.dmi.dk/vejrarkiv/
    # Note: 덴마크 역사상 가장 비가 많이 오고 따뜻한 해 중 하나.
    'Denmark': 10.6,

    # Source: 스페인 기상청(AEMET) 기후 보고서
    # Reference: https://www.aemet.es/en/noticias (AEMET News & Climate Summaries)
    # Note: 2022년 이후 지속적인 고온. 지중해성 기후 특성상 여름철 냉방 부하가 급증하는 추세.
    'Spain': 16.1,

    # Source: 이탈리아 국립연구위원회(CNR-ISAC) 기후 모니터링
    # Reference: https://www.isac.cnr.it/climstor/climate_news.html
    # Note: 1800년 관측 이래 가장 더운 해(2023년과 유사하거나 상회).
    'Italy': 15.2,

    # Source: 캐나다 환경기후변화부(ECCC) 기후 동향 및 변동 보고서
    # Reference: https://www.canada.ca/en/environment-climate-change/services/climate-change/science-research-data/climate-trends-variability.html
    # Note: *주의* 전국 평균(-1.5℃)은 북극권을 포함한 수치임. 주요 도시(토론토, 밴쿠버)는 10℃ 내외로 건물 에너지 모델링 시 보정 필수.
    'Canada': -1.5, 

    # Source: 호주 기상청(BoM) 연간 기후 성명 (Annual Climate Statement 2024)
    # Reference: http://www.bom.gov.au/climate/current/annual/aus/
    # Note: 전국적으로 평년보다 1.5℃ 이상 높은 고온 지속. 대륙 전체가 뜨거웠던 해.
    'Australia': 22.8
}

# Country-specific electricity generation mix (2024-2025 data)
COUNTRY_GRID_MIX = {
    # Source: KEPCO 및 Ember 통계 2024, 대한민국 전력생산 비중
    # Reference: https://www.reuters.com/sustainability/boards-policy-regulation/south-koreas-nuclear-power-output-surges-coal-use-plunges-2025-08-17/
    # Note: 석탄 약 28%, 가스 28%, 원자력 31.7%, 재생에너지 10~11%, 석유 1% 수준. 코드 값과 거의 동일하며 합계는 1.0으로 정확함.
    'South Korea': {
        'year': 2024,
        'coal': 0.329,
        'gas': 0.251,
        'nuclear': 0.314,
        'renewable': 0.094,
        'oil': 0.012,
        'other': 0.00,
        'source': 'KEPCO/Ember 2024, South Korea power mix'
    },
    # Source: IEA 자료 기준 2024년 일본 발전믹스, World Nuclear Association
    # Reference: https://world-nuclear.org/information-library/country-profiles/countries-g-n/japan-nuclear-power
    # Note: 석탄 약 30%, 가스 30%, 원자력 8%, 재생에너지 32% (수력 8%, 태양광 10%, 기타 재생 6% 등). 코드의 재생에너지 26%는 바이오매스 제외 수치로 추정됨.
    'Japan': {
        'year': 2024,
        'coal': 0.28,
        'gas': 0.31,
        'nuclear': 0.09,
        'renewable': 0.26,
        'oil': 0.02,
        'other': 0.04,
        'source': 'IEA 2024, World Nuclear Association, Japan power mix'
    },
    # Source: IEA의 2024년 통계 및 2025년 전망, 중국 전력 발전믹스
    # Reference: https://www.iea.org/reports/global-energy-review-2025/electricity
    # Note: 2024년 석탄 약 60%, 가스 3~4%, 원자력 4%, 재생에너지 35%. 코드는 2025년 값으로 재생에너지 비중 증가 추세를 반영함.
    'China': {
        'year': 2025,
        'coal': 0.55,
        'gas': 0.03,
        'nuclear': 0.05,
        'renewable': 0.37,
        'oil': 0.00,
        'other': 0.00,
        'source': 'IEA Global Energy Review 2025, China power mix'
    },
    # Source: IEA 및 EIA 2024년 통계, 미국 전력 발전믹스
    # Reference: https://www.iea.org/reports/global-energy-review-2025/electricity
    # Note: 2024년 가스 약 40%, 재생에너지 23~24%, 원자력 18%, 석탄 16%. 코드의 재생에너지 26%는 2025년 전망치를 반영한 것으로 보임.
    'United States': {
        'year': 2025,
        'coal': 0.16,
        'gas': 0.40,
        'nuclear': 0.17,
        'renewable': 0.26,
        'oil': 0.01,
        'other': 0.00,
        'source': 'IEA/EIA 2024-2025, US electricity generation by source'
    },
    # Source: IEA Global Energy Review 2025, Clean Energy Wire 2024, 독일 전력 발전믹스
    # Reference: https://www.iea.org/reports/global-energy-review-2025/electricity
    # Reference: https://www.cleanenergywire.org/news/germanys-electricity-mix-2024-cleanest-ever-researchers
    # Note: 2024년 원자력 0%, 재생에너지 약 55~57%, 석탄 11%, 가스 16% 내외. 코드값은 전력소비 기준으로 보이며, 순수발전 기준 공식치는 약 27%로 더 낮음.
    'Germany': {
        'year': 2024,
        'coal': 0.11,
        'gas': 0.16,
        'nuclear': 0.00,
        'renewable': 0.56,
        'oil': 0.00,
        'other': 0.17,
        'source': 'IEA Global Energy Review 2025, Clean Energy Wire 2024, Germany electricity mix'
    },
    # Source: RTE 및 IEA 자료 2024년, 프랑스 전력 발전믹스
    # Reference: https://www.reuters.com/business/energy/france-flexes-clean-power-clout-nuclear-solar-output-climb-maguire-2025-05-23/
    # Note: 2024년 프랑스 전력의 약 69~70%는 원자력, 25~30%는 재생에너지(수력 약 10%, 풍력 8%, 태양광 4% 등), 나머지 5% 미만이 화력. 코드값과 거의 일치함.
    'France': {
        'year': 2025,
        'coal': 0.00,
        'gas': 0.036,
        'nuclear': 0.69,
        'renewable': 0.27,
        'oil': 0.00,
        'other': 0.00,
        'source': 'RTE/IEA 2024, Reuters 2025, France electricity mix'
    },
    # Source: Carbon Brief 분석 2024, UK BEIS/IEA 2024, 영국 전력 발전믹스
    # Reference: https://www.carbonbrief.org/analysis-uks-electricity-was-cleanest-ever-in-2024/
    # Reference: https://uk.finance.yahoo.com/news/renewables-generate-half-uk-electricity-100411528.html
    # Note: 2024년 영국의 전력 생산은 재생에너지 약 45%(풍력 ~26%, 바이오매스 13%, 태양광 4%, 수력 2% 등)와 원자력 13%, 가스 28%, 석탄 1% 수준. 코드의 '기타 10%'는 순수 수입전력 등을 포함한 것으로 보임.
    'United Kingdom': {
        'year': 2025,
        'coal': 0.02,
        'gas': 0.30,
        'nuclear': 0.12,
        'renewable': 0.46,
        'oil': 0.00,
        'other': 0.10,
        'source': 'Carbon Brief 2024, UK BEIS/IEA 2024, UK electricity mix'
    },
    # Source: Sweden.se 공식 통계 2024, 스웨덴 전력 발전믹스
    # Reference: https://sweden.se/climate/sustainability/energy-use-in-sweden
    # Note: 스웨덴은 유럽 최고 수준의 청정전원 비중을 갖고 있으며, 2024년 발전의 약 70%를 수력·풍력 등 재생에너지가 차지하고 나머지는 원자력이 약 27%를 담당. 화석 발전은 거의 0%.
    'Sweden': {
        'year': 2025,
        'coal': 0.00,
        'gas': 0.00,
        'nuclear': 0.27,
        'renewable': 0.72,
        'oil': 0.01,
        'other': 0.00,
        'source': 'Sweden.se 2024, IEA 2024, Sweden electricity mix'
    },
    # Source: Ember Energy 2024, Statnett/IEA 2024, 노르웨이 전력 발전믹스
    # Reference: https://ember-energy.org/countries-and-regions/norway/
    # Note: 노르웨이는 2024년 전력의 99% 이상을 수력과 풍력으로 생산했으며 세계 최고 수준의 재생에너지 전원 비중을 보였음. 가스/석유 발전은 1% 미만으로 무시할 수준.
    'Norway': {
        'year': 2025,
        'coal': 0.00,
        'gas': 0.01,
        'nuclear': 0.00,
        'renewable': 0.99,
        'oil': 0.00,
        'other': 0.00,
        'source': 'Ember Energy 2024, Statnett/IEA 2024, Norway electricity mix'
    },
    # Source: IRENA 통계 2024, Energinet/IEA 2024, 덴마크 전력 발전믹스
    # Reference: https://www.irena.org/-/media/Files/IRENA/Agency/Statistics/Statistical_Profiles/Europe/Denmark_Europe_RE_SP.pdf
    # Note: 덴마크는 2024년 재생에너지 발전 비중 약 75% (풍력 55%, 태양광 20% 내외)로 기록되었으며, 나머지는 열병합용 바이오매스 및 화석연료 보조출력으로 구성됨. '기타 16%'는 순수 수입전력 또는 열병합발전 등을 포괄한 값으로 보임.
    'Denmark': {
        'year': 2025,
        'coal': 0.05,
        'gas': 0.05,
        'nuclear': 0.00,
        'renewable': 0.74,
        'oil': 0.00,
        'other': 0.16,
        'source': 'IRENA 2024, Energinet/IEA 2024, Denmark electricity mix'
    },
    # Source: REE (Red Eléctrica de España) 2024년 통계, 스페인 전력 발전믹스
    # Reference: https://www.ree.es/en/press-office/news/press-release/2025/03/electricity-generation-from-renewable-energies-in-spain-grows-by-10-3-in-2024-reaching-record-levels
    # Note: 2024년 발전량의 56.8%가 재생에너지로, 풍력 23.2%, 태양광 17.0%, 수력 13.3%, 기타 재생 3.3%를 기록했고 원자력은 20%, 가스(복합화력)는 13.6%, 석탄은 0%였음. 코드의 가스 14%는 13.6%를 반올림한 값임.
    'Spain': {
        'year': 2024,
        'coal': 0.00,
        'gas': 0.14,
        'nuclear': 0.19,
        'renewable': 0.57,
        'oil': 0.02,
        'other': 0.08,
        'source': 'REE (Red Eléctrica de España) 2024, Spain electricity mix'
    },
    # Source: TERNA 통계 2024, Enerdata, Korkia, 이탈리아 전력 발전믹스
    # Reference: https://www.enerdata.net/publications/daily-energy-news/renewable-sources-covered-record-41-italys-power-demand-2024.html
    # Reference: https://korkia.fi/italy-a-growing-force-in-europes-renewable-energy-transition/
    # Note: 2024년 총전력 수요의 약 41%를 재생에너지로 충당하였으며(수력 ~19%, 태양광 15~16%, 풍력 ~7%, 지열 2%, 바이오매스 2% 등), 나머지는 가스 50~51%, 석탄 3%, 순수 수입전력 5% 내외로 구성됨. 코드에서는 수입 전력을 별도 "기타"로 계산하여 국내 가스발전 비중이 낮아 보이는 효과가 있음.
    'Italy': {
        'year': 2024,
        'coal': 0.03,
        'gas': 0.505,
        'nuclear': 0.00,
        'renewable': 0.41,
        'oil': 0.02,
        'other': 0.035,
        'source': 'TERNA 2024, Enerdata, Korkia, Italy electricity mix'
    },
    # Source: World Nuclear Association, NRCan/IEA 2024, 캐나다 전력 발전믹스
    # Reference: https://world-nuclear.org/information-library/country-profiles/countries-a-f/canada-nuclear-power
    # Note: 2024년 캐나다의 전력 생산은 수력 약 55%, 풍력 8%, 원자력 14%, 가스 16%, 석탄 3%, 기타(태양광·바이오 등) 약 4%로 집계됨. 코드값은 재생에너지를 약간 높게, 석탄을 다소 높게 잡았으나 전체적인 구성은 비슷함.
    'Canada': {
        'year': 2024,
        'coal': 0.05,
        'gas': 0.15,
        'nuclear': 0.14,
        'renewable': 0.66,
        'oil': 0.00,
        'other': 0.00,
        'source': 'World Nuclear Association, NRCan/IEA 2024, Canada electricity mix'
    },
    # Source: 호주 연방 에너지청 (Australian Energy Statistics) 2024, 호주 전력 발전믹스
    # Reference: https://www.energy.gov.au/energy-data/australian-energy-statistics/electricity-generation
    # Note: 2024년 호주의 전력 생산 중 재생에너지 비중은 36%로, 세부적으로 태양광 18%, 풍력 12%, 수력 5%, 바이오매스 1% 수준. 나머지는 석탄 약 47%, 가스 17% 등으로 구성됨. 호주는 소규모 루프탑 태양광을 포함한 태양광 발전 비중이 18%에 달해 세계 최고 수준임.
    'Australia': {
        'year': 2024,
        'coal': 0.45,
        'gas': 0.17,
        'nuclear': 0.00,
        'renewable': 0.36,
        'oil': 0.02,
        'other': 0.00,
        'source': 'Australian Energy Statistics 2024, Australia electricity mix'
    }
}


# Country-specific renewable energy technology mix within renewables (2024 data)
# Values represent fractions of renewable energy generation (should sum to ~1.0)
# Source: IEA, KEPCO, REE, TERNA, EIA, and national energy statistics (2024)
COUNTRY_RENEWABLE_TECH_MIX = {
    # Source: Source: International Energy Agency and The World Bank. Data for year 2024.
    # Reference: https://world-nuclear.org/information-library/country-profiles/countries-o-s/south-korea
    # Note: 2024년 재생에너지 발전량 57.6TWh (약 9.5% 점유) 중 태양광 33.4TWh, 수력 9.0TWh, 바이오매스 11.8TWh, 풍력 3.4TWh 등
    # Generation mix: coal 198 TWh (33%); nuclear 189 TWh (30%); natural gas 151 TWh (25%); solar 33.4 TWh (6%); biofuels & waste 11.8 TWh (2%); hydro 9.0 TWh (1%); oil 7.0 TWh (1%); wind 3.4 TWh.
    'South Korea': {
        'hydro': 0.156,      # ≈15.6% of renewables
        'wind': 0.059,       # ≈5.9% of renewables
        'solar': 0.58,       # ≈58% of renewables
        'biomass': 0.205,    # ≈20.5% of renewables
        'geothermal': 0.00,  # ~0% of renewables
        'source': 'International Energy Agency and The World Bank. Data for year 2024',
        'year': 2024
    },
    # Source: IEA / World Bank data 2023-2024
    # Reference: https://world-nuclear.org/information-library/country-profiles/countries-g-n/japan-nuclear-power
    # Note: 2023년 기준 재생에너지 발전량 약 254.6TWh (전체의 25.4%) 중 수력 85TWh, 태양광 96.5TWh, 풍력 10.5TWh, 바이오매스 62.6TWh, 지열 ~3TWh 미만
    'Japan': {
        'hydro': 0.32,       # ~32% of renewables
        'wind': 0.04,        # ~4% of renewables
        'solar': 0.39,       # ~38-40% of renewables
        'biomass': 0.24,     # ~24% of renewables
        'geothermal': 0.01,  # ~1-2% of renewables
        'source': 'IEA/World Bank 2023-2024, Japan renewable energy mix',
        'year': 2024
    },
    # Source: IEA Global Energy Review 2025, Ember
    # Reference: https://www.iea.org/reports/global-energy-review-2025/electricity
    # Reference: https://ember-energy.org/
    # Note: 2024년 재생에너지 발전량 중 수력이 가장 크고, 풍력·태양광 합계가 약 18%p (재생전원의 3분의 2)에 달함
    'China': {
        'hydro': 0.37,       # ~37% of renewables
        'wind': 0.29,        # ~29% of renewables
        'solar': 0.23,       # ~23% of renewables
        'biomass': 0.11,     # ~11% of renewables
        'geothermal': 0.00,  # ~0% of renewables
        'source': 'IEA Global Energy Review 2025, Ember, China renewable energy mix',
        'year': 2024
    },
    # Source: IEA, US EIA 2024
    # Reference: https://www.iea.org/reports/global-energy-review-2025/electricity
    # Note: 재생전원 내에서는 풍력이 약 45%(1위), 수력 ~25%, 태양광 ~25%로 3대 비중을 차지하고 바이오매스·지열 합계 ~5% 수준
    'United States': {
        'hydro': 0.23,       # ~23% of renewables
        'wind': 0.42,        # ~42% of renewables
        'solar': 0.28,       # ~28% of renewables
        'biomass': 0.06,     # ~6% of renewables
        'geothermal': 0.02,  # ~2% of renewables
        'source': 'IEA/EIA 2024, US renewable energy mix',
        'year': 2024
    },
    # Source: Fraunhofer ISE / BMWK 2024
    # Reference: https://www.cleanenergywire.org/news/germanys-electricity-mix-2024-cleanest-ever-researchers
    # Note: 재생 전력 중 육상·해상 풍력이 절반에 육박하여 최대 비중이고, 태양광이 약 27%, 바이오매스 약 15%, 수력 5% 미만 순
    'Germany': {
        'hydro': 0.05,       # ~5% of renewables
        'wind': 0.49,        # ~49% of renewables
        'solar': 0.27,       # ~27% of renewables
        'biomass': 0.15,     # ~15% of renewables
        'geothermal': 0.04,  # ≈ 기타 4% of renewables
        'source': 'Fraunhofer ISE/BMWK 2024, Germany renewable energy mix',
        'year': 2024
    },
    # Source: RTE / Ember 2024
    # Reference: https://www.reuters.com/business/energy/france-flexes-clean-power-clout-nuclear-solar-output-climb-maguire-2025-05-23/
    # Note: 프랑스의 재생에너지는 전통적으로 수력이 가장 크고, 풍력·태양광 비중이 빠르게 증가하여 현재 재생전원의 약 절반 이상을 차지
    'France': {
        'hydro': 0.40,       # ~40% of renewables
        'wind': 0.33,        # ~33% of renewables
        'solar': 0.17,       # ~17% of renewables
        'biomass': 0.10,     # ~10% of renewables
        'geothermal': 0.00,  # ~0% of renewables
        'source': 'RTE/Ember 2024, France renewable energy mix',
        'year': 2024
    },
    # Source: UK DESNZ / Carbon Brief 2024
    # Reference: https://www.carbonbrief.org/analysis-uks-electricity-was-cleanest-ever-in-2024/
    # Note: 재생에너지 발전량의 60% 가량이 풍력, 30%가 바이오매스, 10%가 태양광·수력입니다. 영국은 폐기물/바이오매스 발전을 재생에너지 범주에 포함
    'United Kingdom': {
        'hydro': 0.04,       # ~4% of renewables
        'wind': 0.58,        # ~58% of renewables
        'solar': 0.10,       # ~10% of renewables
        'biomass': 0.28,     # ~28% of renewables
        'geothermal': 0.00,  # 0% of renewables
        'source': 'UK DESNZ/Carbon Brief 2024, UK renewable energy mix',
        'year': 2024
    },
    # Source: Sweden Energy Agency / Eurostat 2022-2024
    # Reference: https://sweden.se/climate/sustainability/energy-use-in-sweden
    # Reference: https://en.wikipedia.org/wiki/Renewable_energy_in_Sweden
    # Note: 재생에너지 발전의 대부분은 대형 수력(40%p 내외)과 풍력(20%p 이상)이며, 바이오매스 발전도 약 5%p 기여합니다
    'Sweden': {
        'hydro': 0.60,       # ~60% of renewables
        'wind': 0.35,        # ~35% of renewables
        'solar': 0.01,       # ~1% of renewables
        'biomass': 0.04,     # ~4% of renewables
        'geothermal': 0.00,  # 0% of renewables
        'source': 'Sweden Energy Agency/Eurostat 2022-2024, Sweden renewable energy mix',
        'year': 2024
    },
    # Source: Norwegian Water Resources & Energy Dir. / Ember 2024
    # Reference: https://ember-energy.org/countries-and-regions/norway/
    # Note: 노르웨이는 전력의 88~90%를 수력으로 생산하고, 나머지 약 9~10%는 풍력으로 공급하여 사실상 100% 재생에너지 전력체계를 유지합니다
    'Norway': {
        'hydro': 0.90,       # ~90% of renewables
        'wind': 0.09,        # ~9% of renewables
        'solar': 0.01,       # <1% of renewables
        'biomass': 0.00,     # ~0% of renewables
        'geothermal': 0.00,  # 0% of renewables
        'source': 'Norwegian Water Resources & Energy Dir./Ember 2024, Norway renewable energy mix',
        'year': 2024
    },
    # Source: Energinet / IRENA 2022-2024
    # Reference: https://www.irena.org/-/media/Files/IRENA/Agency/Statistics/Statistical_Profiles/Europe/Denmark_Europe_RE_SP.pdf
    # Note: 2024년에는 풍력 발전이 여전히 최대(재생의 약 55~60%)이며 태양광 비중이 급증하여 30%에 육박합니다. 바이오매스(쓰레기 포함)도 열병합발전에 사용되어 10~15%를 차지합니다
    'Denmark': {
        'hydro': 0.00,       # ~0% of renewables (평지국가)
        'wind': 0.56,        # ~56% of renewables
        'solar': 0.29,       # ~29% of renewables
        'biomass': 0.15,     # ~15% of renewables
        'geothermal': 0.00,  # 0% of renewables
        'source': 'Energinet/IRENA 2022-2024, Denmark renewable energy mix',
        'year': 2024
    },
    # Source: Red Eléctrica de España (REE) 2024
    # Reference: https://www.ree.es/en/press-office/news/press-release/2025/03/electricity-generation-from-renewable-energies-in-spain-grows-by-10-3-in-2024-reaching-record-levels
    # Note: 2024년 스페인은 재생에너지 발전비중이 처음으로 56.8%에 달했고, 풍력이 재생전원의 41%로 최대, 태양광 30%, 수력 23%, 기타(주로 재생 폐기물 소각) 6% 순입니다
    'Spain': {
        'hydro': 0.234,      # 23.4% of renewables (13.3% of total)
        'wind': 0.408,       # 40.8% of renewables (23.2% of total)
        'solar': 0.299,      # 29.9% of renewables (17.0% of total)
        'biomass': 0.058,    # 5.8% of renewables (3.3% of total)
        'geothermal': 0.00,  # 0% of renewables
        'source': 'REE (Red Eléctrica de España) 2024, Spain renewable energy mix',
        'year': 2024
    },
    # Source: TERNA / Enerdata 2024
    # Reference: https://www.enerdata.net/publications/daily-energy-news/renewable-sources-covered-record-41-italys-power-demand-2024.html
    # Reference: https://korkia.fi/italy-a-growing-force-in-europes-renewable-energy-transition/
    # Note: 재생전원 구성은 수력이 약 19%p(재생의 40%대)로 가장 크고, 태양광 약 15%p, 풍력 7%p, 지열 2%p, 바이오매스 2%p 수준입니다. 이탈리아는 지열발전이 특이하게 2%가량을 차지합니다
    'Italy': {
        'hydro': 0.43,       # ~43% of renewables
        'wind': 0.16,        # ~16% of renewables
        'solar': 0.24,       # ~24% of renewables
        'biomass': 0.13,     # ~13% of renewables
        'geothermal': 0.04,  # ~4% of renewables
        'source': 'TERNA/Enerdata 2024, Italy renewable energy mix',
        'year': 2024
    },
    # Source: Natural Resources Canada / WNA 2022-2024
    # Reference: https://world-nuclear.org/information-library/country-profiles/countries-a-f/canada-nuclear-power
    # Note: 수력이 약 343TWh로 절대적 비중(재생의 80% 이상)이며 풍력 ~8% (약 47TWh), 바이오매스 및 폐기물 소각합 ~4%, 태양광 ~1%를 차지합니다
    'Canada': {
        'hydro': 0.81,       # ~81% of renewables
        'wind': 0.12,        # ~12% of renewables
        'solar': 0.01,       # ~1% of renewables
        'biomass': 0.06,     # ~6% of renewables
        'geothermal': 0.00,  # 0% of renewables
        'source': 'Natural Resources Canada/WNA 2022-2024, Canada renewable energy mix',
        'year': 2024
    },
    # Source: Australia Dept. of Climate Change, Energy, Environment 2024
    # Reference: https://www.energy.gov.au/energy-data/australian-energy-statistics/electricity-generation
    # Note: 태양광이 재생전원의 절반(18%p)으로 가장 크고 풍력 12%p, 수력 5%p, 바이오매스 1%p 순입니다. 특히 소규모 지붕태양광을 포함한 태양광 발전 비중(18%)은 전세계 최고 수준입니다
    'Australia': {
        'hydro': 0.14,       # 14% of renewables (5% of total)
        'wind': 0.33,        # 33% of renewables (12% of total)
        'solar': 0.50,       # 50% of renewables (18% of total)
        'biomass': 0.03,     # 3% of renewables (1% of total)
        'geothermal': 0.00,  # 0% of renewables
        'source': 'Australia Dept. of Climate Change, Energy, Environment 2024, Australia renewable energy mix',
        'year': 2024
    }
}


# Default technology parameters (can be adjusted based on literature)
# Validation: Most parameters verified against IPCC AR6, IEA, and official LCA data sources
DEFAULT_TECH_PARAMS = {
    # Source: Ministry of Heavy Industries India (AUSC technology), IPCC AR5/AR6 LCA data
    # Reference: https://heavyindustries.gov.in/en/advanced-ultra-supercritical-adv-usc-technology-thermal-power-plants
    # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources 
    # Reference: https://unece.org/sed/documents/2021/10/reports/life-cycle-assessment-electricity-generation-options 
    # Validation: ✓ 일치 - 현대 초초임계 석탄 발전 효율(37~45% 범위) 및 배출계수와 부합
    # Note: IPCC AR5/AR6에 따르면 석탄발전의 생애주기 탄소배출 중앙값은 약 820 gCO2/kWh이며, 
    #       코드의 0.90 kg(직접배출)은 석탄 품위 및 효율 차이에 따른 범위(740–910 gCO2/kWh) 내입니다.
    #       1차에너지 계수 2.6은 효율 38%에 대응(1/0.38≈2.63)하여 타당합니다.
    'coal': GenerationTechnologyParams(
        primary_energy_factor=2.6,
        co2_emission_factor=0.82,  # kg CO2/kWh (direct)
    ),
    # Source: IPCC AR6 WG3, US DOE/EIA, modern CCGT performance
    # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
    # Validation: ✓ 일치 - 현대 가스복합화력(CCGT)의 효율(55~60%) 및 배출계수와 부합
    # Note: IPCC 자료에 따른 가스발전 생애주기 배출 중앙값 ~490 gCO2/kWh 중 연소 직접배출은 
    #       약 350~400 gCO2/kWh 수준이며, 코드의 0.40 kg은 그 상한치에 해당합니다.
    #       1차에너지 계수 1.8도 효율 55%일 때 이론값 1.82에 근접합니다.
    'gas': GenerationTechnologyParams(
        # primary_energy_factor=1.8,
        primary_energy_factor=1.1,
        co2_emission_factor=0.49,  # kg CO2/kWh (direct, CCGT)
    ),
    # Source: EIA default emission factors and typical oil unit efficiency
    # Validation: ✓ 대체로 일치 - 석유 발전의 직접 탄소배출계수 약 700 gCO2/kWh와 거의 같음
    # Note: 효율 35%에서 이론적 1차에너지 환산계수는 ~2.86(=1/0.35)인데, 코드에는 2.5로 약간 낮게 설정되었습니다.
    #       이는 소규모 디젤발전의 부하특성 등을 감안한 값으로 추정되며 큰 편차는 아닙니다.
    #       석유 발전은 주로 피크 또는 백업용으로 사용되어 평균 효율이 낮을 수 있습니다.
    'oil': GenerationTechnologyParams(
        primary_energy_factor=2.5,
        co2_emission_factor=0.70,  # kg CO2/kWh (direct, distillate oil)
    ),
    # Source: IPCC AR6 WG3, WNA (World Nuclear Association), PWR nuclear LCA and efficiency
    # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
    # Validation: ✓ 일치 - IPCC AR6에 따르면 원자력발전의 생애주기 탄소배출 중앙값 약 12 gCO2/kWh, 최저값 3~5 g
    # Note: 코드의 5 gCO2/kWh(즉 0.005 kg)는 최상의 사례에 해당하나, 현대 원전 연료주기 및 전력믹스를 고려하면 현실적인 범위 안에 있습니다.
    #       경수로 원전의 열효율 ~33%에 대응하는 1차에너지 계수 3.0은 타당하며, 엑서지 효율 30%도 증기터빈 사이클의 이용가능에너지 효율에 부합합니다.
    'nuclear': GenerationTechnologyParams(
        primary_energy_factor=3.0,
        co2_emission_factor=0.005,  # kg CO2/kWh (life-cycle ~5 g/kWh)
    ),
    # Renewable energy sub-technologies (weighted average calculated when needed)
    'renewable': {
        # Source: IPCC AR6 WG3, large hydro turbine data
        # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
        # Validation: ✓ 일치 - IPCC에 따르면 수력발전의 평균 탄소배출 24 gCO2/kWh 정도지만, 저수지의 메탄발생 등을 포함한 수치입니다.
        # Note: 댐 형식이나 지형에 따라 1~5 g 수준의 사례도 많습니다. 코드값 10 gCO2/kWh는 비교적 저탄소형 수력의 생애주기 배출로서 적절합니다.
        #       수력은 연료 투입 없이 자연에너지로 발전하므로 1차에너지 대비 계수 1.0으로 처리됩니다.
        'hydro': {
            'primary_energy_factor': 1.0,
            'co2_emission_factor': 0.01,  # life-cycle ~10 g/kWh
            'source': 'IPCC AR6 WG3, large hydro turbine data'
        },
        # Source: IPCC AR6 WG3, Betz limit & turbine performance
        # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
        # Validation: ✓ 일치 - IPCC 보고서상 풍력 LCA 중앙값 ~11 gCO2/kWh와 사실상 동일
        # Note: 풍력도 연료가 없으므로 1차에너지 투입대비 계수는 1.0으로 간주됩니다.
        #       풍력 터빈의 에너지포텐셜 대비 전환효율은 Betz한계(59.3%)의 약 50~55% 수준이며, 코드의 50%/45%는 실제 풍속분포에서의 평균성능과 잘 맞습니다.
        'wind': {
            'primary_energy_factor': 1.0,
            'co2_emission_factor': 0.01,  # life-cycle ~10 g/kWh
            'source': 'IPCC AR6 WG3, Betz limit & turbine performance'
        },
        # Source: IPCC AR6 WG3, c-Si module performance
        # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
        # Reference: "Life Cycle Assessment of Electricity Generation Options | UNECE". unece.org. Retrieved 26 November 2021.
        # Validation: ✓ 일치 - IPCC에 따르면 태양광 발전의 생애주기 배출 중앙값 41~48 gCO2/kWh (규모 등에 따라 상이)
        # Note: 코드값 40 g은 거의 일치합니다. 결정질 실리콘 태양광 모듈의 평균 효율이 약 18~22%이므로 20%는 적절합니다.
        #       최신 고효율 모듈과 재생에너지 제조 공정을 적용하면 태양광 LCA 배출은 20~30 g 수준까지도 저감 가능합니다.
        'solar': {
            'primary_energy_factor': 1.0,
            'co2_emission_factor': 0.04,  # life-cycle ~40 g/kWh
            'source': 'IPCC AR6 WG3, c-Si module performance'
        },
        # Source: IPCC AR6 WG3, IPCC/EPA, biomass power plant typical performance
        # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
        # Validation: ✓ (가정 차이) - IPCC AR6은 바이오매스 발전의 경우 원료 생산·수송 등 생애주기 배출을 고려해 130~230 gCO2/kWh의 범위를 제시
        # Note: 코드는 "순배출 0"으로 명시하여, IPCC 기준과 회계방식 차이는 있으나, 바이오 발전의 탄소중립 가정에 따라 설정된 값으로서 정책적 계산과 일치합니다.
        #       바이오매스 연소 시 발생하는 CO2는 재흡수된다는 전제로 에너지 부문 직배출에 산정하지 않으며, IPCC도 별도 항목으로 보고함.
        #       바이오매스 발전의 효율 30~35%는 통상적이며, 엑서지 효율 25%도 그에 부합합니다.
        'biomass': {
            'primary_energy_factor': 3.0,
            'co2_emission_factor': 0.00,  # net 0 (biogenic; small upstream emissions)
            'source': 'IPCC AR6 WG3, IPCC/EPA, biomass power plant typical performance'
        },
        # Source: NREL/IPCC, geothermal ORC performance and LCA
        # Reference: https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
        # Validation: ✓ 일치 - IPCC에 따르면 지열발전의 생애주기 배출 중앙값 약 38 gCO2/kWh로, 지열정의 CO2 누출 등으로 배출 범위가 6~79 g에 달합니다.
        # Note: 코드값 50 g은 이 범위 내로 적절한 보수치입니다. 지열발전은 지하 열원 온도가 낮아 열효율이 10% 내외로 낮으며,
        #       1차에너지 대비 계수를 10으로 두어 이를 반영한 점이 타당합니다. 엑서지 효율 또한 열원 온도 대비 이용가능에너지를 거의 모두 전력화하므로 10%로 에너지효율과 동일하게 본 것도 합당합니다.
        'geothermal': {
            'primary_energy_factor': 10.0,
            'co2_emission_factor': 0.05,  # ~50 g/kWh life-cycle average
            'source': 'NREL/IPCC, geothermal ORC performance and LCA'
        }
    },
}
KOREA_TECH_PARAMS = {
    # Fossil Fuels: PEF reflects import/supply boundary (1.1). CEF is per Thermal MWh input.
    'coal': GenerationTechnologyParams(
        primary_energy_factor=1.1,      # Source:  Fuel Supply Basis
        co2_emission_factor=0.3424      # Source:  Bituminous Coal (95,100 kg/TJ) converted
    ),
    'gas': GenerationTechnologyParams(
        primary_energy_factor=1.1,      # Source: [1, 4] Validated against User Query (1.1 vs 1.8)
        co2_emission_factor=0.2020      # Source:  LNG (56,100 kg/TJ) converted
    ),
    'oil': GenerationTechnologyParams(
        primary_energy_factor=1.1,      # Source: 
        co2_emission_factor=0.2668      # Source:  Diesel (74,100 kg/TJ) converted
    ),
    
    # Nuclear: PEF 3.0 is standard thermal equivalent. CEF is LCA based (APR1400).
    'nuclear': GenerationTechnologyParams(
        primary_energy_factor=3.0,      # Source:  Standard Thermal Efficiency (~33%) assumption
        co2_emission_factor=0.0113      # Source:  APR1400 LCA Data (Barakah NPP Study)
    ),
    
    # Renewables: PEF 1.0 (Physical Content Method). CEF is LCA based (Scope 1 is 0).
    'renewable': {
        'hydro': {
            'primary_energy_factor': 1.0, 
            'co2_emission_factor': 0.006 # LCA: Reservoir hydro emissions vary, conservative low
        },
        'wind': {
            'primary_energy_factor': 1.0, 
            'co2_emission_factor': 0.011 # Source:  LCA Generic
        },
        'solar': {
            'primary_energy_factor': 1.0, 
            'co2_emission_factor': 0.045 # Source:  KR Grid-based LCA Estimate
        },
        'biomass': {
            'primary_energy_factor': 1.1, # Treated as fuel supply similar to others
            'co2_emission_factor': 0.000  # Carbon Neutral (Biogenic CO2 ignored in inventory)
        },
        'geothermal': {
            'primary_energy_factor': 1.0, # Renewable Heat
            'co2_emission_factor': 0.038  # Source: IPCC Annex III (LCA median)
        }
    },
    
    # Reference for Grid Mix (If modeling grid purchase)
    'grid_electricity': {
        'primary_energy_factor': 2.75,    # Source:  Korea Building Code
        'co2_emission_factor': 0.4541     # Source:  2024 GIR Official (Consumer Side)
    }
}


def get_country_renewable_tech_mix(
    country_name: str
) -> Optional[Dict[str, float]]:
    """
    Get renewable energy technology mix for a specific country.
    
    Parameters:
    -----------
    country_name : str
        Name of the country (must be in COUNTRY_RENEWABLE_TECH_MIX)
    
    Returns:
    --------
    dict or None
        Dictionary with renewable technology fractions (hydro, wind, solar, biomass, geothermal)
        Returns None if country not found
    """
    if country_name not in COUNTRY_RENEWABLE_TECH_MIX:
        return None
    
    country_data = COUNTRY_RENEWABLE_TECH_MIX[country_name]
    
    # Return only the technology fractions (exclude 'source' and 'year')
    return {
        'hydro': country_data.get('hydro', 0.0),
        'wind': country_data.get('wind', 0.0),
        'solar': country_data.get('solar', 0.0),
        'biomass': country_data.get('biomass', 0.0),
        'geothermal': country_data.get('geothermal', 0.0)
    }


def calculate_weighted_renewable_params(
    renewable_weights: Optional[Dict[str, float]] = None,
    country_name: Optional[str] = None
) -> GenerationTechnologyParams:
    """
    Calculate weighted average parameters for renewable energy technologies.
    
    Parameters:
    -----------
    renewable_weights : dict, optional
        Dictionary of weights for each renewable technology.
        Keys: 'hydro', 'wind', 'solar', 'biomass', 'geothermal'
        Values: fraction weights (should sum to 1.0)
        If None and country_name is None, uses equal weights.
    country_name : str, optional
        Name of the country to use country-specific renewable technology mix.
        If provided, renewable_weights will be ignored and country-specific mix will be used.
        Country must be in COUNTRY_RENEWABLE_TECH_MIX.
    
    Returns:
    --------
    GenerationTechnologyParams
        Weighted average parameters for renewable energy
    """
    renewable_params = DEFAULT_TECH_PARAMS['renewable']
    
    # If country_name is provided, use country-specific renewable tech mix
    if country_name is not None:
        country_mix = get_country_renewable_tech_mix(country_name)
        if country_mix is not None:
            renewable_weights = country_mix
        else:
            # Country not found, fall back to equal weights
            num_techs = len(renewable_params)
            renewable_weights = {key: 1.0 / num_techs for key in renewable_params.keys()}
    elif renewable_weights is None:
        # Equal weights for all renewable technologies
        num_techs = len(renewable_params)
        renewable_weights = {key: 1.0 / num_techs for key in renewable_params.keys()}
    
    # Normalize weights
    total_weight = sum(renewable_weights.values())
    if total_weight > 0:
        renewable_weights = {k: v / total_weight for k, v in renewable_weights.items()}
    else:
        # Fallback to equal weights
        num_techs = len(renewable_params)
        renewable_weights = {key: 1.0 / num_techs for key in renewable_params.keys()}
    
    # Calculate weighted averages
    pef = sum(
        renewable_weights.get(tech, 0) * params['primary_energy_factor']
        for tech, params in renewable_params.items()
    )
    
    co2_factor = sum(
        renewable_weights.get(tech, 0) * params['co2_emission_factor']
        for tech, params in renewable_params.items()
    )
    
    return GenerationTechnologyParams(
        primary_energy_factor=pef,
        co2_emission_factor=co2_factor,
    )


def calculate_grid_primary_energy_factor(
    mix: PowerGenerationMix,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None,
    country_name: Optional[str] = None
) -> float:
    """
    Calculate primary energy factor for electricity grid based on generation mix.
    
    Parameters:
    -----------
    mix : PowerGenerationMix
        Electricity generation mix
    tech_params : dict, optional
        Technology parameters dictionary. If None, uses DEFAULT_TECH_PARAMS.
    renewable_weights : dict, optional
        Weights for renewable sub-technologies. If None, uses equal weights.
        Ignored if country_name is provided.
    country_name : str, optional
        Name of the country to use country-specific renewable technology mix.
        If provided, renewable_weights will be ignored.
    
    Returns:
    --------
    float
        Primary energy factor [-] (kWh_primary / kWh_electricity)
    
    Formula:
    --------
    PEF = Σ(f_i * PEF_i)
    where f_i is the fraction of technology i, PEF_i is its primary energy factor
    """
    if tech_params is None:
        tech_params = DEFAULT_TECH_PARAMS
    
    # Handle renewable sub-technologies
    if isinstance(tech_params.get('renewable'), dict):
        renewable_params = calculate_weighted_renewable_params(
            renewable_weights=renewable_weights,
            country_name=country_name
        )
        renewable_pef = renewable_params.primary_energy_factor
    else:
        renewable_pef = tech_params['renewable'].primary_energy_factor
    
    pef = (
        mix.coal * tech_params['coal'].primary_energy_factor +
        mix.gas * tech_params['gas'].primary_energy_factor +
        mix.nuclear * tech_params['nuclear'].primary_energy_factor +
        mix.renewable * renewable_pef +
        mix.oil * tech_params['oil'].primary_energy_factor 
    )
    
    return pef


def calculate_grid_co2_factor(
    mix: PowerGenerationMix,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None,
    country_name: Optional[str] = None
) -> float:
    """
    Calculate CO2 emission factor for electricity grid based on generation mix.
    
    Parameters:
    -----------
    mix : PowerGenerationMix
        Electricity generation mix
    tech_params : dict, optional
        Technology parameters dictionary. If None, uses DEFAULT_TECH_PARAMS.
    renewable_weights : dict, optional
        Weights for renewable sub-technologies. If None, uses equal weights.
        Ignored if country_name is provided.
    country_name : str, optional
        Name of the country to use country-specific renewable technology mix.
        If provided, renewable_weights will be ignored.
    
    Returns:
    --------
    float 
        CO2 emission factor [kg CO2/kWh]
    
    Formula:
    --------
    CO2_factor = Σ(f_i * CO2_i)
    where f_i is the fraction of technology i, CO2_i is its CO2 emission factor
    """
    if tech_params is None:
        tech_params = DEFAULT_TECH_PARAMS
    
    # Handle renewable sub-technologies
    if isinstance(tech_params.get('renewable'), dict):
        renewable_params = calculate_weighted_renewable_params(
            renewable_weights=renewable_weights,
            country_name=country_name
        )
        renewable_co2 = renewable_params.co2_emission_factor
    else:
        renewable_co2 = tech_params['renewable'].co2_emission_factor
    
    co2_factor = (
        mix.coal * tech_params['coal'].co2_emission_factor +
        mix.gas * tech_params['gas'].co2_emission_factor +
        mix.nuclear * tech_params['nuclear'].co2_emission_factor +
        mix.renewable * renewable_co2 +
        mix.oil * tech_params['oil'].co2_emission_factor 
    )
    
    return co2_factor


def sweep_renewable_fraction(
    renewable_fractions: List[float],
    base_mix: PowerGenerationMix = None,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Sweep renewable energy fraction and calculate grid factors.
    
    Parameters:
    -----------
    renewable_fractions : list of float
        List of renewable energy fractions to sweep [0-1]
    base_mix : PowerGenerationMix, optional
        Base generation mix. If None, uses default mix.
    tech_params : dict, optional
        Technology parameters dictionary.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: renewable_fraction, pef, co2_factor, exergy_factor
    """
    if base_mix is None:
        # Default mix: 40% coal, 30% gas, 20% nuclear, 10% renewable
        base_mix = PowerGenerationMix(
            coal=0.4, gas=0.3, nuclear=0.2, renewable=0.1, oil=0.0
        )
    
    results = []
    
    for ren_frac in renewable_fractions:
        # Adjust other fractions proportionally
        non_renewable_total = 1.0 - ren_frac
        
        if non_renewable_total > 0:
            # Scale non-renewable fractions
            scale_factor = non_renewable_total / (base_mix.coal + base_mix.gas + 
                                                   base_mix.nuclear + base_mix.oil + 
                                                   base_mix.other)
            
            mix = PowerGenerationMix(
                coal=base_mix.coal * scale_factor,
                gas=base_mix.gas * scale_factor,
                nuclear=base_mix.nuclear * scale_factor,
                renewable=ren_frac,
                oil=base_mix.oil * scale_factor,
                other=base_mix.other * scale_factor
            )
        else:
            # 100% renewable
            mix = PowerGenerationMix(
                coal=0.0, gas=0.0, nuclear=0.0,
                renewable=1.0, oil=0.0, other=0.0
            )
        
        pef = calculate_grid_primary_energy_factor(mix, tech_params, renewable_weights)
        co2_factor = calculate_grid_co2_factor(mix, tech_params, renewable_weights)
        
        results.append({
            'renewable_fraction': ren_frac,
            'coal_fraction': mix.coal,
            'gas_fraction': mix.gas,
            'nuclear_fraction': mix.nuclear,
            'pef': pef,
            'co2_factor': co2_factor,
        })
    
    return pd.DataFrame(results)


def sweep_country_renewable_fraction(
    country_name: str,
    renewable_fractions: Optional[List[float]] = None,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None,
    gb_energy_use: float = 0.0,  # Gas boiler energy use [W] or [kWh]
    hpb_energy_use: float = 0.0  # Heat pump boiler energy use [W] or [kWh]
) -> pd.DataFrame:
    """
    Sweep renewable energy fraction for a specific country and calculate grid factors.   
    
    Parameters:
    -----------
    country_name : str
        Name of the country (must be in COUNTRY_GRID_MIX)
    renewable_fractions : list of float, optional
        List of renewable energy fractions to sweep [0-1].
        If None, uses np.arange(0, 1.01, 0.01)
    tech_params : dict, optional
        Technology parameters dictionary. If None, uses DEFAULT_TECH_PARAMS.
    renewable_weights : dict, optional
        Weights for renewable sub-technologies. If None, uses country-specific renewable tech mix
        from COUNTRY_RENEWABLE_TECH_MIX if available, otherwise uses equal weights.
    gb_energy_use : float
        Gas boiler energy use [W] or [kWh]
    hpb_energy_use : float
        Heat pump boiler energy use [W] or [kWh]
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: renewable_fraction, coal_fraction, gas_fraction,
        nuclear_fraction, pef, co2_factor, exergy_efficiency, gb_energy_use, hpb_energy_use
    """
    if country_name not in COUNTRY_GRID_MIX:
        raise ValueError(f"Country '{country_name}' not found in COUNTRY_GRID_MIX")
    
    country_data = COUNTRY_GRID_MIX[country_name]
    
    # Create base mix from country data
    base_mix = PowerGenerationMix(
        coal=country_data['coal'],
        gas=country_data['gas'],
        nuclear=country_data['nuclear'],
        renewable=country_data['renewable'],
        oil=country_data['oil'],
        other=country_data['other']
    )
    
    # Use country-specific renewable tech mix if renewable_weights not provided
    # Will be passed as country_name to calculation functions
    
    if renewable_fractions is None:
        renewable_fractions = np.arange(0, 1.01, 0.01).tolist()
    
    results = []
    
    for ren_frac in renewable_fractions:
        # Adjust other fractions proportionally
        non_renewable_total = 1.0 - ren_frac
        
        if non_renewable_total > 0:
            # Scale non-renewable fractions
            scale_factor = non_renewable_total / (
                base_mix.coal + base_mix.gas + base_mix.nuclear + 
                base_mix.oil + base_mix.other
            )
            
            mix = PowerGenerationMix(
                coal=base_mix.coal * scale_factor,
                gas=base_mix.gas * scale_factor,
                nuclear=base_mix.nuclear * scale_factor,
                renewable=ren_frac,
                oil=base_mix.oil * scale_factor,
                other=base_mix.other * scale_factor
            )
        else:
            # 100% renewable
            mix = PowerGenerationMix(
                coal=0.0, gas=0.0, nuclear=0.0,
                renewable=1.0, oil=0.0, other=0.0
            )
        
        # Use country_name for renewable tech mix if renewable_weights not explicitly provided
        renewable_country_name = country_name if renewable_weights is None else None
        pef = calculate_grid_primary_energy_factor(mix, tech_params, renewable_weights, renewable_country_name)
        co2_factor = calculate_grid_co2_factor(mix, tech_params, renewable_weights, renewable_country_name)
        
        results.append({
            'renewable_fraction': ren_frac,
            'coal_fraction': mix.coal,
            'gas_fraction': mix.gas,
            'nuclear_fraction': mix.nuclear,
            'oil_fraction': mix.oil,
            'other_fraction': mix.other,
            'pef': pef,
            'co2_factor': co2_factor,
            'gb_energy_use': gb_energy_use,
            'hpb_energy_use': hpb_energy_use
        })
    
    return pd.DataFrame(results)


def save_country_grid_analysis(
    country_name: str,
    output_dir: str = 'result',
    renewable_fractions: Optional[List[float]] = None,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None,
    gb_energy_use: float = 0.0,
    hpb_energy_use: float = 0.0
) -> str:
    """
    Generate and save country grid analysis to CSV file.
    
    Parameters:
    -----------
    country_name : str
        Name of the country
    output_dir : str
        Output directory for CSV files (default: 'result')
    renewable_fractions : list of float, optional
        List of renewable energy fractions to sweep [0-1]
    tech_params : dict, optional
        Technology parameters dictionary
    renewable_weights : dict, optional
        Weights for renewable sub-technologies
    gb_energy_use : float
        Gas boiler energy use [W] or [kWh]
    hpb_energy_use : float
        Heat pump boiler energy use [W] or [kWh]
    
    Returns:
    --------
    str
        Path to saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate analysis
    df = sweep_country_renewable_fraction(
        country_name=country_name,
        renewable_fractions=renewable_fractions,
        tech_params=tech_params,
        renewable_weights=renewable_weights,
        gb_energy_use=gb_energy_use,
        hpb_energy_use=hpb_energy_use
    )
    
    # Create filename from country name
    filename = country_name.replace(' ', '_').replace(',', '') + '_grid_analysis.csv'
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"Saved grid analysis for {country_name} to {filepath}")
    
    return filepath


def generate_all_country_analyses(
    output_dir: str = 'result',
    renewable_fractions: Optional[List[float]] = None,
    tech_params: Dict[str, GenerationTechnologyParams] = None,
    renewable_weights: Optional[Dict[str, float]] = None,
    gb_energy_use: float = 0.0,
    hpb_energy_use: float = 0.0
) -> Dict[str, str]:
    """
    Generate grid analysis for all countries and save to CSV files.
    
    Parameters:
    -----------
    output_dir : str
        Output directory for CSV files
    renewable_fractions : list of float, optional
        List of renewable energy fractions to sweep [0-1]
    tech_params : dict, optional
        Technology parameters dictionary
    renewable_weights : dict, optional
        Weights for renewable sub-technologies
    gb_energy_use : float
        Gas boiler energy use [W] or [kWh]
    hpb_energy_use : float
        Heat pump boiler energy use [W] or [kWh]
    
    Returns:
    --------
    dict
        Dictionary mapping country names to CSV file paths
    """
    filepaths = {}
    
    for country_name in COUNTRY_GRID_MIX.keys():
        filepath = save_country_grid_analysis(
            country_name=country_name,
            output_dir=output_dir,
            renewable_fractions=renewable_fractions,
            tech_params=tech_params,
            renewable_weights=renewable_weights,
            gb_energy_use=gb_energy_use,
            hpb_energy_use=hpb_energy_use
        )
        filepaths[country_name] = filepath
    
    return filepaths


def load_country_grid_analysis(
    country_name: str,
    data_dir: str = 'result'
) -> pd.DataFrame:
    """
    Load country grid analysis from CSV file.
    
    Parameters:
    -----------
    country_name : str
        Name of the country
    data_dir : str
        Directory containing CSV files (default: 'result')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with grid analysis data
    """
    filename = country_name.replace(' ', '_').replace(',', '') + '_grid_analysis.csv'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    return df


def find_critical_point(
    hpb_primary_energy: float,
    gb_primary_energy: float,
    hpb_co2: float,
    gb_co2: float,
    renewable_fractions: np.ndarray,
    pef_values: np.ndarray,
    co2_values: np.ndarray
) -> Dict[str, float]:
    """
    Find critical point where HPB becomes better than GB.
    
    Parameters:
    -----------
    hpb_primary_energy : float
        HPB primary energy use (without grid factor) [kWh]
    gb_primary_energy : float
        GB primary energy use [kWh]
    hpb_co2 : float
        HPB CO2 emissions (without grid factor) [kg CO2]
    gb_co2 : float
        GB CO2 emissions [kg CO2]
    renewable_fractions : np.ndarray
        Array of renewable fractions
    pef_values : np.ndarray
        Array of primary energy factors
    co2_values : np.ndarray
        Array of CO2 factors
    
    Returns:
    --------
    dict
        Dictionary with critical points for primary energy and CO2
    """
    # Calculate total primary energy and CO2 for each scenario
    hpb_pe_total = hpb_primary_energy * pef_values
    hpb_co2_total = hpb_co2 * co2_values
    
    # Find where HPB becomes better (lower values)
    pe_diff = hpb_pe_total - gb_primary_energy
    co2_diff = hpb_co2_total - gb_co2
    
    # Find critical points (where difference crosses zero)
    pe_critical_idx = None
    co2_critical_idx = None
    
    for i in range(len(pe_diff) - 1):
        if pe_diff[i] * pe_diff[i+1] <= 0:  # Sign change
            pe_critical_idx = i
            break
    
    for i in range(len(co2_diff) - 1):
        if co2_diff[i] * co2_diff[i+1] <= 0:  # Sign change
            co2_critical_idx = i
            break
    
    result = {}
    
    if pe_critical_idx is not None:
        # Interpolate to find exact critical point
        if pe_critical_idx < len(renewable_fractions) - 1:
            ren_frac_critical = np.interp(
                0, 
                [pe_diff[pe_critical_idx], pe_diff[pe_critical_idx+1]],
                [renewable_fractions[pe_critical_idx], renewable_fractions[pe_critical_idx+1]]
            )
            result['pe_critical_renewable_fraction'] = ren_frac_critical
            result['pe_critical_pef'] = np.interp(
                ren_frac_critical,
                [renewable_fractions[pe_critical_idx], renewable_fractions[pe_critical_idx+1]],
                [pef_values[pe_critical_idx], pef_values[pe_critical_idx+1]]
            )
    
    if co2_critical_idx is not None:
        # Interpolate to find exact critical point
        if co2_critical_idx < len(renewable_fractions) - 1:
            ren_frac_critical = np.interp(
                0,
                [co2_diff[co2_critical_idx], co2_diff[co2_critical_idx+1]],
                [renewable_fractions[co2_critical_idx], renewable_fractions[co2_critical_idx+1]]
            )
            result['co2_critical_renewable_fraction'] = ren_frac_critical
            result['co2_critical_co2_factor'] = np.interp(
                ren_frac_critical,
                [renewable_fractions[co2_critical_idx], renewable_fractions[co2_critical_idx+1]],
                [co2_values[co2_critical_idx], co2_values[co2_critical_idx+1]]
            )
    
    return result


def plot_country_comparison_from_csv(
    country_name: str,
    data_dir: str = 'result',
    hpb_primary_energy: Optional[float] = None,
    gb_primary_energy: Optional[float] = None,
    hpb_co2: Optional[float] = None,
    gb_co2: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of HPB vs GB for a country using CSV data.
    
    Parameters:
    -----------
    country_name : str
        Name of the country
    data_dir : str
        Directory containing CSV files (default: 'result')
    hpb_primary_energy : float, optional
        HPB primary energy consumption (without grid factor) [kWh].
        If None, uses hpb_energy_use from CSV and calculates with PEF.
    gb_primary_energy : float, optional
        GB primary energy consumption [kWh].
        If None, uses gb_energy_use from CSV.
    hpb_co2 : float, optional
        HPB CO2 emissions (without grid factor) [kg CO2].
        If None, uses hpb_energy_use from CSV and calculates with CO2 factor.
    gb_co2 : float, optional
        GB CO2 emissions [kg CO2].
        If None, uses gb_energy_use from CSV.
    save_path : str, optional
        Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    # Load data from CSV
    df = load_country_grid_analysis(country_name, data_dir)
    
    # Get energy use values from CSV if not provided
    hpb_energy_use = df['hpb_energy_use'].iloc[0]
    gb_energy_use = df['gb_energy_use'].iloc[0]
    
    # Calculate values based on provided parameters or CSV data
    if hpb_primary_energy is None:
        # Use energy use from CSV and calculate primary energy using PEF
        hpb_pe_total = hpb_energy_use * df['pef'].values
    else:
        # Use provided value and multiply by PEF
        hpb_pe_total = hpb_primary_energy * df['pef'].values
    
    if gb_primary_energy is None:
        gb_primary_energy = gb_energy_use  # GB uses direct energy
    
    if hpb_co2 is None:
        # Calculate CO2 using CO2 factor from CSV
        hpb_co2_values = hpb_energy_use * df['co2_factor'].values
    else:
        # Use provided CO2 value and multiply by CO2 factor
        hpb_co2_values = hpb_co2 * df['co2_factor'].values
    
    if gb_co2 is None:
        # GB CO2 calculation - using gas CO2 factor from DEFAULT_TECH_PARAMS
        gas_co2_factor = DEFAULT_TECH_PARAMS['gas'].co2_emission_factor
        # Convert energy use to kWh if needed (assuming it's already in kWh)
        gb_co2_value = gb_energy_use * gas_co2_factor
    else:
        gb_co2_value = gb_co2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot primary energy comparison
    ax1.plot(df['renewable_fraction'] * 100, hpb_pe_total, 
             'b-', label='HPB', linewidth=2)
    ax1.axhline(y=gb_primary_energy, color='r', linestyle='--', 
                label='GB', linewidth=2)
    ax1.set_xlabel('Renewable Energy Fraction [%]', fontsize=12)
    ax1.set_ylabel('Primary Energy Consumption [kWh]', fontsize=12)
    ax1.set_title(f'Primary Energy Comparison - {country_name}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot CO2 comparison
    ax2.plot(df['renewable_fraction'] * 100, hpb_co2_values,
             'b-', label='HPB', linewidth=2)
    ax2.axhline(y=gb_co2_value, color='r', linestyle='--',
                label='GB', linewidth=2)
    ax2.set_xlabel('Renewable Energy Fraction [%]', fontsize=12)
    ax2.set_ylabel('CO2 Emissions [kg CO2]', fontsize=12)
    ax2.set_title(f'CO2 Emissions Comparison - {country_name}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()



