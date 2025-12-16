# ENEX Analysis Engine - 사용 예시

이 문서는 ENEX Analysis Engine의 다양한 사용 예시를 제공합니다.

## 목차

1. [기본 사용법](#기본-사용법)
2. [전기 보일러 분석](#전기-보일러-분석)
3. [가스 보일러 분석](#가스-보일러-분석)
4. [히트펌프 시스템 분석](#히트펌프-시스템-분석)
5. [밸런스 출력 및 분석](#밸런스-출력-및-분석)
6. [동적 시뮬레이션](#동적-시뮬레이션)
7. [여러 시스템 비교 분석](#여러-시스템-비교-분석)
8. [팬 및 펌프 사용 예시](#팬-및-펌프-사용-예시)

---

## 기본 사용법

### 예시 1: 간단한 전기 보일러 분석

```python
from enex_analysis import ElectricBoiler, print_balance

# 전기 보일러 시스템 생성
boiler = ElectricBoiler()

# 시스템 파라미터 설정
boiler.T_w_tank = 60  # 탱크 온수 온도 [°C]
boiler.T_w_sup = 10   # 공급수 온도 [°C]
boiler.T_w_serv = 45  # 사용 온수 온도 [°C]
boiler.dV_w_serv = 1.2  # 사용 유량 [L/min]

# 탱크 크기 설정
boiler.r0 = 0.2  # 탱크 반경 [m]
boiler.H = 0.8   # 탱크 높이 [m]

# 시스템 계산 실행
boiler.system_update()

# 결과 출력
print(f"전기 히터 입력 전력: {boiler.E_heater:.2f} W")
print(f"탱크 열손실: {boiler.Q_l_tank:.2f} W")
print(f"엑서지 효율: {boiler.X_eff:.3f}")

# 엑서지 밸런스 출력
print_balance(boiler.exergy_balance, decimal=2)
```

**출력 예시:**
```
전기 히터 입력 전력: 5234.56 W
탱크 열손실: 234.56 W
엑서지 효율: 0.456

HOT WATER TANK EXERGY BALANCE: =====================

IN ENTRIES:
E_heater: 5234.56 [W]
X_w_sup_tank: 123.45 [W]

OUT ENTRIES:
X_w_tank: 4567.89 [W]
X_l_tank: 234.56 [W]

CONSUMED ENTRIES:
X_c_tank: 555.56 [W]

MIXING VALVE EXERGY BALANCE: =====================

IN ENTRIES:
X_w_tank: 4567.89 [W]
X_w_sup_mix: 234.12 [W]

OUT ENTRIES:
X_w_serv: 4567.89 [W]

CONSUMED ENTRIES:
X_c_mix: 234.12 [W]
```

---

## 전기 보일러 분석

### 예시 2: 파라미터 변화에 따른 영향 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import ElectricBoiler

# 탱크 온도 변화에 따른 분석
tank_temps = np.linspace(50, 70, 21)  # 50°C ~ 70°C
power_inputs = []
exergy_effs = []

boiler = ElectricBoiler()
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

for T_tank in tank_temps:
    boiler.T_w_tank = T_tank
    boiler.system_update()
    power_inputs.append(boiler.E_heater)
    exergy_effs.append(boiler.X_eff)

# 결과 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

ax1.plot(tank_temps, power_inputs, 'o-', color='dm.blue6')
ax1.set_xlabel('Tank Temperature [°C]')
ax1.set_ylabel('Power Input [W]')
ax1.grid(True, alpha=0.3)

ax2.plot(tank_temps, exergy_effs, 's-', color='dm.red6')
ax2.set_xlabel('Tank Temperature [°C]')
ax2.set_ylabel('Exergy Efficiency [-]')
ax2.grid(True, alpha=0.3)

dm.simple_layout(fig)
dm.save_and_show(fig)
```

### 예시 3: 탱크 단열재 두께의 영향 분석

```python
import numpy as np
from enex_analysis import ElectricBoiler

# 단열재 두께 변화
insulation_thicknesses = np.linspace(0.05, 0.20, 16)  # 5cm ~ 20cm
heat_losses = []

boiler = ElectricBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

for x_ins in insulation_thicknesses:
    boiler.x_ins = x_ins
    boiler.system_update()
    heat_losses.append(boiler.Q_l_tank)

print("단열재 두께에 따른 열손실:")
for x_ins, Q_loss in zip(insulation_thicknesses, heat_losses):
    print(f"  {x_ins*100:.1f} cm: {Q_loss:.2f} W")
```

---

## 가스 보일러 분석

### 예시 4: 가스 보일러 기본 분석

```python
from enex_analysis import GasBoiler, print_balance

# 가스 보일러 시스템 생성
boiler = GasBoiler()

# 시스템 파라미터 설정
boiler.eta_comb = 0.9  # 연소 효율
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.T_exh = 70  # 배기가스 온도
boiler.dV_w_serv = 1.2

# 시스템 계산
boiler.system_update()

# 결과 출력
print(f"천연가스 에너지 입력: {boiler.E_NG:.2f} W")
print(f"배기가스 열손실: {boiler.Q_exh:.2f} W")
print(f"천연가스 엑서지: {boiler.X_NG:.2f} W")
print(f"전체 엑서지 소멸: {boiler.X_c_tot:.2f} W")
print(f"엑서지 효율: {boiler.X_eff:.3f}")

# 연소실 엑서지 밸런스 출력
print("\n=== 연소실 엑서지 밸런스 ===")
print_balance({"combustion chamber": boiler.exergy_balance["combustion chamber"]})
```

### 예시 5: 연소 효율의 영향 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import GasBoiler

# 연소 효율 변화
eta_comb_values = np.linspace(0.80, 0.95, 16)
energy_inputs = []
exergy_effs = []

boiler = GasBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

for eta in eta_comb_values:
    boiler.eta_comb = eta
    boiler.system_update()
    energy_inputs.append(boiler.E_NG)
    exergy_effs.append(boiler.X_eff)

# 시각화
fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))
ax.plot(eta_comb_values, exergy_effs, 'o-', color='dm.green6')
ax.set_xlabel('Combustion Efficiency [-]')
ax.set_ylabel('Exergy Efficiency [-]')
ax.grid(True, alpha=0.3)
dm.simple_layout(fig)
dm.save_and_show(fig)
```

---

## 히트펌프 시스템 분석

### 예시 6: 공기원 히트펌프 보일러 분석

```python
from enex_analysis import HeatPumpBoiler, print_balance

# 공기원 히트펌프 보일러 생성
hp_boiler = HeatPumpBoiler()

# 시스템 파라미터 설정
hp_boiler.COP = 2.5
hp_boiler.eta_fan = 0.6
hp_boiler.dP = 200  # 팬 압력 차이 [Pa]

hp_boiler.T0 = 0  # 환경 온도 [°C]
hp_boiler.T_a_ext_out = -5  # 외부 공기 출구 온도
hp_boiler.T_r_ext = -10  # 외부 냉매 온도

hp_boiler.T_w_tank = 60
hp_boiler.T_r_tank = 65  # 탱크 냉매 온도
hp_boiler.T_w_serv = 45
hp_boiler.T_w_sup = 10

hp_boiler.dV_w_serv = 1.2

# 시스템 계산
hp_boiler.system_update()

# 결과 출력
print(f"압축기 입력 전력: {hp_boiler.E_cmp:.2f} W")
print(f"외부 팬 입력 전력: {hp_boiler.E_fan:.2f} W")
print(f"총 입력 전력: {hp_boiler.E_cmp + hp_boiler.E_fan:.2f} W")
print(f"냉매 열전달률 (탱크): {hp_boiler.Q_r_tank:.2f} W")
print(f"외부 공기 유량: {hp_boiler.dV_a_ext:.3f} m³/s")
print(f"엑서지 효율: {hp_boiler.X_eff:.3f}")

# 외부 유닛 엑서지 밸런스
print("\n=== 외부 유닛 엑서지 밸런스 ===")
print_balance({"external unit": hp_boiler.exergy_balance["external unit"]})
```

### 예시 7: 공기원 히트펌프 냉방 시스템 분석

```python
from enex_analysis import AirSourceHeatPump_cooling, print_balance

# 공기원 히트펌프 냉방 시스템 생성
ashp_cooling = AirSourceHeatPump_cooling()

# 시스템 파라미터 설정
ashp_cooling.T0 = 32  # 환경 온도 [°C]
ashp_cooling.T_a_room = 20  # 실내 공기 온도
ashp_cooling.T_r_int = 10  # 내부 냉매 온도
ashp_cooling.T_a_int_out = 15  # 내부 공기 출구 온도
ashp_cooling.T_a_ext_out = 42  # 외부 공기 출구 온도
ashp_cooling.T_r_ext = 47  # 외부 냉매 온도

ashp_cooling.Q_r_int = 6000  # 실내 열부하 [W]
ashp_cooling.Q_r_max = 9000  # 최대 냉방 용량 [W]
ashp_cooling.COP_ref = 4  # 기준 COP

# 시스템 계산
ashp_cooling.system_update()

# 결과 출력
print(f"계산된 COP: {ashp_cooling.COP:.2f}")
print(f"압축기 입력 전력: {ashp_cooling.E_cmp:.2f} W")
print(f"내부 팬 입력 전력: {ashp_cooling.E_fan_int:.2f} W")
print(f"외부 팬 입력 전력: {ashp_cooling.E_fan_ext:.2f} W")
print(f"총 입력 전력: {ashp_cooling.E_cmp + ashp_cooling.E_fan_int + ashp_cooling.E_fan_ext:.2f} W")
print(f"엑서지 효율: {ashp_cooling.X_eff:.3f}")

# 전체 엑서지 밸런스
print("\n=== 전체 시스템 엑서지 밸런스 ===")
for subsystem in ashp_cooling.exergy_balance:
    print_balance({subsystem: ashp_cooling.exergy_balance[subsystem]})
```

### 예시 8: 지열 히트펌프 보일러 분석

```python
from enex_analysis import GroundSourceHeatPumpBoiler, print_balance

# 지열 히트펌프 보일러 생성
gshp_boiler = GroundSourceHeatPumpBoiler()

# 시간 파라미터
gshp_boiler.time = 10  # 시뮬레이션 시간 [h]

# 보어홀 파라미터
gshp_boiler.H_b = 200  # 보어홀 높이 [m]
gshp_boiler.r_b = 0.08  # 보어홀 반경 [m]
gshp_boiler.R_b = 0.108  # 보어홀 열저항 [mK/W]

# 지반 파라미터
gshp_boiler.k_g = 2.0  # 지반 열전도율 [W/mK]
gshp_boiler.c_g = 800  # 지반 비열 [J/kgK]
gshp_boiler.rho_g = 2000  # 지반 밀도 [kg/m³]
gshp_boiler.T_g = 11  # 지반 온도 [°C]

# 유체 파라미터
gshp_boiler.dV_f = 24  # 순환수 유량 [L/min]
gshp_boiler.E_pmp = 200  # 펌프 전력 [W]

# 온도 파라미터
gshp_boiler.T0 = 0
gshp_boiler.T_w_tank = 60
gshp_boiler.T_w_serv = 45
gshp_boiler.T_w_sup = 10
gshp_boiler.T_r_tank = 65
gshp_boiler.dT_r_exch = -5

gshp_boiler.dV_w_serv = 1.2

# 시스템 계산
gshp_boiler.system_update()

# 결과 출력
print(f"계산된 COP: {gshp_boiler.COP:.2f}")
print(f"압축기 입력 전력: {gshp_boiler.E_cmp:.2f} W")
print(f"펌프 입력 전력: {gshp_boiler.E_pmp:.2f} W")
print(f"보어홀 열전달률: {gshp_boiler.Q_bh:.2f} W/m")
print(f"g-function 값: {gshp_boiler.g_i:.4f} mK/W")
print(f"보어홀 벽 온도: {gshp_boiler.T_b:.2f} K ({gshp_boiler.T_b - 273.15:.2f} °C)")
print(f"유체 입구 온도: {gshp_boiler.T_f_in:.2f} K ({gshp_boiler.T_f_in - 273.15:.2f} °C)")
print(f"유체 출구 온도: {gshp_boiler.T_f_out:.2f} K ({gshp_boiler.T_f_out - 273.15:.2f} °C)")
print(f"엑서지 효율: {gshp_boiler.X_eff:.3f}")

# 지반 엑서지 밸런스
print("\n=== 지반 엑서지 밸런스 ===")
print_balance({"ground": gshp_boiler.exergy_balance["ground"]})
```

---

## 밸런스 출력 및 분석

### 예시 9: 에너지, 엔트로피, 엑서지 밸런스 비교

```python
from enex_analysis import ElectricBoiler, print_balance

boiler = ElectricBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

boiler.system_update()

# 에너지 밸런스 출력
print("=" * 60)
print("에너지 밸런스")
print("=" * 60)
print_balance(boiler.energy_balance)

# 엔트로피 밸런스 출력
print("\n" + "=" * 60)
print("엔트로피 밸런스")
print("=" * 60)
print_balance(boiler.entropy_balance)

# 엑서지 밸런스 출력
print("\n" + "=" * 60)
print("엑서지 밸런스")
print("=" * 60)
print_balance(boiler.exergy_balance)
```

### 예시 10: 밸런스 검증

```python
from enex_analysis import ElectricBoiler

boiler = ElectricBoiler()
boiler.T_w_tank = 60
boiler.T_w_sup = 10
boiler.T_w_serv = 45
boiler.dV_w_serv = 1.2

boiler.system_update()

# 에너지 밸런스 검증 (온수 탱크)
tank_energy = boiler.energy_balance["hot water tank"]
energy_in = sum(tank_energy["in"].values())
energy_out = sum(tank_energy["out"].values())
energy_error = abs(energy_in - energy_out)

print(f"온수 탱크 에너지 밸런스:")
print(f"  입력: {energy_in:.2f} W")
print(f"  출력: {energy_out:.2f} W")
print(f"  오차: {energy_error:.2f} W")
print(f"  상대 오차: {energy_error/energy_in*100:.4f} %")

# 엑서지 밸런스 검증
tank_exergy = boiler.exergy_balance["hot water tank"]
exergy_in = sum(tank_exergy["in"].values())
exergy_out = sum(tank_exergy["out"].values())
exergy_consumed = sum(tank_exergy["con"].values())
exergy_error = abs(exergy_in - exergy_out - exergy_consumed)

print(f"\n온수 탱크 엑서지 밸런스:")
print(f"  입력: {exergy_in:.2f} W")
print(f"  출력: {exergy_out:.2f} W")
print(f"  소멸: {exergy_consumed:.2f} W")
print(f"  오차: {exergy_error:.2f} W")
print(f"  상대 오차: {exergy_error/exergy_in*100:.4f} %")
```

---

## 동적 시뮬레이션

### 예시 11: 전기 히터 동적 시뮬레이션

```python
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import ElectricHeater

# 전기 히터 생성
heater = ElectricHeater()

# 시스템 파라미터 설정
heater.E_heater = 1000  # 입력 전력 [W]
heater.T0 = 0  # 기준 온도 [°C]
heater.T_mr = 15  # 방 표면 온도 [°C]
heater.T_a_room = 20  # 실내 공기 온도 [°C]
heater.T_init = 20  # 초기 온도 [°C]

heater.dt = 10  # 시간 간격 [s]

# 시스템 계산 (수렴할 때까지 반복)
heater.system_update()

# 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(dm.cm2in(15), dm.cm2in(10)))

# 온도 이력
axes[0, 0].plot(heater.time, [T-273.15 for T in heater.T_hb_list], 
                label='Heater Body', color='dm.red6')
axes[0, 0].plot(heater.time, [T-273.15 for T in heater.T_hs_list], 
                label='Heater Surface', color='dm.blue6')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Temperature [°C]')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 열전달률 이력
axes[0, 1].plot(heater.time, heater.Q_conv_list, 
                label='Convection', color='dm.green6')
axes[0, 1].plot(heater.time, heater.Q_rad_hs_list, 
                label='Radiation', color='dm.orange6')
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Heat Transfer Rate [W]')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 엔트로피 생성 이력
axes[1, 0].plot(heater.time, heater.S_g_hb_list, 
                label='Heater Body', color='dm.purple6')
axes[1, 0].plot(heater.time, heater.S_g_hs_list, 
                label='Heater Surface', color='dm.cyan6')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Entropy Generation [W/K]')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 엑서지 소멸 이력
axes[1, 1].plot(heater.time, heater.X_c_hb_list, 
                label='Heater Body', color='dm.red3')
axes[1, 1].plot(heater.time, heater.X_c_hs_list, 
                label='Heater Surface', color='dm.blue3')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Exergy Consumption [W]')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

dm.simple_layout(fig)
dm.save_and_show(fig)

print(f"수렴 시간: {heater.time[-1]:.1f} s")
print(f"최종 히터 본체 온도: {heater.T_hb_list[-1]-273.15:.2f} °C")
print(f"최종 히터 표면 온도: {heater.T_hs_list[-1]-273.15:.2f} °C")
print(f"엑서지 효율: {heater.X_eff:.3f}")
```

---

## 여러 시스템 비교 분석

### 예시 12: 다양한 보일러 시스템 비교

```python
import pandas as pd
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import ElectricBoiler, GasBoiler, HeatPumpBoiler

# 공통 파라미터
common_params = {
    'T_w_tank': 60,
    'T_w_sup': 10,
    'T_w_serv': 45,
    'dV_w_serv': 1.2
}

# 전기 보일러
elec_boiler = ElectricBoiler()
for key, value in common_params.items():
    setattr(elec_boiler, key, value)
elec_boiler.system_update()

# 가스 보일러
gas_boiler = GasBoiler()
for key, value in common_params.items():
    setattr(gas_boiler, key, value)
gas_boiler.system_update()

# 히트펌프 보일러
hp_boiler = HeatPumpBoiler()
for key, value in common_params.items():
    setattr(hp_boiler, key, value)
hp_boiler.COP = 2.5
hp_boiler.system_update()

# 결과 비교
results = {
    'System': ['Electric Boiler', 'Gas Boiler', 'Heat Pump Boiler'],
    'Energy Input [W]': [
        elec_boiler.E_heater,
        gas_boiler.E_NG,
        hp_boiler.E_cmp + hp_boiler.E_fan
    ],
    'Exergy Efficiency [-]': [
        elec_boiler.X_eff,
        gas_boiler.X_eff,
        hp_boiler.X_eff
    ],
    'Total Exergy Consumption [W]': [
        elec_boiler.X_c_tot,
        gas_boiler.X_c_tot,
        hp_boiler.X_c_tot
    ]
}

df = pd.DataFrame(results)
print(df.to_string(index=False))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

axes[0].bar(df['System'], df['Energy Input [W]'], color=['dm.blue6', 'dm.red6', 'dm.green6'])
axes[0].set_ylabel('Energy Input [W]')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(df['System'], df['Exergy Efficiency [-]'], color=['dm.blue6', 'dm.red6', 'dm.green6'])
axes[1].set_ylabel('Exergy Efficiency [-]')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

dm.simple_layout(fig)
dm.save_and_show(fig)
```

### 예시 13: COP 변화에 따른 히트펌프 보일러 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import HeatPumpBoiler

# COP 범위
cop_values = np.linspace(2.0, 4.0, 21)
energy_inputs = []
exergy_effs = []

hp_boiler = HeatPumpBoiler()
hp_boiler.T_w_tank = 60
hp_boiler.T_w_sup = 10
hp_boiler.T_w_serv = 45
hp_boiler.dV_w_serv = 1.2

for cop in cop_values:
    hp_boiler.COP = cop
    hp_boiler.system_update()
    energy_inputs.append(hp_boiler.E_cmp + hp_boiler.E_fan)
    exergy_effs.append(hp_boiler.X_eff)

# 시각화
fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)))
ax.plot(cop_values, exergy_effs, 'o-', color='dm.blue6', linewidth=2, markersize=4)
ax.set_xlabel('COP [-]')
ax.set_ylabel('Exergy Efficiency [-]')
ax.grid(True, alpha=0.3)
dm.simple_layout(fig)
dm.save_and_show(fig)

# 최적 COP 찾기
optimal_idx = np.argmax(exergy_effs)
print(f"최대 엑서지 효율: {exergy_effs[optimal_idx]:.3f}")
print(f"해당 COP: {cop_values[optimal_idx]:.2f}")
```

---

## 팬 및 펌프 사용 예시

### 예시 14: 팬 성능 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import Fan

# 팬 객체 생성
fan = Fan()

# 유량 범위
flow_rates = np.linspace(0.5, 3.0, 26)  # m³/s

# 팬 1 분석
pressures_1 = []
powers_1 = []
efficiencies_1 = []

for dV in flow_rates:
    try:
        pressure = fan.get_pressure(fan.fan1, dV)
        power = fan.get_power(fan.fan1, dV)
        efficiency = fan.get_efficiency(fan.fan1, dV)
        
        pressures_1.append(pressure)
        powers_1.append(power)
        efficiencies_1.append(efficiency)
    except:
        pressures_1.append(np.nan)
        powers_1.append(np.nan)
        efficiencies_1.append(np.nan)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(dm.cm2in(18), dm.cm2in(5)))

axes[0].plot(flow_rates, pressures_1, 'o-', color='dm.blue6', markersize=3)
axes[0].set_xlabel('Flow Rate [m³/s]')
axes[0].set_ylabel('Pressure [Pa]')
axes[0].grid(True, alpha=0.3)

axes[1].plot(flow_rates, powers_1, 's-', color='dm.red6', markersize=3)
axes[1].set_xlabel('Flow Rate [m³/s]')
axes[1].set_ylabel('Power [W]')
axes[1].grid(True, alpha=0.3)

axes[2].plot(flow_rates, efficiencies_1, '^-', color='dm.green6', markersize=3)
axes[2].set_xlabel('Flow Rate [m³/s]')
axes[2].set_ylabel('Efficiency [-]')
axes[2].grid(True, alpha=0.3)

dm.simple_layout(fig)
dm.save_and_show(fig)

# 성능 그래프 출력
fan.show_graph()
```

### 예시 15: 펌프 성능 분석

```python
import numpy as np
from enex_analysis import Pump

# 펌프 객체 생성
pump = Pump()

# 유량 및 압력 차이
flow_rates = np.linspace(2, 6, 21) / 3600  # m³/h를 m³/s로 변환
pressure_diff = 100000  # Pa (1 bar)

# 펌프 1 분석
powers = []
efficiencies = []

for dV in flow_rates:
    try:
        power = pump.get_power(pump.pump1, dV, pressure_diff)
        efficiency = pump.get_efficiency(pump.pump1, dV)
        
        powers.append(power)
        efficiencies.append(efficiency)
    except:
        powers.append(np.nan)
        efficiencies.append(np.nan)

print("펌프 성능 분석:")
print(f"압력 차이: {pressure_diff/1000:.1f} kPa")
print("\n유량 [m³/h] | 전력 [W] | 효율 [-]")
print("-" * 40)
for dV, P, eta in zip(flow_rates*3600, powers, efficiencies):
    if not np.isnan(P):
        print(f"{dV:8.2f}   | {P:7.2f} | {eta:.3f}")

# 성능 그래프 출력
pump.show_graph()
```

---

## 고급 예시

### 예시 16: 태양열 보조 가스 보일러 분석

```python
from enex_analysis import SolarAssistedGasBoiler, print_balance

# 태양열 보조 가스 보일러 생성
solar_boiler = SolarAssistedGasBoiler()

# 태양열 파라미터
solar_boiler.I_DN = 500  # 직달 일사량 [W/m²]
solar_boiler.I_dH = 200  # 확산 일사량 [W/m²]
solar_boiler.A_stc = 2  # 집열기 면적 [m²]
solar_boiler.alpha = 0.95  # 흡수율

# 온도 파라미터
solar_boiler.T0 = 0
solar_boiler.T_w_comb = 60
solar_boiler.T_w_serv = 45
solar_boiler.T_w_sup = 10
solar_boiler.T_exh = 70

solar_boiler.dV_w_serv = 1.2

# 시스템 계산
solar_boiler.system_update()

# 결과 출력
print(f"태양열 흡수 열전달률: {solar_boiler.Q_sol:.2f} W")
print(f"집열기 출구 온수 온도: {solar_boiler.T_w_stc_out-273.15:.2f} °C")
print(f"천연가스 에너지 입력: {solar_boiler.E_NG:.2f} W")
print(f"태양열 기여도: {solar_boiler.Q_sol/(solar_boiler.Q_sol+solar_boiler.E_NG)*100:.1f} %")
print(f"엑서지 효율: {solar_boiler.X_eff:.3f}")

# 태양열 패널 엑서지 밸런스
print("\n=== 태양열 패널 엑서지 밸런스 ===")
print_balance({"solar thermal panel": solar_boiler.exergy_balance["solar thermal panel"]})
```

### 예시 17: 일사량 변화에 따른 태양열 시스템 분석

```python
import numpy as np
import matplotlib.pyplot as plt
import dartwork_mpl as dm
from enex_analysis import SolarAssistedGasBoiler

# 일사량 범위
I_DN_values = np.linspace(0, 1000, 21)
gas_inputs = []
solar_contributions = []

solar_boiler = SolarAssistedGasBoiler()
solar_boiler.I_dH = 200
solar_boiler.A_stc = 2
solar_boiler.T_w_comb = 60
solar_boiler.T_w_serv = 45
solar_boiler.T_w_sup = 10
solar_boiler.dV_w_serv = 1.2

for I_DN in I_DN_values:
    solar_boiler.I_DN = I_DN
    solar_boiler.system_update()
    gas_inputs.append(solar_boiler.E_NG)
    total_energy = solar_boiler.Q_sol + solar_boiler.E_NG
    solar_contributions.append(solar_boiler.Q_sol / total_energy * 100)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(dm.cm2in(15), dm.cm2in(5)))

axes[0].plot(I_DN_values, gas_inputs, 'o-', color='dm.red6')
axes[0].set_xlabel('Direct Normal Irradiance [W/m²]')
axes[0].set_ylabel('Natural Gas Input [W]')
axes[0].grid(True, alpha=0.3)

axes[1].plot(I_DN_values, solar_contributions, 's-', color='dm.orange6')
axes[1].set_xlabel('Direct Normal Irradiance [W/m²]')
axes[1].set_ylabel('Solar Contribution [%]')
axes[1].grid(True, alpha=0.3)

dm.simple_layout(fig)
dm.save_and_show(fig)
```

---

## 팁 및 모범 사례

1. **항상 `system_update()` 호출**: 모든 파라미터를 설정한 후 반드시 `system_update()`를 호출해야 결과를 사용할 수 있습니다.

2. **단위 확인**: 입력 파라미터의 단위를 확인하세요. 온도는 [°C], 유량은 [L/min]로 입력합니다.

3. **밸런스 검증**: 에너지 밸런스는 입력과 출력이 일치해야 하며, 엑서지 밸런스는 입력 = 출력 + 소멸이어야 합니다.

4. **반복 계산**: 지열 히트펌프 시스템은 반복 수치해법을 사용하므로, 수렴 조건을 확인하세요.

5. **시각화**: `dartwork-mpl`을 사용하여 일관된 스타일의 그래프를 생성할 수 있습니다.

6. **성능 최적화**: 대량의 시뮬레이션을 수행할 때는 NumPy 배열을 활용하여 벡터화 연산을 사용하세요.

---

## 문제 해결

### 일반적인 오류

1. **음수 유량 비율**: `alp`가 음수가 되는 경우, 탱크 온도가 공급수 온도보다 낮거나 사용 온도가 공급수 온도보다 낮을 수 있습니다.

2. **수렴 실패**: 지열 히트펌프 시스템에서 수렴이 실패하는 경우, 초기 추정값이나 파라미터 범위를 조정하세요.

3. **온도 변환 중복**: `system_update()`를 여러 번 호출하면 온도 변환이 중복될 수 있으므로 주의하세요.

### 디버깅 팁

```python
# 밸런스 검증 함수
def verify_balance(system, subsystem_name, balance_type='exergy'):
    if balance_type == 'exergy':
        balance = system.exergy_balance[subsystem_name]
        in_sum = sum(balance['in'].values())
        out_sum = sum(balance['out'].values())
        con_sum = sum(balance['con'].values())
        error = abs(in_sum - out_sum - con_sum)
        print(f"{subsystem_name} 엑서지 밸런스 오차: {error:.2e} W")
    elif balance_type == 'energy':
        balance = system.energy_balance[subsystem_name]
        in_sum = sum(balance['in'].values())
        out_sum = sum(balance['out'].values())
        error = abs(in_sum - out_sum)
        print(f"{subsystem_name} 에너지 밸런스 오차: {error:.2e} W")

# 사용 예시
from enex_analysis import ElectricBoiler
boiler = ElectricBoiler()
boiler.system_update()
verify_balance(boiler, 'hot water tank', 'exergy')
verify_balance(boiler, 'hot water tank', 'energy')
```

---

이 문서의 예시들을 참고하여 ENEX Analysis Engine을 효과적으로 활용하시기 바랍니다. 추가 질문이나 예시가 필요하시면 이슈를 제기해 주세요.

