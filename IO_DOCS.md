# ENEX Analysis Engine - 입출력 인터페이스 문서

이 문서는 ENEX Analysis Engine의 모든 컴포넌트 클래스와 함수의 상세한 입출력 인터페이스를 설명합니다.

## 목차

1. [유틸리티 함수](#유틸리티-함수)
2. [정적 시스템 모델](#정적-시스템-모델)
3. [동적 시스템 모델](#동적-시스템-모델)
4. [보조 함수](#보조-함수)

---

## 유틸리티 함수

### `calc_util.py` 모듈

#### 온도 변환

##### `C2K(C)`
섭씨 온도를 켈빈 온도로 변환합니다.

**입력:**
- `C` (float): 섭씨 온도 [°C]

**출력:**
- `float`: 켈빈 온도 [K]

**수식:**
$$T_K = T_C + 273.15$$

##### `K2C(K)`
켈빈 온도를 섭씨 온도로 변환합니다.

**입력:**
- `K` (float): 켈빈 온도 [K]

**출력:**
- `float`: 섭씨 온도 [°C]

**수식:**
$$T_C = T_K - 273.15$$

#### 시간 변환 상수

다음 상수들이 제공됩니다:

- `h2s = 3600`: 시간을 초로 변환
- `s2h = 1/3600`: 초를 시간으로 변환
- `d2h = 24`: 일을 시간으로 변환
- `h2d = 1/24`: 시간을 일로 변환
- `m2s = 60`: 분을 초로 변환
- `s2m = 1/60`: 초를 분으로 변환

**사용 예시:**
```python
from enex_analysis import calc_util as cu

time_hours = 2.5
time_seconds = time_hours * cu.h2s  # 9000 초
```

#### 길이 변환 상수

- `m2cm = 100`: 미터를 센티미터로 변환
- `cm2m = 1/100`: 센티미터를 미터로 변환
- `m2mm = 1e3`: 미터를 밀리미터로 변환
- `mm2m = 1e-3`: 밀리미터를 미터로 변환

#### 에너지/전력 변환 상수

- `J2kWh = 1/3.6e6`: 줄을 킬로와트시로 변환
- `kWh2J = 3.6e6`: 킬로와트시를 줄로 변환
- `W2kW = 1e-3`: 와트를 킬로와트로 변환
- `kW2W = 1e3`: 킬로와트를 와트로 변환

#### 부피 변환 상수

- `m32L = 1e3`: 세제곱미터를 리터로 변환
- `L2m3 = 1e-3`: 리터를 세제곱미터로 변환

---

## 정적 시스템 모델

### `ElectricBoiler`

전기 히터를 사용하는 온수 보일러 시스템을 모델링합니다. 시스템은 온수 탱크와 혼합 밸브로 구성됩니다.

#### 입력 파라미터 (`__post_init__`)

**온도 파라미터 [°C]:**
- `T_w_tank` (float, 기본값: 60): 탱크 내 온수 온도
- `T_w_sup` (float, 기본값: 10): 공급수 온도
- `T_w_serv` (float, 기본값: 45): 사용 온수 온도
- `T0` (float, 기본값: 0): 기준 온도 (환경 온도)

**유량 파라미터:**
- `dV_w_serv` (float, 기본값: 1.2): 사용 유량 [L/min]

**탱크 기하학적 파라미터:**
- `r0` (float, 기본값: 0.2): 탱크 내부 반경 [m]
- `H` (float, 기본값: 0.8): 탱크 높이 [m]

**탱크 열적 파라미터:**
- `x_shell` (float, 기본값: 0.01): 쉘 두께 [m]
- `x_ins` (float, 기본값: 0.10): 단열재 두께 [m]
- `k_shell` (float, 기본값: 25): 쉘 열전도율 [W/mK]
- `k_ins` (float, 기본값: 0.03): 단열재 열전도율 [W/mK]
- `h_o` (float, 기본값: 15): 전체 열전달 계수 [W/m²K]

#### 출력 속성 (`system_update()` 후)

**계산된 유량:**
- `dV_w_sup_tank` (float): 탱크로 공급되는 유량 [m³/s]
- `dV_w_sup_mix` (float): 혼합 밸브로 공급되는 유량 [m³/s]
- `alp` (float): 유량 비율 [-]

**에너지 항목:**
- `E_heater` (float): 전기 히터 입력 전력 [W]
- `Q_w_tank` (float): 탱크에서 나가는 열전달률 [W]
- `Q_w_sup_tank` (float): 탱크로 들어가는 열전달률 [W]
- `Q_l_tank` (float): 탱크 열손실 [W]
- `Q_w_serv` (float): 사용 온수의 열전달률 [W]

**엑서지 효율:**
- `X_eff` (float): 전체 엑서지 효율 [-]

**밸런스 딕셔너리:**
- `energy_balance` (dict): 에너지 밸런스 (서브시스템별)
- `entropy_balance` (dict): 엔트로피 밸런스 (서브시스템별)
- `exergy_balance` (dict): 엑서지 밸런스 (서브시스템별)

**밸런스 구조:**
```python
balance = {
    "subsystem_name": {
        "in": {"term1": value1, "term2": value2, ...},
        "out": {"term3": value3, ...},
        "con": {"term4": value4, ...},  # exergy만 해당
        "gen": {"term5": value5, ...}   # entropy만 해당
    }
}
```

#### 물리적 의미

전기 보일러 시스템의 에너지 밸런스:

$$E_{heater} + Q_{w,sup,tank} = Q_{w,tank} + Q_{l,tank}$$

엑서지 소멸은 엔트로피 생성과 관련됩니다:

$$X_c = T_0 \cdot S_g$$

---

### `GasBoiler`

가스 보일러의 성능 계산 및 동적 시뮬레이션을 수행하는 고급 모델입니다. (저탕조 없이 직접 급탕 공급)

#### 정적/동적 해석 메서드
- `analyze_steady(T0, dV_w_serv, return_dict=True)`: 주어진 외기온도와 온수 수요에서 정상상태 해석
- `analyze_dynamic(simulation_period_sec, dt_s, schedule_entries, T0_schedule, ...)`: 스케줄을 통한 동적 시뮬레이션 수행 및 결과 저장

#### 주요 출력 속성 (결과 DataFrame 내)
- **에너지 항목**: `E_NG` (가스 입력), `Q_exh` (배기 손실), `Q_comb_w` (수측 열전달)
- **엑서지 항목**: `X_NG`, `X_exh`, `X_c_comb` (연소실 엑서지 파괴)

---

### `AirSourceHeatPumpBoiler`

공기원 히트펌프 보일러의 성능을 계산하며, `scipy.optimize`를 통한 최적 운전점 탐색과 `CoolProp`을 활용한 정밀한 냉매 물성치(과열도/과냉각도 고려)를 지원하는 고급 모델입니다.

#### 정적/동적 해석 메서드
- `analyze_steady(T_tank_w, T0, dV_w_serv, Q_cond_load, return_dict=True)`: 압축기 운전 주파수 및 팬 속도를 최적화하여 정상상태 성능 도출
- `analyze_dynamic(...)`: 시계열 환경/부하 데이터 기반 동적 시뮬레이션

#### 주요 출력 속성
- **에너지 항목**: `E_cmp` (압축기 전력), `E_fan_ou` (팬 전력), `Q_cond_load` (응축기 열전달) 
- **엑서지 항목**: 압축기, 응축기, 증발기 팽창밸브 각 구간의 엑서지 파괴량 (postprocess_exergy 메서드를 통해 표출)
- **성능 지표**: `COP`

---

### `SolarAssistedGasBoiler`

태양열 집열기와 가스 보일러를 결합한 시스템을 모델링합니다.

#### 입력 파라미터

**태양열 파라미터:**
- `alpha` (float, 기본값: 0.95): 집열기 흡수율 [-]
- `I_DN` (float, 기본값: 500): 직달 일사량 [W/m²]
- `I_dH` (float, 기본값: 200): 확산 일사량 [W/m²]
- `A_stc` (float, 기본값: 2): 태양열 집열기 면적 [m²]

**열전달 파라미터:**
- `h_r` (float, 기본값: 2): 공기층 복사 열전달 계수 [W/m²K]
- `x_air` (float, 기본값: 0.01): 공기층 두께 [m]
- `x_ins` (float, 기본값: 0.05): 단열재 두께 [m]
- `k_ins` (float, 기본값: 0.03): 단열재 열전도율 [W/mK]

**온도 파라미터 [°C]:**
- `T_w_comb` (float, 기본값: 60): 연소실 온수 온도
- 기타: `GasBoiler`와 동일

#### 출력 속성

**태양열 관련:**
- `Q_sol` (float): 태양열 흡수 열전달률 [W]
- `T_w_stc_out` (float): 집열기 출구 온수 온도 [K]
- `T_stc` (float): 집열기 평균 온도 [K]
- `Q_l` (float): 집열기 열손실 [W]

**엑서지 항목:**
- `X_sol` (float): 태양열 엑서지 [W]
- `X_c_stc` (float): 집열기 엑서지 소멸 [W]

**밸런스:** 서브시스템별로 구성 (태양열 패널, 연소실, 혼합 밸브)

#### 물리적 의미

태양열의 엔트로피는 다음 식으로 계산됩니다:

$$S_{sol} = k_D \cdot I_{DN}^{0.9} + k_d \cdot I_{dH}^{0.9}$$

여기서 $k_D = 0.000462$, $k_d = 0.0014$는 엔트로피 계수입니다.

---

### `GroundSourceHeatPumpBoiler`

지열원 히트펌프 보일러의 성능을 계산하는 모델입니다. g-function을 기반으로 한 지중 열교환기(Borehole) 모델링과 사이클 최적 운전점 탐색 기능을 포함합니다.

#### 정적/동적 해석 메서드
- `analyze_steady(T_tank_w, T_b_f_in, dV_mix_w_out, ...)`: 지열 루프 온도 조건에서 히트펌프 정상상태 해석
- `analyze_dynamic(...)`: g-function 중첩 원리를 적용한 지중 온도 변화 및 장기 동적 시뮬레이션

#### 주요 출력 속성
- **에너지 항목**: `E_cmp`, `E_pmp` (지중 순환 펌프), `Q_evap`, `Q_cond`
- **지중 상태**: `T_b_f_in`, `T_b_f_out` (지중 루프 입출구 온도), `T_b_wall` (보어홀 벽면 온도)
- **엑서지 항목**: 지중열교환기 및 사이클 각 콤포넌트들의 엑서지 파괴량

---

### `AirSourceHeatPump_cooling`

공기원 히트펌프 냉방 시스템을 모델링합니다.

#### 입력 파라미터

**팬 파라미터:**
- `fan_int` (Fan 객체, 기본값: `Fan().fan1`): 내부 유닛 팬
- `fan_ext` (Fan 객체, 기본값: `Fan().fan3`): 외부 유닛 팬

**성능 파라미터:**
- `Q_r_max` (float, 기본값: 9000): 최대 냉방 용량 [W]
- `COP_ref` (float, 기본값: 4): 기준 COP [-]

**온도 파라미터 [°C]:**
- `T0` (float, 기본값: 32): 환경 온도
- `T_a_room` (float, 기본값: 20): 실내 공기 온도
- `T_r_int` (float, 기본값: 10): 내부 유닛 냉매 온도
- `T_a_int_out` (float, 기본값: 15): 내부 유닛 공기 출구 온도
- `T_a_ext_out` (float, 기본값: 42): 외부 유닛 공기 출구 온도
- `T_r_ext` (float, 기본값: 47): 외부 유닛 냉매 온도

**부하 파라미터:**
- `Q_r_int` (float, 기본값: 6000): 실내 열부하 [W]

#### 출력 속성

**에너지 항목:**
- `COP` (float): 계산된 COP [-]
- `E_cmp` (float): 압축기 입력 전력 [W]
- `E_fan_int` (float): 내부 팬 입력 전력 [W]
- `E_fan_ext` (float): 외부 팬 입력 전력 [W]
- `Q_r_ext` (float): 외부 유닛 열전달률 [W]
- `dV_int` (float): 내부 유닛 공기 유량 [m³/s]
- `dV_ext` (float): 외부 유닛 공기 유량 [m³/s]

**엑서지 항목:**
- `X_r_int` (float): 내부 냉매 엑서지 [W]
- `X_r_ext` (float): 외부 냉매 엑서지 [W]
- `X_a_int_in` (float): 내부 공기 입구 엑서지 [W]
- `X_a_int_out` (float): 내부 공기 출구 엑서지 [W]
- `X_c_int` (float): 내부 유닛 엑서지 소멸 [W]
- `X_c_r` (float): 냉매 루프 엑서지 소멸 [W]
- `X_c_ext` (float): 외부 유닛 엑서지 소멸 [W]
- `X_eff` (float): 전체 엑서지 효율 [-]

**밸런스:** 서브시스템별로 구성 (내부 유닛, 냉매 루프, 외부 유닛)

---

### `AirSourceHeatPump_heating`

공기원 히트펌프 난방 시스템을 모델링합니다.

#### 입력 파라미터

**온도 파라미터 [°C]:**
- `T0` (float, 기본값: 0): 환경 온도
- `T_a_room` (float, 기본값: 20): 실내 공기 온도
- `T_r_int` (float, 기본값: 35): 내부 유닛 냉매 온도
- `T_a_int_out` (float, 기본값: 30): 내부 유닛 공기 출구 온도
- `T_a_ext_out` (float, 기본값: -5): 외부 유닛 공기 출구 온도
- `T_r_ext` (float, 기본값: -10): 외부 유닛 냉매 온도

**기타 파라미터:** `AirSourceHeatPump_cooling`과 유사

#### 출력 속성

냉방 모드와 유사하지만, 열전달 방향이 반대입니다.

---

### `GroundSourceHeatPump_cooling`

지열 히트펌프 냉방 시스템을 모델링합니다.

#### 입력 파라미터

**온도 파라미터 [°C]:**
- `T0` (float, 기본값: 32): 환경 온도
- `T_g` (float, 기본값: 15): 지반 온도
- `T_a_room` (float, 기본값: 20): 실내 공기 온도
- `T_r_exch` (float, 기본값: 25): 열교환기 냉매 온도
- `dT_r_exch` (float, 기본값: 5): 열교환기 온도 차이 [K]

**기타 파라미터:** `GroundSourceHeatPumpBoiler`와 유사

#### 출력 속성

`GroundSourceHeatPumpBoiler`와 유사하지만, 냉방 모드에 맞게 조정됩니다.

---

### `GroundSourceHeatPump_heating`

지열 히트펌프 난방 시스템을 모델링합니다.

#### 입력 파라미터

**온도 파라미터 [°C]:**
- `T0` (float, 기본값: 0): 환경 온도
- `T_g` (float, 기본값: 15): 지반 온도
- `T_r_exch` (float, 기본값: 5): 열교환기 냉매 온도
- `dT_r_exch` (float, 기본값: -5): 열교환기 온도 차이 [K]

**기타 파라미터:** `GroundSourceHeatPump_cooling`과 유사

---

### `ElectricHeater`

전기 히터의 동적 열전달을 모델링합니다. 시간에 따른 온도 변화를 계산합니다.

#### 입력 파라미터

**히터 재료 특성:**
- `c` (float, 기본값: 500): 비열 [J/kgK]
- `rho` (float, 기본값: 7800): 밀도 [kg/m³]
- `k` (float, 기본값: 50): 열전도율 [W/mK]

**기하학적 파라미터:**
- `D` (float, 기본값: 0.005): 두께 [m]
- `H` (float, 기본값: 0.8): 높이 [m]
- `W` (float, 기본값: 1.0): 너비 [m]

**전기 입력:**
- `E_heater` (float, 기본값: 1000): 히터 입력 전력 [W]

**온도 파라미터 [°C]:**
- `T0` (float, 기본값: 0): 기준 온도
- `T_mr` (float, 기본값: 15): 방 표면 온도
- `T_init` (float, 기본값: 20): 히터 초기 온도
- `T_a_room` (float, 기본값: 20): 실내 공기 온도

**복사 특성:**
- `epsilon_hs` (float, 기본값: 1): 히터 표면 방사율 [-]
- `epsilon_rs` (float, 기본값: 1): 방 표면 방사율 [-]

**시간 파라미터:**
- `dt` (float, 기본값: 10): 시간 간격 [s]

#### 출력 속성

**시간 이력 (리스트):**
- `time` (list): 시간 배열 [s]
- `T_hb_list` (list): 히터 본체 온도 이력 [K]
- `T_hs_list` (list): 히터 표면 온도 이력 [K]

**에너지 이력:**
- `E_heater_list` (list): 입력 전력 이력 [W]
- `Q_st_list` (list): 축열 열전달률 이력 [W]
- `Q_cond_list` (list): 전도 열전달률 이력 [W]
- `Q_conv_list` (list): 대류 열전달률 이력 [W]
- `Q_rad_hs_list` (list): 히터 표면 복사 열전달률 이력 [W]
- `Q_rad_rs_list` (list): 방 표면 복사 열전달률 이력 [W]

**엔트로피/엑서지 이력:**
- `S_*_list`: 각 엔트로피 항목의 시간 이력
- `X_*_list`: 각 엑서지 항목의 시간 이력

**최종 값:**
- `X_eff` (float): 엑서지 효율 [-]

**밸런스:** 서브시스템별로 구성 (히터 본체, 히터 표면)

---

### `Fan`

팬의 성능 데이터를 저장하고 분석하는 클래스입니다.

#### 입력 파라미터

`__post_init__`에서 세 가지 팬 데이터가 자동으로 로드됩니다:
- `fan1`: 원심 팬 (유량, 압력, 효율 데이터)
- `fan2`: 원심 팬 (유량, 압력, 효율 데이터)
- `fan3`: 축류 팬 (유량, 전력 데이터)

#### 메서드

##### `get_efficiency(fan, dV_fan)`
주어진 유량에 대한 팬 효율을 계산합니다.

**입력:**
- `fan` (dict): 팬 데이터 딕셔너리
- `dV_fan` (float): 유량 [m³/s]

**출력:**
- `float`: 효율 [-]

##### `get_pressure(fan, dV_fan)`
주어진 유량에 대한 팬 압력을 계산합니다.

**입력:**
- `fan` (dict): 팬 데이터 딕셔너리
- `dV_fan` (float): 유량 [m³/s]

**출력:**
- `float`: 압력 [Pa]

##### `get_power(fan, dV_fan)`
주어진 유량에 대한 팬 소비 전력을 계산합니다.

**입력:**
- `fan` (dict): 팬 데이터 딕셔너리
- `dV_fan` (float): 유량 [m³/s]

**출력:**
- `float`: 전력 [W]

**수식:**
$$P = \frac{\Delta P \cdot \dot{V}}{\eta}$$

##### `show_graph()`
유량 대비 압력 및 효율 그래프를 출력합니다.

---

### `Pump`

펌프의 성능 데이터를 저장하고 분석하는 클래스입니다.

#### 입력 파라미터

`__post_init__`에서 두 가지 펌프 데이터가 자동으로 로드됩니다:
- `pump1`: 펌프 1 (유량, 효율 데이터)
- `pump2`: 펌프 2 (유량, 효율 데이터)

#### 메서드

##### `get_efficiency(pump, dV_pmp)`
주어진 유량에 대한 펌프 효율을 계산합니다.

**입력:**
- `pump` (dict): 펌프 데이터 딕셔너리
- `dV_pmp` (float): 유량 [m³/s]

**출력:**
- `float`: 효율 [-]

##### `get_power(pump, V_pmp, dP_pmp)`
주어진 유량과 압력 차이에 대한 펌프 소비 전력을 계산합니다.

**입력:**
- `pump` (dict): 펌프 데이터 딕셔너리
- `V_pmp` (float): 유량 [m³/s]
- `dP_pmp` (float): 압력 차이 [Pa]

**출력:**
- `float`: 전력 [W]

**수식:**
$$P = \frac{\Delta P \cdot \dot{V}}{\eta}$$

##### `show_graph()`
유량 대비 효율 그래프를 출력합니다.

---

## 동적 시스템 모델

### `ElectricBoiler_Dynamic`

전기 보일러의 동적 시뮬레이션을 수행합니다. 시간에 따른 시스템 상태 변화를 계산합니다.

#### 입력 파라미터

**시간 파라미터:**
- `dt` (float, 기본값: 60): 시간 간격 [s]
- `Sim_time` (float, 기본값: 86400): 시뮬레이션 시간 [s] (24시간)

**기타 파라미터:** `ElectricBoiler`와 동일

#### 출력 속성

**시간 배열:**
- `time` (numpy.ndarray): 시간 배열 [s]

**기타:** `ElectricBoiler`와 유사하지만, 시간에 따른 변화를 고려합니다.

---

## 보조 함수

### COP 계산 함수

#### `calculate_ASHP_cooling_COP(T_a_int_out, T_a_ext_in, Q_r_int, Q_r_max, COP_ref)`

공기원 히트펌프 냉방 모드의 COP를 계산합니다.

**입력:**
- `T_a_int_out` (float): 실내 공기 온도 [K]
- `T_a_ext_in` (float): 외부 공기 온도 [K]
- `Q_r_int` (float): 실내 열부하 [W]
- `Q_r_max` (float): 최대 냉방 용량 [W]
- `COP_ref` (float): 기준 COP [-]

**출력:**
- `float`: 계산된 COP [-]

**수식:**
$$\text{PLR} = \frac{Q_{r,int}}{Q_{r,max}}$$

$$\text{EIR}_{T} = 0.38 + 0.02 \cdot T_{a,int,out} + 0.01 \cdot T_{a,ext,in}$$

$$\text{EIR}_{PLR} = 0.22 + 0.50 \cdot \text{PLR} + 0.26 \cdot \text{PLR}^2$$

$$\text{COP} = \frac{\text{PLR} \cdot \text{COP}_{ref}}{\text{EIR}_{T} \cdot \text{EIR}_{PLR}}$$

**참고:** [IBPSA 논문](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf)

---

#### `calculate_ASHP_heating_COP(T0, Q_r_int, Q_r_max)`

공기원 히트펌프 난방 모드의 COP를 계산합니다.

**입력:**
- `T0` (float): 환경 온도 [K]
- `Q_r_int` (float): 실내 열부하 [W]
- `Q_r_max` (float): 최대 난방 용량 [W]

**출력:**
- `float`: 계산된 COP [-]

**수식:**
$$\text{PLR} = \frac{Q_{r,int}}{Q_{r,max}}$$

$$\text{COP} = -7.46 \cdot (\text{PLR} - 0.0047 \cdot T_0 - 0.477)^2 + 0.0941 \cdot T_0 + 4.34$$

**참고:** [MDPI 논문](https://www.mdpi.com/2071-1050/15/3/1880)

---

#### `calculate_GSHP_COP(Tg, T_cond, T_evap, theta_hat)`

지열 히트펌프의 수정된 카르노 기반 COP를 계산합니다.

**입력:**
- `Tg` (float): 지반 온도 [K]
- `T_cond` (float): 응축기 냉매 온도 [K]
- `T_evap` (float): 증발기 냉매 온도 [K]
- `theta_hat` (float): 무차원 평균 유체 온도 [-]

**출력:**
- `float`: 계산된 COP [-]

**수식:**
$$\Delta T = T_g - T_{evap}$$

$$\text{COP} = \frac{1}{1 - \frac{T_g}{T_{cond}} + \frac{\Delta T \cdot \hat{\theta}}{T_{cond}}}$$

**참고:** [ScienceDirect 논문](https://www.sciencedirect.com/science/article/pii/S0360544219304347)

---

### 지열 시스템 함수

#### `G_FLS(t, ks, as_, rb, H)`

지열 보어홀의 g-function을 계산합니다. 이 함수는 시간 의존 열저항을 계산하는 데 사용됩니다.

**입력:**
- `t` (float): 시간 [s]
- `ks` (float): 지반 열전도율 [W/mK]
- `as_` (float): 지반 열확산율 [m²/s]
- `rb` (float): 보어홀 반경 [m]
- `H` (float): 보어홀 높이 [m]

**출력:**
- `float`: g-function 값 [mK/W]

**참고:** 이 함수는 캐싱을 사용하여 계산 속도를 향상시킵니다.

---

### 열전달 함수

#### `calc_h_vertical_plate(T_s, T_inf, L)`

수직 평판의 자연 대류 열전달 계수를 계산합니다.

**입력:**
- `T_s` (float): 표면 온도 [K]
- `T_inf` (float): 유체 온도 [K]
- `L` (float): 특성 길이 [m]

**출력:**
- `float`: 열전달 계수 [W/m²K]

**수식:**

Rayleigh 수:
$$\text{Ra}_L = \frac{g \beta \Delta T L^3}{\nu^2} \text{Pr}$$

Nusselt 수 (Churchill & Chu):
$$\text{Nu}_L = \left(0.825 + \frac{0.387 \text{Ra}_L^{1/6}}{[1 + (0.492/\text{Pr})^{9/16}]^{8/27}}\right)^2$$

열전달 계수:
$$h = \frac{\text{Nu}_L \cdot k_{air}}{L}$$

**참고:** [Churchill & Chu (1975)](https://doi.org/10.1016/0017-9310(75)90243-4)

---

### 밸런스 출력 함수

#### `print_balance(balance, decimal=2)`

에너지, 엔트로피, 또는 엑서지 밸런스를 출력합니다.

**입력:**
- `balance` (dict): 밸런스 딕셔너리
- `decimal` (int, 기본값: 2): 소수점 자릿수

**출력:**
- 없음 (콘솔에 출력)

**예시 출력:**
```
HOT WATER TANK EXERGY BALANCE: =====================

IN ENTRIES:
E_heater: 5234.56 [W]
X_w_sup_tank: 123.45 [W]

OUT ENTRIES:
X_w_tank: 4567.89 [W]
X_l_tank: 234.56 [W]

CONSUMED ENTRIES:
X_c_tank: 555.56 [W]
```

---

## 상수

### 물리 상수

다음 상수들이 `enex_engine.py`에 정의되어 있습니다:

- `c_a = 1005`: 공기 비열 [J/kgK]
- `rho_a = 1.225`: 공기 밀도 [kg/m³]
- `k_a = 0.0257`: 공기 열전도율 [W/mK]
- `c_w = 4186`: 물 비열 [J/kgK]
- `rho_w = 1000`: 물 밀도 [kg/m³]
- `mu_w = 0.001`: 물 동점성계수 [Pa·s]
- `k_w = 0.606`: 물 열전도율 [W/mK]
- `sigma = 5.67e-8`: Stefan-Boltzmann 상수 [W/m²K⁴]

### 엔트로피 계수

- `k_D = 0.000462`: 직달 일사 엔트로피 계수 [-]
- `k_d = 0.0014`: 확산 일사 엔트로피 계수 [-]

### 연료 특성

- `ex_eff_NG = 0.93`: 천연가스 엑서지 효율 [-] (Shukuya 2013)

---

## 사용 시 주의사항

1. **온도 단위**: 모든 온도 입력은 섭씨 [°C]로 입력하되, 내부적으로 켈빈 [K]로 변환됩니다.

2. **유량 단위**: 유량은 [L/min]로 입력하되, 내부적으로 [m³/s]로 변환됩니다.

3. **시스템 업데이트**: 모든 컴포넌트 클래스는 `system_update()` 메서드를 호출한 후에만 결과를 사용할 수 있습니다.

4. **반복 계산**: 일부 시스템(예: 지열 히트펌프)은 반복 수치해법을 사용하므로, 수렴 조건을 확인하세요.

5. **밸런스 딕셔너리**: 밸런스 딕셔너리는 `system_update()` 호출 후에 생성됩니다.

6. **동적 모델**: `ElectricHeater`와 같은 동적 모델은 시간 이력을 리스트로 저장하므로, 메모리 사용량에 주의하세요.


## 신재생 에너지 시스템 및 열저장 모델

### `PV_to_Converter`

태양광 패널(PV Cell), 충전 컨트롤러, 배터리, DC/AC 인버터로 이어지는 전체 시스템의 에너지, 엔트로피, 엑서지 밸런스를 계산합니다.

#### 입력 파라미터 (단위: K 및 무차원)
- `A_pv`, `alp_pv`, `eta_pv`: PV 패널 면적, 흡수율, 효율
- `I_DN`, `I_dH`: 직달 및 확산 일사량 [W/m²]
- `eta_ctrl`, `eta_batt`, `eta_DC_AC`: 각 전력 변환기기의 효율
- `T_ctrl`, `T_batt`, `T_DC_AC`: 각 기기의 운전 온도 [K]

#### 메서드
- `system_update()`: 제공된 일사량과 온도를 기반으로 밸런스 속성들을 업데이트

#### 출력 속성
- `E_pv0`, `E_pv1`, `E_pv2`, `E_pv3`: 각 단계별 생산/전달된 에너지 [W]
- 각 컴포넌트별 소멸된 엑서지(`X_c_pv`, `X_c_ctrl`, `X_c_batt`, `X_c_DC_AC`)

---

### `StratifiedTankTDMA`

1차원 열성층화(Thermal Stratification) 온수 탱크 모델로, TDMA (Tri-Diagonal Matrix Algorithm)를 활용하여 노드별 온도를 계산합니다. 부력에 의한 자연 대류를 유효 열전도율(Effective Thermal Conductivity)로 반영합니다.

#### 입력 파라미터
- `H`, `N`, `r0`: 탱크 높이, 층(노드) 개수, 내부 반경
- `x_shell`, `x_ins`, `k_shell`, `k_ins`: 탱크 벽면 및 단열 두께/열전도율
- `h_w`, `h_o`, `C_d_mix`: 내/외부 열전달 계수 및 혼합 계수

#### 주요 메서드
- `update_tank_temp(T, dt, T_in, dV_use, T_amb, T0, ...)`: 시간 간격 `dt` 동안 탱크 온도 프로파일(numpy 배열 `T`)을 업데이트
- `effective_conductivity(T_upper, T_lower)`: 역전층 발생 시 혼합 효과를 모사하기 위한 유효 열전도율 계산
- `info(as_dict, precision)`: 탱크 설정 및 현재 상태 반환

---
