# ENEX Analysis Engine

Exergy-focused thermodynamic modeling for building energy systems

## 개요

**ENEX Analysis Engine**은 건물 에너지 시스템의 열역학적 분석을 위한 Python 라이브러리입니다. 특히 제2법칙(엑서지, exergy) 분석에 중점을 두고 있으며, 전기 보일러, 가스 보일러, 히트펌프 등 다양한 난방 시스템의 컴포넌트 모델을 제공합니다. 각 컴포넌트에 대해 에너지(energy), 엔트로피(entropy), 엑서지(exergy) 밸런스를 일관되게 계산할 수 있습니다.

이 라이브러리는 교육, 연구, 프로토타이핑 목적으로 설계되었으며, 특히 엑서지 분석이 중요한 건물 에너지 시스템 연구에 적합합니다.

## 주요 기능

### 컴포넌트 모델

- **전기 보일러** (`ElectricBoiler`): 전기 히터를 사용하는 온수 시스템
- **가스 보일러** (`GasBoiler`): 천연가스 연소를 사용하는 온수 시스템
- **공기원 히트펌프 보일러** (`HeatPumpBoiler`): 공기원 히트펌프를 사용하는 온수 시스템
- **태양열 보조 가스 보일러** (`SolarAssistedGasBoiler`): 태양열 집열기와 가스 보일러를 결합한 시스템
- **지열 히트펌프 보일러** (`GroundSourceHeatPumpBoiler`): 지열 히트펌프를 사용하는 온수 시스템
- **공기원 히트펌프** (`AirSourceHeatPump_cooling`, `AirSourceHeatPump_heating`): 냉방/난방용 공기원 히트펌프
- **지열 히트펌프** (`GroundSourceHeatPump_cooling`, `GroundSourceHeatPump_heating`): 냉방/난방용 지열 히트펌프
- **전기 히터** (`ElectricHeater`): 전기 히터의 동적 열전달 분석
- **보조 장치**: 팬(`Fan`), 펌프(`Pump`) 모델

### 열역학적 분석

각 컴포넌트는 다음 세 가지 밸런스를 자동으로 계산합니다:

1. **에너지 밸런스**: 제1법칙 열역학에 따른 에너지 보존
2. **엔트로피 밸런스**: 엔트로피 생성 및 전달 분석
3. **엑서지 밸런스**: 제2법칙 열역학에 따른 엑서지 소멸 분석

### 유틸리티 기능

- 단위 변환 함수 (온도, 시간, 길이, 에너지, 전력 등)
- COP(성능계수) 계산 함수
- 지열 시스템용 g-function 계산
- 자연 대류 열전달 계수 계산
- 밸런스 결과 출력 및 시각화

## 설치

### 요구사항

- Python >= 3.10
- uv 패키지 관리자

### 설치 방법

#### Option A: 로컬 개발 환경 (기여자용, 권장)

```bash
# 1) uv 설치 (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac의 경우
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 저장소 클론
git clone https://github.com/BET-lab/enex_analysis_engine.git
cd enex_analysis_engine

# 3) 가상 환경 생성 및 동기화
uv sync
```

#### Option B: 패키지로 설치

```bash
# 패키지 설치 (향후 PyPI 배포 시)
pip install enex-analysis
```

## 빠른 시작

### 기본 사용 예시

```python
from enex_analysis import ElectricBoiler

# 전기 보일러 시스템 생성
boiler = ElectricBoiler()

# 시스템 파라미터 설정
boiler.T_w_tank = 60  # 탱크 온수 온도 [°C]
boiler.T_w_sup = 10   # 공급수 온도 [°C]
boiler.T_w_serv = 45  # 사용 온수 온도 [°C]
boiler.dV_w_serv = 1.2  # 사용 유량 [L/min]

# 시스템 계산 실행
boiler.system_update()

# 결과 확인
print(f"전기 히터 입력 전력: {boiler.E_heater:.2f} W")
print(f"엑서지 효율: {boiler.X_eff:.3f}")

# 엑서지 밸런스 출력
from enex_analysis import print_balance
print_balance(boiler.exergy_balance)
```

### 출력 예시

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

## 프로젝트 구조

```
enex_analysis_engine/
├── src/
│   └── enex_analysis/
│       ├── __init__.py          # 패키지 초기화
│       ├── calc_util.py          # 단위 변환 유틸리티
│       ├── enex_engine.py        # 정적 시스템 모델
│       └── enex_engine_dynamic.py # 동적 시스템 모델
├── pyproject.toml                # 프로젝트 설정
├── uv.lock                       # 의존성 잠금 파일
└── README.md                     # 이 파일
```

### 주요 모듈

- **`calc_util.py`**: 단위 변환 상수 및 헬퍼 함수
  - 온도 변환: `C2K()`, `K2C()`
  - 시간 변환: `h2s`, `s2h` 등
  - 에너지/전력 변환: `J2kWh`, `W2kW` 등

- **`enex_engine.py`**: 정적(steady-state) 시스템 모델
  - 모든 보일러 및 히트펌프 컴포넌트 클래스
  - COP 계산 함수
  - 보조 함수들 (열전달 계수, g-function 등)

- **`enex_engine_dynamic.py`**: 동적(dynamic) 시스템 모델
  - 시간에 따른 변화를 고려한 시뮬레이션 모델

## 핵심 개념

### 엑서지(Exergy)란?

엑서지는 시스템이 주변 환경과 평형을 이룰 때까지 할 수 있는 최대 유용한 일의 양입니다. 엑서지 분석은 에너지의 "품질"을 평가하는 데 사용되며, 제2법칙 열역학의 관점에서 시스템의 효율성을 분석합니다.

엑서지 소멸(consumption)은 비가역성(irreversibility)의 척도이며, 다음 관계식으로 계산됩니다:

$$X_c = T_0 \cdot S_g$$

여기서:
- $X_c$: 엑서지 소멸 [W]
- $T_0$: 기준 온도 [K]
- $S_g$: 엔트로피 생성률 [W/K]

### 엑서지 효율

엑서지 효율은 유용한 엑서지 출력과 엑서지 입력의 비율로 정의됩니다:

$$\eta_{ex} = \frac{X_{out}}{X_{in}}$$

이 값은 0과 1 사이이며, 1에 가까울수록 열역학적으로 효율적인 시스템입니다.

## 문서

- **[IO_DOCS.md](IO_DOCS.md)**: 모든 컴포넌트의 상세한 입출력 인터페이스 문서
- **[EXAMPLES.md](EXAMPLES.md)**: 다양한 사용 예시 및 튜토리얼

## 의존성

주요 의존성 패키지:

- `numpy`: 수치 계산
- `scipy`: 과학 계산 (최적화, 적분 등)
- `matplotlib`: 시각화
- `dartwork-mpl`: 플롯 스타일링 (https://github.com/dartwork-repo/dartwork-mpl)
- `pandas`: 데이터 처리
- `dataclasses`: 데이터 클래스 지원

전체 의존성 목록은 `pyproject.toml`을 참조하세요.

## 라이센스

[라이센스 정보 추가 필요]

## 기여

기여를 환영합니다! 이슈를 제기하거나 풀 리퀘스트를 보내주세요.

## 참고 문헌

이 라이브러리는 다음 연구 및 이론을 기반으로 합니다:

- Shukuya, M. (2013). *Exergy theory and applications in the built environment*. Springer.
- 지열 히트펌프 COP 계산: [논문 참조](https://www.sciencedirect.com/science/article/pii/S0360544219304347)
- 공기원 히트펌프 COP 계산: [IBPSA 논문](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1118.pdf)

## 연락처

프로젝트 관리자: Habin Jo (habinjo0608@gmail.com)
