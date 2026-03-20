# EXP21 Main Porting Guide - 2026-03-20

## 목적

이 문서는 현재 브랜치에서 성능이 가장 안정적인 `exp21` 모델을 `origin/main` 브랜치에서 실행하기 위해,

- 어떤 파일 차이가 있는지
- `origin/main`에 무엇을 추가해야 하는지
- 최소 반영 범위가 무엇인지

를 정리한 포팅 가이드다.

대상 모델:
- `models/exp21_strict_upgrade_30day_ops/final_model.zip`

같이 전달해야 하는 파일:
- `models/exp21_strict_upgrade_30day_ops/train_config_used.json`

## 왜 추가 작업이 필요한가

`exp21`은 단순 PPO 모델이 아니라 아래 전제를 깔고 학습되었다.

1. `MaskablePPO` 기반
2. 런타임에 `train_config_used.json`에서 `game_overrides`, `env_options`를 다시 읽음
3. `action_mask`를 예측 시점에 같이 넘김
4. `disable_auto_buy_action=true`인 strict-upgrade 구조

즉 `origin/main`이 이 전제를 모르면:

- 30일/1500 기준이 아니라 다른 게임 설정으로 실행될 수 있고
- strict-upgrade 모델인데 auto-buy 액션이 다시 열릴 수 있고
- `MaskablePPO` 모델인데 action mask 없이 예측해서 이상하게 움직일 수 있다

## 파일별 차이와 필요한 반영

### 1. `ai/agent.py`

현재 브랜치에서 `origin/main` 대비 중요한 차이:

1. `MaskablePPO` 알고리즘 이름 인식
- `_ALGO_PATH_HINTS`에 `maskableppo`, `maskable_ppo` 추가

2. 저장된 학습 설정 읽기
- `_load_saved_config(model_path)`
- `_detect_algo(model_path, algo_name)`

3. `RandomAgent.predict()`가 `action_mask`를 받을 수 있음

4. `TrainedAgent`가 `MaskablePPO`를 직접 로드할 수 있음
- `algo_name == "MaskablePPO"`면 `sb3_contrib.MaskablePPO.load(...)`

5. `TrainedAgent.predict()`가 `action_mask`를 받아
- `self.model.predict(..., action_masks=action_mask)`로 호출함

6. observation 길이 mismatch 완화
- `_adapt_observation()` 추가
- 런타임 observation 길이가 달라도 최소한 잘라내기/제로패딩으로 대응

#### main에 꼭 들어가야 하는 이유

`exp21`은 `MaskablePPO + action_mask` 구조다.
이 지원이 없으면:

- 모델 로드 자체가 실패하거나
- 로드는 되더라도 invalid action이 섞여 실행 품질이 망가질 수 있다

#### 최소 반영 체크리스트

- `MaskablePPO` 로드 지원
- `predict(obs, action_mask=...)` 시그니처 지원
- `self.model.predict(..., action_masks=action_mask)` 호출
- `_load_saved_config()` 또는 동등 기능

### 2. `modes/model_runtime.py`

이 파일은 `origin/main`에는 없고 현재 브랜치에서 새로 추가된 공통 헬퍼다.

역할:
- 모델 폴더의 `train_config_used.json`을 읽음
- `game_overrides`
- `env_options`
를 런타임에서 꺼내줌

핵심 함수:
- `load_model_runtime_options(model_path) -> (game_overrides, env_options)`

#### main에 꼭 들어가야 하는 이유

`exp21`은 학습 시 아래 값으로 돌았다.

- `target_money = 1500`
- `day_limit = 30`
- `disable_auto_buy_action = true`

이 값들을 런타임에 다시 적용하지 않으면:

- 다른 게임 룰로 실행될 수 있고
- strict-upgrade 전제가 깨질 수 있다

#### 최소 반영 체크리스트

- `train_config_used.json` 읽기
- `game_overrides`에서 `_comment` 같은 키 제외
- `env_options`에서 `_comment` 같은 키 제외
- watch/ai/versus 모드에서 재사용

### 3. `modes/watch_mode.py`

현재 브랜치에서 `origin/main` 대비 중요한 차이:

1. `modes.model_runtime.load_model_runtime_options` 사용

2. `Shop(...)` 생성 시 모델 저장 당시 설정 반영
- `target_money`
- `day_limit`
- `**env_options`

3. 액션 수 확장 반영
- `NUM_ACTIONS` 사용
- action names가 12개 기준으로 확장됨

4. 예측 시 `action_mask` 전달
- `mask = self.shop.get_action_mask()`
- `action = self.agent.predict(obs, action_mask=mask)`

5. 액션 분포 디버그 패널도 `NUM_ACTIONS` 기준으로 동작

#### main에 꼭 들어가야 하는 이유

watch 모드에서 이 부분이 빠지면:

- strict 모델이 학습 때와 다른 action space에서 움직임
- runtime env 설정이 달라짐
- 디버그 패널도 예전 7개 액션 기준이라 실제 행동 해석이 틀어짐

#### 최소 반영 체크리스트

- `load_model_runtime_options()` import
- `Shop(target_money=..., day_limit=..., **env_options)`
- `mask = self.shop.get_action_mask()`
- `self.agent.predict(obs, action_mask=mask)`
- `NUM_ACTIONS` 기반 action count / panel 처리

## origin/main에 반영해야 하는 최소 코드

우선순위 기준:

1. `ai/agent.py`
- `MaskablePPO` 로드
- `action_mask` 지원

2. `modes/model_runtime.py`
- 새 파일 추가

3. `modes/watch_mode.py`
- runtime config 로드
- action mask 전달

이 세 개가 없으면 `exp21`을 watch 모드에서 정상 재생했다고 보기 어렵다.

## 권장 전달 패키지

팀장에게 넘길 때 권장 구성:

### 모델 파일
- `models/exp21_strict_upgrade_30day_ops/final_model.zip`
- `models/exp21_strict_upgrade_30day_ops/train_config_used.json`

### 코드 파일
- `ai/agent.py`
- `modes/model_runtime.py`
- `modes/watch_mode.py`

### 선택 반영
같은 모델을 watch 외에도 AI/versus에서 안정적으로 돌리려면 같이 맞추는 편이 좋다.
- `modes/ai_mode.py`
- `modes/versus_mode.py`

## 실행 커맨드

watch 모드:
```bash
python main.py --mode watch --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

AI 모드:
```bash
python main.py --mode ai --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

## 한 줄 결론

`exp21`을 `origin/main`에서 정상 실행하려면 모델만 넘기면 부족하고,

- `ai/agent.py`
- `modes/model_runtime.py`
- `modes/watch_mode.py`

이 세 파일 수준의 포팅이 최소 필요하다.

핵심은 두 가지다.

1. `MaskablePPO + action_mask` 지원
2. `train_config_used.json` 기반 런타임 설정 복원

## 7. 이 수정으로 exp26 / exp27 / exp28도 실행되나

짧게 말하면:
- **대부분은 같이 살아난다**
- 다만 **완전 재현**은 `main` 브랜치의 환경 구조가 얼마나 최신인지에 달려 있다

이유:
- `exp21`, `exp26`, `exp27`, `exp28`은 모두 같은 계열이다
- 공통 전제:
  - `MaskablePPO`
  - strict-upgrade (`disable_auto_buy_action=true`)
  - `train_config_used.json`의 `target_money`, `day_limit`, `env_options` 복원 필요
  - 예측 시 `action_mask` 필요

즉 이 문서에서 정리한 3개 파일 포팅은 `exp21` 전용이 아니라, 사실상 `exp26/27/28`까지 포함한 공통 최소 조건이다.

### 7.1 바로 실행될 가능성이 큰 조건

아래가 `origin/main`에도 이미 비슷하게 있으면:

- 현재와 같은 액션 수 구조
- strict-upgrade 관련 액션/마스크 구조
- observation 차이가 크지 않음

그러면 다음 모델들도 거의 문제 없이 실행될 가능성이 높다.

- `exp21`
- `exp26`
- `exp27`
- `exp28`

### 7.2 추가 포팅이 필요할 수 있는 조건

반대로 `origin/main`이 예전 구조라면, 아래 파일까지 더 맞춰야 한다.

- `ai/gym_env.py`
- `core/shop.py`
- `config/settings.py`

왜냐하면 이 파일들이 실제로 아래를 결정하기 때문이다.

1. observation 길이와 구성
2. 액션 수와 액션 의미
3. strict-upgrade 관련 action mask
4. 업그레이드 액션 정의

즉 `ai/agent.py`의 `_adapt_observation()`이 길이 mismatch를 어느 정도 완화해주긴 하지만,
이건 안전장치일 뿐이고 구조가 너무 다르면 완전한 재현은 어렵다.

## 8. 적용 범위 체크리스트

### 8.1 최소 실행 조건

아래 3개가 들어가면:
- `ai/agent.py`
- `modes/model_runtime.py`
- `modes/watch_mode.py`

이 계열 모델은 적어도 아래 조건은 충족한다.

- `MaskablePPO` 로드 가능
- `action_mask`와 함께 예측 가능
- 모델 저장 당시 `target_money/day_limit/env_options` 복원 가능

즉 `watch`에서 “아예 이상하게 망가지는 문제”는 크게 줄어든다.

### 8.2 완전 재현 조건

아래까지 현재 브랜치와 맞아야 한다.

- `ai/gym_env.py`
- `core/shop.py`
- `config/settings.py`

이게 맞아야:
- observation semantics
- action semantics
- strict-upgrade behavior

가 학습 당시와 최대한 같아진다.

## 9. 팀장 전달 시 설명 문구 추천

팀장에게는 이렇게 설명하면 된다.

- `exp21`은 strict-upgrade + maskable PPO 기반이라 모델 파일만으로는 부족합니다.
- 최소한 `ai/agent.py`, `modes/model_runtime.py`, `modes/watch_mode.py` 수준의 포팅이 필요합니다.
- 이 수정은 `exp21`뿐 아니라 `exp26/27/28` 계열에도 공통으로 적용됩니다.
- 다만 main 브랜치의 `gym_env/shop/settings`가 너무 옛날 버전이면, 완전 재현을 위해 그쪽도 추가 동기화가 필요합니다.
