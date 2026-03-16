# RL Handoff - 2026-03-13

이 문서는 2026-03-13 세션에서 진행한 강화학습 실험, 코드 변경, 현재 기준 모델, 실패/성공 방향을 다른 환경의 Codex가 그대로 이어받을 수 있도록 정리한 핸드오프 문서다.

## 현재 결론

- 현재 기준 모델: `models/exp11_priority_table_longtrain/best_model.zip`
- 현재 최고 실험: `exp11_priority_table_longtrain`
- 현재 다음 실험: `config/experiments/exp14_lategame_curriculum.json`
- 핵심 결론:
  - 초기 병목은 reward보다 observation design이었다.
  - `relative position` 관측 추가 후 실제 승리가 나오기 시작했다.
  - 마지막 테이블 편향은 `priority table` 요약 관측으로 완화했고, 장기학습으로 성능까지 회복했다.
  - stale carry / upgrade timing / trait choice는 RL보다 rule-based controller가 더 적합하다고 판단했다.

## 현재 기준 실행 커맨드

학습:
```bash
python -m ai.train --config config/experiments/exp14_lategame_curriculum.json --save-path models/exp14_lategame_curriculum
```

AI 솔로 관찰:
```bash
python main.py --mode ai --model models/exp11_priority_table_longtrain/best_model.zip --speed 4
```

AI 대전 관찰:
```bash
python main.py --mode versus --model models/exp11_priority_table_longtrain/best_model.zip --speed 4
```

exp13 룰 컨트롤러까지 켤 때만:
```bash
python main.py --mode ai --model models/exp11_priority_table_longtrain/best_model.zip --speed 4 --rule-controller
python main.py --mode versus --model models/exp11_priority_table_longtrain/best_model.zip --speed 4 --rule-controller
```

## 실험 흐름 요약

### exp01_base
- 목적: baseline 확인
- 설정: 기본 구조 그대로
- 결과: 실패
- 해석: 너무 긴 에피소드에서 초기 성공 경험 없이 이동 편향으로 붕괴

### exp02_short
- 목적: 짧은 에피소드 커리큘럼
- 변경: `target_money=400`, `day_limit=8`
- 결과: baseline보다 낫지만 실제 서빙 거의 없음
- 해석: 방향은 맞지만 첫 성공 체인 생성엔 부족

### exp03_first_success
- 목적: 첫 주문-조리-서빙 체인 생성
- 변경: `take_order`, `submit_kitchen`, `pickup_food`, `serve_food` 보상 강화
- 결과: 처음으로 실제 서빙/수익 발생
- 해석: “돈”보다 “작업 체인 완성” 보상이 유효

### exp04_finish_loop
- 목적: 완료 루프와 결제까지 강화
- 변경: customer payment / win 비중 상향
- 결과: 후퇴
- 해석: 중간 단계 체인이 아직 약한 상태에서 완료 보상을 키우면 상호작용 빈도가 줄며 다시 붕괴

### exp05_win_curriculum
- 목적: exp03 축 유지 + 첫 승리 경험 유도
- 변경: target money만 더 낮춤
- 결과: exp03보다 소폭 개선
- 해석: 새 reward보다 이미 잘 나온 축 유지가 더 효과적

### exp06_win_unlock
- 목적: 더 쉬운 승리 조건으로 빠른 승리 경험 유도
- 변경: `target_money=100`
- 결과: 후퇴
- 해석: 이 시점에서는 목표 금액을 더 내리는 것이 오히려 불안정

### exp07_relpos_obs
- 목적: “무엇을 할지”뿐 아니라 “어디로 갈지”도 보이게 만들기
- 변경: 테이블/주방/바/쓰레기통 상대좌표 관측 추가
- 결과: 큰 개선, 실제 승리 발생
- 해석: 핵심 병목은 observation design이었다

### exp08_loose_interact_mask
- 목적: strict interact mask가 너무 보수적인지 확인
- 변경: interact mask 완화
- 결과: 실패, `INTERACT` spam
- 해석: strict gating이 실제로는 필요한 제약이었다

### exp09_relpos_longtrain
- 목적: exp07 구조 유지 + 장기학습
- 변경: total timesteps 증가
- 결과: 큰 개선, 승리율 8/10
- 해석: 이 구간에서는 reward보다 sample 부족이 병목

### exp10_priority_table_obs
- 목적: 마지막 테이블 무시 편향 완화
- 변경: “가장 급한 테이블” 요약 관측 6차원 추가
- 결과: 전체 성능은 약간 후퇴, 하지만 마지막 테이블 편향은 확실히 완화
- 해석: 고정 슬롯 입력 편향을 줄이는 데는 성공

### exp11_priority_table_longtrain
- 목적: exp10 편향 완화 상태 유지 + 장기학습으로 성능 회복
- 변경: 3M 장기학습
- 결과: 성공, 현재 최고 모델
- 핵심 성능:
  - 평균 돈 약 `169.6`
  - 평균 서빙 약 `7.5`
  - 평균 이탈 약 `0.3`
  - 승리 `10/10`
- 해석: 편향 완화와 성능을 함께 잡음

### exp12_ops_alignment
- 목적: stale carry / upgrade timing / 운영 상태를 RL이 더 잘 배우게 만들기
- 변경:
  - stale carry / urgent work / trait pending / best-buy score 관측 추가
  - `stale_carry_cleared` 보상 추가
  - `buy_upgrade` 보상 상향
- 결과: 실패에 가까움
- 해석:
  - stale carry는 거의 학습되지 않음
  - 업그레이드도 실제로 거의 사용되지 않음
  - 이 문제는 RL reward보다 controller split이 더 적합

### exp13_rule_aligned
- 목적: 운영 edge case를 rule-based controller로 분리
- 변경:
  - `ai/controller.py` 추가
  - stale carry는 trash로 강제 이동
  - 업그레이드는 idle window에서만 best-buy heuristic으로 처리
  - trait는 heuristic 자동선택 유지
- 중요:
  - 새 학습 실험이 아님
  - 실시간 플레이 보조 컨트롤러
  - 기본 실행에는 꺼져 있고 `--rule-controller`일 때만 켜짐

### exp14_lategame_curriculum
- 목적: late-game 분포 학습
- 이유:
  - `target_money=150`, `day_limit=8`은 early-game 최적화엔 좋지만
  - 테이블 확장/손님 밀집/후반 운영을 거의 안 보게 만든다
- 설정:
  - `target_money=500`
  - `day_limit=14`
  - `total_timesteps=4000000`
- 상태: 아직 결과 분석 전

## 중요 코드 변경 요약

### RL / 학습 관련
- `ai/train.py`
  - `MaskablePPO` 기반 학습
  - TensorBoard 없을 때도 학습 가능
  - 학습 종료 후 `plots/` 그래프 자동 저장
- `ai/gym_env.py`
  - action mask 노출
  - relative-position observation 추가
  - priority-table observation 추가
  - 이후 stale/ops 신호도 추가됨
- `ai/agent.py`
  - 구버전 모델과 현재 관측 차원이 다를 때 추론용 shape adapter 추가
  - 관측이 더 길면 잘라내고, 짧으면 zero padding

### 게임 로직 관련
- `core/shop.py`
  - action mask 계산
  - stale carry 판정
  - best auto-buy scoring
  - trait heuristic 선택
  - priority-table 실험 이후 편향 완화용 상태 확장
  - exp13용 `should_auto_buy_now()` 추가

### 실행 / 관찰 관련
- `modes/base_mode.py`
  - 배속 기능 추가 (`0.5x`, `1x`, `2x`, `4x`, `8x`)
- `main.py`
  - `--speed`
  - `--rule-controller`
- `modes/ai_mode.py`
  - AI 솔로 관찰 모드
- `modes/versus_mode.py`
  - AI 관찰용 rule controller 연결

### 룰 컨트롤러
- `ai/controller.py`
  - exp13에서 추가
  - stale carry / auto-buy override 판단
- 주의:
  - 기본 실행에서는 사용하지 않음
  - `--rule-controller`를 붙였을 때만 사용

## 중요한 분석 포인트

### 1. 마지막 테이블 편향
- exp09 분석에서 `table_id=3`를 거의 무시하는 편향이 실제로 존재했다.
- 원인 추정:
  - 고정 슬롯 MLP 입력 편향
- 해결:
  - exp10에서 priority table summary 추가
  - exp11에서 장기학습으로 성능 회복

### 2. observation 변경 이후 구모델 호환
- exp07 이후 관측 차원이 여러 번 바뀌었다.
- 그래서 예전 모델은 최신 코드와 shape mismatch가 날 수 있다.
- 현재는 `ai/agent.py`의 adapter 덕분에 플레이 관찰은 가능하다.
- 단, 이 adapter는 “추론 호환”용이다.
- 학습 성능 비교는 항상 같은 observation family 내에서 보는 것이 맞다.

### 3. stale carry / upgrade / trait
- RL로 끝까지 해결하려 했던 시도는 exp12에서 효과가 없었다.
- 현재 판단:
  - stale carry: controller 또는 game rule로 처리
  - upgrade timing: controller 쪽이 더 적합
  - trait choice: 현재는 heuristic이며 RL이 직접 학습하는 구조가 아님

### 4. late-game 멈춤 문제
- 현재 가장 유력한 원인:
  - `target_money=150`, `day_limit=8`에서 early-game 승리만 배우고
  - 테이블 확장 이후 복잡한 운영 상태를 학습 중 거의 보지 못함
- 그래서 exp14는 종료 조건을 없애는 대신 늦춘다.

## 현재 추천 기준

- 현재 기준 모델 채택: `exp11_priority_table_longtrain`
- 현재 플레이 품질 보강 실험: `exp13_rule_aligned`
- 현재 다음 학습 실험: `exp14_lategame_curriculum`

## 실행 시 주의점

- exp11 순수 모델 확인:
```bash
python main.py --mode ai --model models/exp11_priority_table_longtrain/best_model.zip --speed 4
```

- exp13 룰 컨트롤러까지 포함한 확인:
```bash
python main.py --mode ai --model models/exp11_priority_table_longtrain/best_model.zip --speed 4 --rule-controller
```

- 두 실행을 혼동하면 안 된다.
  - 기본 실행은 순수 학습 모델 관찰
  - `--rule-controller`는 운영 보정까지 포함한 플레이 관찰

## 다른 Codex가 이어서 할 일

1. `exp14_lategame_curriculum` 학습 결과 확인
2. 후반부 테이블 확장 이후 멈춤이 줄었는지 관찰
3. 필요하면 late-game 전용 evaluation script 추가
4. stale carry / upgrade 문제는 RL보다 controller 쪽 수정으로 계속 다루는 것이 우선
