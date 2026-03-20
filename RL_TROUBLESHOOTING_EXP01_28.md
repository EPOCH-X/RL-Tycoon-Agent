# RL Troubleshooting Through Exp28

이 문서는 `exp01`부터 `exp28`까지 진행하면서 실제로 부딪힌 문제와 해결 과정을
이해하기 쉽게 다시 정리한 트러블슈팅 문서다.

목표는 두 가지다.

1. 처음 보는 사람도 “무슨 문제가 있었고 왜 그런지”를 빠르게 이해할 것
2. 다음에 비슷한 문제가 생겼을 때 바로 체크할 수 있는 참고서를 남길 것

날짜별 진행 요약은 handoff 문서를 보면 되고, 이 문서는 **문제 유형 중심**으로 읽으면 된다.

---

## 1. 전체 흐름 한눈에 보기

우리가 반복해서 부딪힌 큰 문제는 아래 8가지였다.

1. 게임 환경이 바뀌었는데 옛 실험과 그대로 비교한 문제
2. 학습 목표와 실제 게임 목표가 어긋난 문제
3. 업그레이드 학습이 애초에 불가능했던 액션 설계 문제
4. observation이 부족해서 병목을 못 본 문제
5. reward에만 있고 실제 게임에서 안 나오던 이벤트 문제
6. 좋은 패턴을 reward로 직접 강제했다가 망한 문제
7. 학습과 실행(runtime)이 달라서 실제 플레이가 이상했던 문제
8. 로그 부족으로 “왜 잘되고 왜 망하는지”를 분리 못 하던 문제

`exp28`까지 오면서 위 문제들을 거의 한 번씩 해결했고,
현재 기준선은 `exp21`에서 `exp28`로 올라왔다.

---

## 2. 문제 1: 환경이 바뀌었는데 옛 실험과 그대로 비교함

### 증상

- `exp11`, `exp14` 같은 예전 실험이 현재 환경에서도 더 좋아 보이거나
- 반대로 지금 실험이 나빠 보였는데, 사실은 게임 환경 자체가 달라져 있었다.

### 원인

`main` 머지 이후 게임 환경이 바뀌었다.

- `waiting_queue`
- `shop_rating`
- `final_score`
- `game_end`
- 업그레이드 비용/효과 일부

즉 과거 실험과 현재 실험은 같은 환경(MDP)이 아니었다.

### 해결

- `exp15_postmerge_rebaseline`으로 baseline을 다시 잡았다.
- 비교 목적상 새 reward 축을 많이 중립화했다.

### 교훈

- 게임 규칙/경제/점수 체계가 바뀌면, 바로 비교하지 말고 **rebaseline 먼저** 잡아야 한다.

---

## 3. 문제 2: 학습 목표와 실제 게임 목표가 안 맞음

### 증상

- reward는 오르는데 실제 플레이가 멍청해 보임
- 돈은 벌지만 승리로 못 닫음
- 운영은 되는데 high score가 안 나옴

### 원인

초기 reward는 단기 이벤트에 너무 치우쳐 있었다.

- 주문 받기
- 음식 서빙
- 즉시 수익

반면 실제 목표는 점점 아래로 이동했다.

- 30일 운영
- 평점 유지
- final score 극대화

### 해결

- `exp17`에서 queue/rating/final score 축 일부 복원
- `exp21`에서 30일 / 1500 목표로 전환
- `exp26~28`에서는 high score와 평점 원인 진단까지 같이 보기 시작

### 교훈

- “reward가 좋아졌다”와 “게임 목표에 가까워졌다”는 다르다.
- 최종 판단은 항상 아래를 같이 봐야 한다.
  - `mean_reward`
  - `final_score`
  - `rating`
  - `queue/angry leave`

---

## 4. 문제 3: 업그레이드 학습이 애초에 불가능한 구조

### 증상

- 돈만 생기면 테이블만 삼
- waiter, chef, kitchen의 장기 가치를 잘 못 배움
- RL이 “무엇을 사는지”를 전혀 모르는 느낌

### 원인

초기 구조는 사실상 auto-buy였다.

- RL은 `ACTION_BUY_UPGRADE`만 냄
- 실제로 뭘 살지는 `Shop` 내부 heuristic이 결정

즉 RL은 “살까 말까”만 배우고 “무엇을 살까”는 못 배웠다.

### 해결

#### exp18
- 업그레이드 액션을 분리함
  - `buy_table`
  - `hire_waiter`
  - `hire_bartender`
  - `kitchen_expand`
  - `hire_chef`

하지만 실패:
- 기존 auto-buy action이 남아서 shortcut으로 도망감

#### exp19
- `disable_auto_buy_action` 추가
- strict-upgrade 환경에서는 기존 auto-buy action을 차단

이 시점부터 진짜 업그레이드 선택 학습이 시작됐다.

### 교훈

- RL에게 배우게 하고 싶은 선택이 있으면 shortcut heuristic을 같이 차단해야 한다.

---

## 5. 문제 4: observation이 부족해서 병목을 못 봄

### 증상

- 테이블은 늘리는데 운영이 못 따라감
- queue pressure가 있는데 waiter/kitchen 대응이 늦음
- seated customer 흐름이 꼬여도 원인을 못 찾음

### 원인

정책이 아래 상태를 충분히 못 보고 있었다.

- queue 길이 / queue 포화
- 가장 오래 기다리는 손님
- kitchen cooking / ready backlog
- 업그레이드 가능 여부와 비용 대비 현재 돈
- seated customer가 주문 전인지, 주문 후인지, 식사 중인지

### 해결

#### exp17
- queue 관련 observation 추가

#### exp25
- 업그레이드 병목 관련 observation 추가

#### exp27
- seated-flow observation 추가
  - waiting_to_order ratio
  - order_taken ratio
  - eating ratio
  - carrying target 여부
  - seated customer 최저 patience

### 교훈

- 정책이 이상하게 굴면 reward보다 먼저 observation을 봐야 한다.
- “병목이 안 보이는 정책”은 좋은 선택을 할 수 없다.

---

## 6. 문제 5: reward에 이름은 있는데 실제 이벤트가 안 나옴

### 증상

- `idle_penalty`, `blocked_move`, `time_penalty`를 config에 넣어도 행동이 전혀 안 고쳐짐

### 원인

reward 계산기에는 항목이 있었지만, 실제 게임 로직에서 그 이벤트를 발생시키지 않았다.

즉 “설정에만 있는 죽은 reward”였다.

### 해결

`core/shop.py`에 실제 이벤트 계측 추가:

- 모든 step에 `time_penalty`
- 막힌 이동 시 `blocked_move`
- `ACTION_NONE`일 때 `idle_penalty`

### 결과

- reward가 실제로 살아났다
- 다만 이걸 너무 강하게 준 `exp22`, `exp23`은 안정성 면에서 실패

### 교훈

- reward가 안 먹히면 먼저 “이 이벤트가 실제로 발생하나?”부터 확인해야 한다.

---

## 7. 문제 6: 좋은 패턴을 reward로 직접 강제하다가 망함

### 증상

- `exp24`에서 고득점 업그레이드 패턴을 찾았고
- `exp25`에서 그 패턴을 reward bias로 넣었더니
- 일부 런이 아예 붕괴했다

대표 증상:
- `served 0`
- `final_score 0`
- 학습 후반 대붕괴

### 원인

분석 결과는 맞았지만,
그 결과를 reward에 직접 넣자 정책이 너무 특정 순서를 따라가려 했다.

정책이 해야 할 일:
- 현재 상황을 해석
- 병목에 맞는 업그레이드 선택

reward가 시킨 일:
- “이 순서가 좋았으니 이걸 더 따라가라”

이 둘이 충돌했다.

### 해결

- `exp25`는 채택 비권장으로 결론
- `exp26`은 `exp21` 구조로 돌아가되
  - trait heuristic 개선
  - rating diagnostics
로 방향 전환

### 교훈

- 로그에서 찾은 좋은 패턴은 “분석 근거”로는 좋지만, reward로 바로 꽂으면 위험하다.

---

## 8. 문제 7: trait는 RL이 아니라 heuristic이었다

### 증상

- “어떤 trait가 좋은지 학습한다”는 느낌으로 봤지만,
- 실제로는 RL이 trait를 고르는 구조가 아니었다.

### 원인

`Shop.auto_select_trait()`가 자동으로 trait를 골랐다.
즉 trait 선택은 heuristic이지 RL action이 아니었다.

### 해결

#### exp24
- trait offer / pick 로그를 남기게 함

#### exp24 분석 후
- trait heuristic 자체를 업데이트

핵심 우선순위:
- `master_chef`
- `patient_service`
- `efficient`
- `gourmet`

### 교훈

- 학습 대상이 아닌 것을 학습 결과처럼 해석하면 안 된다.
- 먼저 heuristic인지 RL action인지부터 분리해야 한다.

---

## 9. 문제 8: 평점이 왜 낮은지 모름

### 증상

- high score에는 평점이 중요해졌는데
- 별점이 생각보다 낮게 나옴
- “잘한 런”도 별 4점 이상이 쉽게 안 나옴

### 원인

평점 구조가 꽤 보수적이었다.

- 초기 satisfaction history에 낮은 값이 깔려 있음
- 느린 서빙 만족도는 낮음
- angry leave는 `-1.0`

즉 무난한 운영만으로는 별점이 잘 안 오른다.

### 해결

`exp26`에서 평점 원인 진단 지표를 추가했다.

- `queue_leave_ratio`
- `angry_leave_ratio`
- `fast_service_ratio`
- `slow_service_ratio`
- `avg_served_satisfaction`

### 결과

이후 해석이 가능해졌다.

- `exp26`: queue보다 angry leave가 더 큰 병목
- `exp27`: angry leave는 줄었지만 queue leave가 늘어남
- `exp28`: angry leave를 크게 줄이면서 rating과 final score를 동시에 올림

### 교훈

- 평점이 낮으면 reward 숫자를 먼저 건드리지 말고, 원인을 분해해서 봐야 한다.

---

## 10. 문제 9: 학습과 실행(runtime)이 달랐음

### 증상

- 학습은 잘 됐는데 `main.py --mode ai/watch`로 보면 이상하게 움직임
- strict 모델인데 실전에서 auto-buy처럼 동작
- `rule-controller` 켜면 심하게 망가짐

### 원인

#### 10.1 runtime config 미반영
- 학습 때는 `disable_auto_buy_action=true`
- 실행 모드는 기본 `Shop()` 생성
- 결과: action space가 달라짐

#### 10.2 rule-controller 충돌
- strict 모델은 auto-buy action을 안 쓰는데
- rule-controller가 그 action을 강제로 넣음

#### 10.3 watch / ai 모드의 deterministic 차이
- `watch`: 기본 stochastic
- `ai`: 기본 deterministic

같은 모델인데 플레이 패턴이 달라 보였다.

### 해결

- `train_config_used.json`에서 `game_overrides`, `env_options`를 runtime에 복원
- strict 모델일 때 rule-controller가 개별 업그레이드 action으로 매핑되게 수정
- `ai mode`에 `--stochastic` 옵션 추가

### 교훈

- 학습 당시와 실행 당시의 action space / env option / inference mode가 다르면 비교가 무의미하다.

---

## 11. 문제 10: main 브랜치에서 MaskablePPO 모델 실행 실패

### 증상

- `main` 브랜치에서 `watch`로 `exp28 final` 실행 시 즉시 크래시

에러:
- `TypeError: MaskableActorCriticPolicy.__init__() got an unexpected keyword argument 'use_sde'`

### 원인

`MaskablePPO` 모델을 `PPO.load()`로 잘못 읽는 경로가 있었다.

### 해결 방향

`main`에서 최소 필요:

- `MaskablePPO` 로드 지원
- `action_mask` 전달
- `train_config_used.json` 기반 runtime option 복원

### 교훈

- exp21~28 계열 모델은 “PPO zip”처럼 보여도 사실상 `MaskablePPO + strict runtime` 모델이다.

---

## 12. 문제 11: merge 후 충돌과 실행 안정성

### 증상

- `EPOCH-8-Yongwan`에서 `main`을 머지하자 여러 RL 핵심 파일 충돌

충돌 파일:
- `ai/agent.py`
- `ai/gym_env.py`
- `ai/reward.py`
- `config/settings.py`
- `core/shop.py`
- `modes/model_runtime.py`
- `modes/tournament_mode.py`
- `modes/versus_mode.py`
- `modes/watch_mode.py`

### 원인

`main`과 현재 브랜치가
- action space
- runtime option
- watch/versus/tournament 실행 방식
에서 서로 다른 방향으로 진화해 있었다.

### 해결

머지 기준:
- RL 실행 구조를 우선 보존
- `MaskablePPO`, runtime option, action mask, strict-upgrade가 깨지지 않는 쪽 채택

검증:
- 충돌 표식 제거
- `py_compile` 통과
- `exp21`, `exp28` 모델 로드 + mask 예측 스모크 테스트 통과

### 교훈

- RL 실험 브랜치를 main과 합칠 때는 “코드가 합쳐졌냐”보다 “학습 전제가 보존됐냐”가 중요하다.

---

## 13. exp28 상위 20런만 따로 보면 무엇이 보였나

### 성능 경향

상위 20런 평균:
- `final_score 49510.8`
- `served 542.8`
- `lost 8.45`
- `rating 0.81707` (약 `4.09★`)
- `queue_leave_ratio 0.2573`
- `angry_leave_ratio 0.4855`
- `fast_service_ratio 0.8994`

### 해석

- 고득점 런은 거의 항상 빠른 서비스 비율이 매우 높다
- queue 손실을 완전히 없애진 못하지만
- seated customer angry leave를 크게 줄여 점수와 평점을 올린다

### 업그레이드 경향

상위 20런 평균 최종 레벨:
- `hire_chef 5.0`
- `hire_waiter 5.0`
- `kitchen_expand 3.0`
- `buy_table 9.2`
- `hire_bartender 0.65`
- 나머지(`cook_speed`, `marketing`, `speed_shoes`, `employee_speed`)는 사실상 0

첫 구매:
- 상위 20런 모두 `hire_chef`

즉 고득점 패턴은:
- 테이블부터가 아니라 **처리력 먼저**
- `chef -> waiter -> kitchen -> table 확장`
흐름이 강하다

### trait 경향

상위 20런 평균 스택:
- `master_chef 2.15`
- `patient_service 1.65`
- `efficient 1.55`
- `gourmet 1.4`
- `charming 0.25`

즉 핵심은:
- 조리 속도
- 인내심
- 이동 효율

### 교훈

- high score를 노리는 상위 런은 “테이블부터 깔기”보다 “처리력부터 확보”에 가깝다.

---

## 14. 현재까지의 가장 중요한 교훈

1. 환경이 바뀌면 baseline부터 다시 잡아라.
2. shortcut action이 남아 있으면 RL은 진짜 선택을 배우지 못한다.
3. reward가 안 먹히면 먼저 실제 이벤트가 발생하는지 확인하라.
4. 좋은 패턴을 찾았다고 reward로 직접 강제하면 망할 수 있다.
5. heuristic인지 RL action인지 먼저 구분하라.
6. 평점은 숫자보다 원인 지표로 분해해서 봐야 한다.
7. 학습/실행/runtime 옵션/action mask가 다르면 결과 해석이 무의미하다.
8. long-horizon 실험은 로그 없이는 절대 해석하지 말라.
9. 안정화 실험은 대개 “새 보상 추가”보다 “업데이트 강도 완화”가 먼저다.

---

## 15. 현재 기준 추천

현재 기준 모델:
- `models/exp28_seated_flow_stability_tuned/final_model.zip`

이전 비교군:
- `models/exp21_strict_upgrade_30day_ops/final_model.zip`

현재 추천 실행:
```bash
python main.py --mode watch --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4 --stochastic
```

---

## 16. 다음에 비슷한 문제가 생기면 먼저 볼 체크리스트

1. 지금 실험은 예전 실험과 같은 환경인가?
2. 학습과 실행의 action space가 같은가?
3. runtime에서 `train_config_used.json`을 복원하고 있는가?
4. reward에 적은 이벤트가 실제 게임에서 발생하는가?
5. heuristic이 RL이 해야 할 선택을 대신하고 있지는 않은가?
6. observation에 병목 신호가 충분히 들어가 있는가?
7. 해석할 수 있는 로그가 남고 있는가?
8. 최고점만 보고 있지 않고, 안정성/후반 유지력도 보고 있는가?

---

## 17. 함께 보면 좋은 문서

- `RL_HANDOFF_2026-03-18.md`
- `RL_HANDOFF_2026-03-19.md`
- `RL_HANDOFF_2026-03-20.md`
- `EXP21_MAIN_PORTING_GUIDE_2026-03-20.md`

이 문서는 “문제와 교훈” 중심 참고서이고,
handoff 문서들은 날짜별 작업 기록이라고 보면 된다.
