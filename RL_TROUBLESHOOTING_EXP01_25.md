# RL Troubleshooting Through Exp25

이 문서는 `exp01`부터 `exp25`까지 진행하며 실제로 “트러블슈팅”이라고 부를 만했던 문제와 수정, 교훈을 별도로 정리한 기록이다. handoff 문서들이 날짜 중심 요약이라면, 이 문서는 문제 유형 중심 참고서에 가깝다.

---

## 1. 가장 큰 흐름 요약

실험 진행 중 반복적으로 나타난 핵심 문제는 아래 다섯 가지였다.

1. 게임 환경이 바뀌었는데, 과거 실험 결과를 그대로 비교하려 한 문제
2. 학습과 실행 환경이 일치하지 않아 “실제로 보면 이상한” 문제가 생긴 점
3. 보상 설계가 실제 운영 목표와 어긋나 학습이 왜곡된 점
4. 액션 설계가 “무엇을 살지”를 학습할 수 없는 구조였던 점
5. 분석 로그가 부족해서 왜 잘 되고 왜 망하는지 분리하기 어려웠던 점

`exp25`까지 오면서 위 다섯 축은 거의 다 한 번씩 터졌고, 그때마다 구조를 조금씩 고쳤다.

---

## 2. 환경 변경을 무시한 비교 문제

### 증상

- `exp11`, `exp14` 같은 기존 결과를 post-merge 환경에서도 그대로 비교하려고 하면 해석이 꼬였다.
- reward가 달라진 것인지
- 게임 경제가 달라진 것인지
- queue / rating / score 시스템이 달라진 것인지
분리가 안 됐다.

### 원인

`main` 머지 이후 게임 자체가 변했다.

- `waiting_queue` 추가
- `shop_rating`, `final_score`, `game_end` 축 강화
- `hire_waiter`, `buy_table` 일부 비용 로직 변경
- stale carry / orphan 정리 이벤트 분화

즉 과거 실험과 현재 실험은 같은 MDP가 아니었다.

### 대응

- `exp15_postmerge_rebaseline`을 추가해 post-merge baseline을 다시 잡았다.
- 새로 들어온 reward 축은 비교 목적상 많이 중립화했다.

### 교훈

- 환경/밸런스가 바뀌면 바로 다음 실험으로 가지 말고 먼저 rebaseline을 잡아야 한다.
- 안 그러면 “성능 향상”처럼 보이는 것이 사실은 환경 변경 효과일 수 있다.

---

## 3. 학습 목표와 실제 게임 목표 불일치

### 증상

- 학습 reward는 오르는데 실제 플레이가 썩 좋아 보이지 않았다.
- 돈은 벌지만 승리로 닫지 못하는 정책
- 운영은 되지만 final score 방향과 어긋나는 정책
- 화면에서 보면 멍청해 보이는데 수치는 괜찮은 정책

### 대표 사례

#### 3.1 exp15 / exp16

- post-merge baseline은 확보됐지만
- `win`을 닫는 힘이 부족했다.

#### 3.2 exp17

- queue / rating / final score 축 일부 복원 후 10일 계열 성능이 크게 개선됐다.
- 이 시점부터 “운영”과 “점수”가 조금 맞아들어가기 시작했다.

### 원인

- reward가 너무 money / immediate service 중심일 때
  - long-run 운영 품질이 빠지기 쉬움
- 반대로 final score / rating을 무리하게 밀면
  - 플레이 체감이 나빠질 수 있음

### 대응

- `exp17`에서 queue-aware 보상과 관측을 부분 복원
- `exp21`에서 30일 운영형 실험으로 목표를 실제 게임에 맞춤

### 교훈

- reward 수치만으로 정책 품질을 판단하면 안 된다.
- “학습 목적함수”, “실제 게임 목표”, “화면 체감”은 분리해서 봐야 한다.

---

## 4. 업그레이드 학습이 불가능했던 액션 설계 문제

### 증상

- 돈이 생기면 테이블만 사는 경향
- 어떤 업그레이드가 장기적으로 좋은지 제대로 못 배움
- `buy_upgrade`는 하는데 “무엇을 샀는지” 학습이 안 됨

### 원인

초기 구조는 사실상 아래와 같았다.

- RL 행동: `ACTION_BUY_UPGRADE`
- 실제 구매 결정: 내부 heuristic auto-buy

즉 RL은 “살지 말지”만 배우고, “무엇을 살지”는 배우지 못했다.

### 대응

#### 4.1 exp18_upgrade_choice_factorized

- 업그레이드 액션을 분리했다.
  - `buy_table`
  - `hire_waiter`
  - `hire_bartender`
  - `kitchen_expand`
  - `hire_chef`

하지만 문제:

- 기존 `ACTION_BUY_UPGRADE`도 남겨 둬서
- 정책이 그 shortcut으로 도망갔다.

#### 4.2 exp19_upgrade_choice_strict

- `disable_auto_buy_action` 옵션을 추가했다.
- strict 환경에서는 `ACTION_BUY_UPGRADE`를 막고, 개별 업그레이드 액션만 쓰게 했다.

결과:

- 이 시점부터 “무엇을 살지” 학습이 실제로 시작됐다.

### 교훈

- 학습시키려는 선택을 heuristic에 숨겨두면 RL은 절대 그 선택을 배우지 못한다.
- factorized action을 도입했다면 shortcut action은 반드시 같이 차단해야 한다.

---

## 5. 관측 부족으로 병목 인식이 안 되던 문제

### 증상

- 새 테이블은 사는데 그 테이블 운영은 못 따라감
- queue pressure가 있는데 waiter / kitchen 대응이 늦음
- 운영 병목과 수용량 병목을 구분 못 함

### 원인

초기 관측에는 아래가 직접적으로 충분히 들어가 있지 않았다.

- `waiting_queue` 압력
- 가장 오래 기다리는 손님 상황
- 주방 backlog / ready food 비율
- 특정 업그레이드 비용 대비 현재 돈 상태

### 대응

#### 5.1 exp17

- queue 관련 관측 추가
  - queue ratio
  - queue full flag
  - oldest waiting patience

#### 5.2 exp25

- 업그레이드 병목 관련 관측 추가
  - occupied / empty table ratio
  - chef ratio
  - waiter level ratio
  - kitchen cooking / ready 비율
  - 각 핵심 업그레이드에 대한 money-to-cost ratio

### 교훈

- 정책이 이상하게 굴 때는 reward보다 먼저 observation에 필요한 상태가 있는지 확인해야 한다.
- 특히 upgrade policy는 “현재 어떤 병목인지”가 안 보이면 제대로 배우기 어렵다.

---

## 6. reward에 있지만 실제로는 발생하지 않던 이벤트 문제

### 증상

- `idle_penalty`, `blocked_move`, `time_penalty`를 reward config에 넣어도
- 플레이어의 멍때림, 막힌 이동, 시간 낭비가 실제로 교정되지 않았다.

### 원인

- reward 계산기에는 항목이 있었지만
- `Shop.step()` 쪽에서 해당 이벤트를 아예 append하지 않고 있었다.

즉 “설정에는 있는데 게임에서는 안 나오는 죽은 reward” 였다.

### 대응

`core/shop.py` 에서 실제 이벤트 발생을 추가했다.

- 모든 step에 `time_penalty`
- 막힌 이동에 `blocked_move`
- `ACTION_NONE`인데 급한 일이 남아 있으면 `idle_penalty`

### 파생 실험

- `exp22`: 강한 responsive penalty
- `exp23`: 약한 responsive penalty

### 결과

- 방향 검증은 됐지만 `exp21`보다 좋지 않았다.
- 특히 `exp22`, `exp23` 모두 후반 안정성이 부족했다.

### 교훈

- 죽어 있는 reward 항목은 빨리 찾아서 살려야 한다.
- 다만 살아 있다고 해서 크게 주면 좋은 건 아니다.

---

## 7. “좋은 패턴을 reward로 직접 밀어넣었다가” 망한 문제

### 증상

- `exp24`에서 고득점 런의 업그레이드 패턴을 찾았고,
- 이를 바탕으로 `exp25`에서 약한 order bias reward를 넣었다.
- 그런데 결과는:
  - best도 `exp21`보다 약했고
  - last는 크게 붕괴했고
  - 일부 런은 `served 0 / final_score 0`

### 원인

- 분석 결과는 맞았지만
- 그 결과를 reward로 바로 꽂아 넣는 순간 정책이 너무 직접적으로 끌려갔다.

정책이 해야 할 일:

- queue / kitchen / waiter 병목을 종합적으로 해석
- 상황별 업그레이드 선택

reward가 강제로 시키려 한 일:

- “이 순서가 좋아 보이니 그쪽으로 더 가라”

이 둘이 충돌하면서 일부 모드가 무너졌다.

### 대응

- `exp25`는 채택 비권장 결론
- 다음 `exp26`은 `exp21` 구조로 회귀
- reward bias 제거
- 대신 trait heuristic 개선 + 평점 diagnostics로 방향 전환

### 교훈

- 분석 결과를 얻었다고 해서 그것을 reward에 직접 넣는 것이 항상 좋은 것은 아니다.
- 특히 long-horizon 환경에서는 “패턴”과 “목적함수”를 분리해서 써야 한다.

---

## 8. trait가 학습 대상이 아니라 heuristic이었다는 점

### 증상

- “어떤 trait가 고득점에 유리한지”를 알고 싶었는데
- 실제로는 trait 선택이 RL이 아니라 heuristic이었다.

### 원인

- `Shop.auto_select_trait()` 가 현재 trait를 자동으로 골랐다.
- RL action space에 trait 선택은 포함되지 않았다.

### 대응

#### 8.1 exp24

- trait 후보 / 선택 로그를 남기게 했다.

#### 8.2 exp24 결과 기반 휴리스틱 개편

- 고득점 런에서 많이 나왔던 패턴을 반영해:
  - `master_chef`
  - `efficient`
  - `patient_service`
  - `gourmet`
  중심으로 휴리스틱을 업데이트했다.

### 교훈

- 학습되지 않는 요소를 학습된 것처럼 해석하면 안 된다.
- 먼저 heuristic/automation인지, 실제 RL action인지 분리해서 봐야 한다.

---

## 9. 평점이 낮게 나오는 구조적 문제

### 증상

- 최종 스코어에 평점 배수가 중요해졌는데
- 실제 별점은 생각보다 낮게 나왔다.
- 심지어 고득점 런도 별 4점대가 잘 안 나왔다.

### 원인

평점 계산 구조가 꽤 보수적이었다.

- 초기 히스토리 20개를 `0.12`로 채움
- 느린 서빙 satisfaction은 `0.2 ~ 0.38`
- angry leave는 `-1.0`
- rolling average 기반

즉 “무난한 운영”만으로는 평점이 쉽게 안 오른다.

### 대응

- `exp26`에서 아래 diagnostics를 추가해 원인 분리 가능하게 했다.
  - `waiting_customers_left`
  - `angry_table_leaves`
  - `avg_served_satisfaction`
  - `fast_service_ratio`
  - `slow_service_ratio`
  - `queue_leave_ratio`
  - `angry_leave_ratio`

### 교훈

- 평점이 낮으면 reward를 더 줄 생각부터 하지 말고,
- queue / angry leave / slow service 중 어디가 실제 원인인지 먼저 분리해야 한다.

---

## 10. 학습과 실행이 완전히 같지 않았던 문제

### 증상

- 학습은 잘 됐다고 나오는데
- 실제 `main.py --mode ai` 로 보면 중간에 멍때리거나 이상한 행동을 보였다.

### 원인

중요한 사례가 두 번 있었다.

#### 10.1 runtime config 미적용

- strict 모델은 학습 때 `disable_auto_buy_action=true`
- 실행 모드에서는 기본 `Shop()` 생성으로 옵션이 빠짐
- 결과적으로 학습 때 없던 action availability가 실전에 열림

#### 10.2 rule-controller 충돌

- strict 모델은 `ACTION_BUY_UPGRADE`를 학습 때 쓰지 않음
- 그런데 `rule-controller`는 업그레이드 필요 시 여전히 `ACTION_BUY_UPGRADE`를 강제로 넣음
- 결과적으로 `--rule-controller`를 켜면 모델이 심하게 망가짐

### 대응

- runtime 모드가 `train_config_used.json`의 `game_overrides`, `env_options`를 읽게 수정
- `ai/controller.py`에서 strict 모델이면 개별 업그레이드 액션으로 매핑하게 수정

### 교훈

- “학습할 때의 action space / env option”과
- “실행할 때의 action space / env option”이 다르면
- 모델 평가는 거의 무의미해진다.

---

## 11. 실시간 로그와 분석 로그의 부재

### 증상

- 학습 중 그래프를 바로 못 보고
- 실험이 끝난 뒤 왜 잘 됐는지/망했는지 해석이 느렸다.

### 대응

- TensorBoard 로그 저장 지원 추가
- `analysis_logs` JSONL 기록 추가
- `exp24`부터 선택 로그 / episode summary를 구조적으로 남김

### 교훈

- 그래프와 진단 로그는 사치가 아니라 필수다.
- 특히 30일 long-horizon에서는 “결과만 보고 감으로 추정”하는 단계가 끝났다.

---

## 12. 렌더링/스프라이트 쪽 트러블슈팅

### 12.1 음식 스프라이트가 안 보인 문제

#### 증상

- `assets/sprites/food/<menu_id>.png`를 넣어도 안 보임

#### 원인

- 파일은 읽혀도, 음식 아이콘을 실제로 그리는 렌더링 경로가 없었다.

#### 대응

- `renderer.py`에:
  - 주방 슬롯 음식 아이콘
  - 플레이어 운반 아이콘
  - 직원 운반 아이콘
  추가
- `asset_manager.py`는 대소문자 정규화 추가

### 12.2 배경 타일 지원을 넣었다가 롤백한 문제

#### 상황

- `assets/tiles` 기반 배경 타일/배경 이미지 지원을 잠깐 추가함

#### 문제

- 실제 의도는 “맵 전체 레스토랑 내부 이미지”였는데,
- 현재 타일 기반 맵 렌더 구조와 정확히 맞는지 요구사항이 계속 바뀜
- 언제든 롤백 가능해야 했음

#### 대응

- 관련 지원 코드를 다시 롤백
- 현재는 다시 기본 색상 기반 맵 렌더링 상태

### 교훈

- 배경은:
  - “타일 이미지 교체”인지
  - “맵 전체 언더레이”인지
  - “정적 배경”인지
  먼저 명확히 해야 한다.

---

## 13. 현재까지의 가장 중요한 교훈

1. shortcut action이 남아 있으면 RL은 진짜 선택을 배우지 않는다.
2. reward에 없는 문제가 아니라, 아예 이벤트가 안 나오는 죽은 reward가 있을 수 있다.
3. 좋은 패턴을 찾았다고 해서 reward로 직접 넣으면 오히려 정책이 망가질 수 있다.
4. heuristic인지 RL action인지 먼저 분리하고 해석해야 한다.
5. long-horizon 실험은 로그 없이 해석하면 안 된다.
6. strict 모델은 runtime action space도 strict여야 한다.
7. 현재 기준 모델은 여전히 `exp21 final` 이고, `exp26`은 그 기준선 위에 진단 능력을 더하는 실험이다.

---

## 14. 현재 기준 추천

- 기준 모델:
  - `models/exp21_strict_upgrade_30day_ops/final_model.zip`
- 기준 해석 태도:
  - `rule-controller` 없이 평가
  - TensorBoard와 `analysis_logs` 같이 보기
  - reward 개선보다 먼저 원인 분해

---

## 15. 다음에 비슷한 문제가 생기면 먼저 볼 것

체크리스트:

1. 현재 실험이 이전 실험과 같은 환경/규칙 위에 있는가?
2. 학습 action space와 실행 action space가 같은가?
3. reward 항목이 실제 이벤트로 발생하는가?
4. heuristic이 선택을 대신하고 있지는 않은가?
5. observation에 필요한 병목 신호가 들어가 있는가?
6. 분석 로그 없이 감으로 해석하고 있지는 않은가?
7. 잘된 패턴을 reward로 너무 직접 밀어넣고 있지는 않은가?

이 문서의 목적은 “무엇이 문제였는지”를 빠르게 다시 찾기 위한 것이다. 실험 자체의 연대기 요약은 `RL_HANDOFF_2026-03-16.md`, `RL_HANDOFF_2026-03-17.md`, `RL_HANDOFF_2026-03-18.md`를 참고하면 된다.
