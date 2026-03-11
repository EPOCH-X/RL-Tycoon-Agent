# DEV_README — 개발자·디자이너 전용 레퍼런스

> 이 문서는 **디자인팀**이 스프라이트를 만들거나, **개발팀**이 코드를 수정할 때 참고하는 용어 사전 + 기술 레퍼런스입니다.
> 코드에 등장하는 **영어 단어(변수명, 상태명, 파일명)** 가 각각 무엇을 뜻하는지 상세히 설명합니다.

---

## 목차

1. [엔티티(Entity) 용어 사전](#1-엔티티entity-용어-사전)
2. [상태(State) 용어 사전](#2-상태state-용어-사전)
3. [액션(Action) 용어 사전](#3-액션action-용어-사전)
4. [JSON 설정 필드 사전](#4-json-설정-필드-사전)
5. [색상 키(Color Key) 매핑](#5-색상-키color-key-매핑)
6. [스프라이트 파일명 ↔ 코드 매핑](#6-스프라이트-파일명--코드-매핑)
7. [그리드·좌표 체계](#7-그리드좌표-체계)
8. [렌더러(Renderer) 그리기 순서](#8-렌더러renderer-그리기-순서)
9. [게임 루프 구조](#9-게임-루프-구조)
10. [강화학습(RL) 용어 사전](#10-강화학습rl-용어-사전)
11. [보상(Reward) 함수 상세](#11-보상reward-함수-상세)
12. [관측(Observation) 공간 상세](#12-관측observation-공간-상세)
13. [config/train_config.json 필드 상세](#13-configtrain_configjson-필드-상세)
14. [Shop 핵심 메서드 사전](#14-shop-핵심-메서드-사전)
15. [파일별 역할 한줄 요약](#15-파일별-역할-한줄-요약)

---

## 1. 엔티티(Entity) 용어 사전

게임에 등장하는 모든 "존재(Entity)"의 영어명과 뜻:

| 영어           | 한글            | 설명                                                      |
| -------------- | --------------- | --------------------------------------------------------- |
| **Player**     | 플레이어        | 사용자가 조작하는 서버(웨이터) 캐릭터                     |
| **Customer**   | 손님            | 테이블에 앉아 음식을 주문하는 NPC                         |
| **Employee**   | 종업원          | 자동으로 서빙하는 AI NPC (고용 업그레이드 필요)           |
| **Table**      | 테이블          | 손님이 앉는 장소. 그리드 1칸 = 테이블 1개                 |
| **Kitchen**    | 주방            | 조리를 담당하는 시설. 주방 카운터에서 주문 전달/음식 수거 |
| **BarStation** | 바(음료 카운터) | 음료를 제조하는 시설. 바 카운터에서 음료 수거             |
| **Shop**       | 매장            | 레스토랑 전체를 나타내는 최상위 객체 (모든 엔티티 포함)   |
| **Entity**     | 엔티티          | 모든 게임 오브젝트의 기본 클래스 (좌표, 스프라이트)       |

### 운반 아이템 타입 (Player.carrying 리스트 내 dict)

| 타입      | 영어  | 한글   | 설명                                                |
| --------- | ----- | ------ | --------------------------------------------------- |
| `"order"` | Order | 주문서 | 손님의 주문을 들고 있는 상태 (→ 주방에 전달)        |
| `"food"`  | Food  | 음식   | 조리 완료된 음식을 들고 있는 상태 (→ 테이블에 서빙) |
| `"drink"` | Drink | 음료   | 제조 완료된 음료를 들고 있는 상태 (→ 테이블에 서빙) |

---

## 2. 상태(State) 용어 사전

### Customer 상태 (customer.state)

| 영어                 | 한글        | 설명                                   | 화면 표시         |
| -------------------- | ----------- | -------------------------------------- | ----------------- |
| `"waiting_to_order"` | 주문 대기   | 테이블에 앉았지만 아직 주문 안 받음    | `?!` 아이콘       |
| `"order_taken"`      | 주문 접수됨 | 주문을 받았고, 음식이 오길 기다리는 중 | 메뉴 이름 표시    |
| `"eating"`           | 식사 중     | 음식·음료를 모두 받고 먹는 중 (3초)    | `eating` 텍스트   |
| `"leaving_happy"`    | 만족 퇴장   | 식사 후 결제하고 나가는 중             | 초록 텍스트       |
| `"leaving_angry"`    | 화남 퇴장   | 인내심이 0이 되어 돈 안 내고 나감      | 빨간색, 벌금 -$30 |

### Employee 상태 (employee.state)

| 영어       | 한글    | 설명                                     |
| ---------- | ------- | ---------------------------------------- |
| `"idle"`   | 대기    | 할 일이 없어 대기 중                     |
| `"moving"` | 이동 중 | 목표 좌표로 이동 중                      |
| `"acting"` | 행동 중 | 목표 도착 후 동작 실행 중 (0.8초 타이머) |

### Employee 작업 타입 (employee.task)

| 영어               | 한글      | 설명                                   |
| ------------------ | --------- | -------------------------------------- |
| `"take_order"`     | 주문 받기 | 테이블로 이동 → 손님 주문 접수         |
| `"submit_kitchen"` | 주방 전달 | 주방으로 이동 → 주문서 전달, 조리 시작 |
| `"pickup_food"`    | 음식 수거 | 주방으로 이동 → 완성된 음식 픽업       |
| `"pickup_drink"`   | 음료 수거 | 바로 이동 → 완성된 음료 픽업           |
| `"serve"`          | 서빙      | 테이블로 이동 → 음식/음료 전달         |

### Kitchen 상태 (조리 큐)

| 영어             | 한글        | 설명                                  |
| ---------------- | ----------- | ------------------------------------- |
| `cooking`        | 조리 중     | 현재 조리되고 있는 요리 리스트        |
| `ready`          | 완성 (대기) | 조리 완료, 서버가 수거하길 대기       |
| `delivery_ready` | 배달 완성   | 배달 주문 조리 완료, 배달 타이머 시작 |
| `capacity`       | 용량        | 동시 조리 가능한 최대 수              |
| `can_accept`     | 접수 가능   | 조리 중 < 용량이면 True               |

### BarStation 상태 (음료 큐)

| 영어        | 한글        | 설명                              |
| ----------- | ----------- | --------------------------------- |
| `preparing` | 제조 중     | 현재 제조되고 있는 음료 리스트    |
| `ready`     | 완성 (대기) | 제조 완료, 수거 대기              |
| `capacity`  | 용량        | 동시 제조 가능한 최대 수 (기본 2) |

### Table 상태

| 영어          | 한글   | 설명                                     |
| ------------- | ------ | ---------------------------------------- |
| `is_occupied` | 점유됨 | 손님이 앉아 있으면 True                  |
| `customer`    | 손님   | 현재 앉아 있는 Customer 객체 (또는 None) |

---

## 3. 액션(Action) 용어 사전

게임에서 사용되는 액션 코드 (settings.py 정의):

| 상수명               | 값  | 한글            | RL 에이전트 | 인간 플레이어 |
| -------------------- | --- | --------------- | ----------- | ------------- |
| `ACTION_UP`          | 0   | 위로 이동       | action=0    | ↑ / W         |
| `ACTION_DOWN`        | 1   | 아래로 이동     | action=1    | ↓ / S         |
| `ACTION_LEFT`        | 2   | 왼쪽 이동       | action=2    | ← / A         |
| `ACTION_RIGHT`       | 3   | 오른쪽 이동     | action=3    | → / D         |
| `ACTION_INTERACT`    | 4   | 상호작용        | action=4    | Space / Enter |
| `ACTION_NONE`        | 5   | 대기            | action=5    | (키 안 누름)  |
| `ACTION_BUY_UPGRADE` | 6   | 업그레이드 구매 | action=6    | U → 1~9       |

---

## 4. JSON 설정 필드 사전

### menu.json 필드

| 필드            | 타입   | 한글      | 설명                                      |
| --------------- | ------ | --------- | ----------------------------------------- |
| `id`            | string | 식별자    | 코드에서 사용하는 고유 키 (예: `"pasta"`) |
| `name`          | string | 표시명    | 화면에 보이는 이름 ("Pasta")              |
| `cook_time`     | float  | 조리 시간 | 초 단위. 주방에서 얼마나 걸리는지         |
| `price`         | int    | 판매가    | 기본 판매 가격 ($)                        |
| `unlock_profit` | int    | 해금 조건 | 이 순이익 이상이면 메뉴 탭에 표시         |
| `unlock_cost`   | int    | 해금 비용 | 이 금액을 내야 메뉴 활성화                |

### customers.json 필드

| 필드            | 타입       | 한글        | 설명                                   |
| --------------- | ---------- | ----------- | -------------------------------------- |
| `id`            | string     | 식별자      | 내부 키 (예: `"vip"`)                  |
| `name`          | string     | 표시명      | 화면 표시 ("VIP")                      |
| `patience`      | float      | 인내심      | 초 단위. 이 시간 내에 서빙 못하면 화남 |
| `wealth_mult`   | float      | 재산 배율   | 음식 가격에 곱해지는 계수              |
| `tip_range`     | [int, int] | 팁 범위     | [최소, 최대] 범위에서 랜덤             |
| `spawn_weight`  | int        | 등장 가중치 | 다른 유형 대비 등장 확률 비중          |
| `color_key`     | string     | 색상 키     | settings.py COLORS 딕셔너리의 키       |
| `group_size`    | [int, int] | 그룹 크기   | [최소, 최대] 인원수 (가족은 2~4)       |
| `unlock_rating` | float      | 해금 평점   | 매장 평점이 이 값 이상이어야 등장      |

### upgrades.json 필드

| 필드              | 타입   | 한글      | 설명                                  |
| ----------------- | ------ | --------- | ------------------------------------- |
| `id`              | string | 식별자    | 내부 키 (예: `"hire_waiter"`)         |
| `name`            | string | 표시명    | 한글 표시 ("종업원 고용")             |
| `category`        | string | 카테고리  | 업그레이드 탭 분류 (아래 참조)        |
| `description`     | string | 설명      | 효과 한줄 설명                        |
| `base_cost`       | int    | 기본 비용 | 레벨 0→1 구매 시 비용                 |
| `cost_multiplier` | float  | 비용 배율 | 레벨 올릴 때마다 비용 × 이 값         |
| `effect_type`     | string | 효과 타입 | 코드에서 분기하는 효과 ID (아래 참조) |
| `effect_value`    | number | 효과 값   | 효과의 수치                           |
| `max_level`       | int    | 최대 레벨 | 최대 구매 횟수                        |
| `unlock_profit`   | int    | 해금 조건 | 순이익이 이 값 이상이어야 표시        |

### category (업그레이드 카테고리)

| 값           | 한글 | 탭      |
| ------------ | ---- | ------- |
| `"facility"` | 시설 | 시설 탭 |
| `"staff"`    | 인력 | 인력 탭 |
| `"personal"` | 개인 | 메뉴 탭 |
| `"business"` | 사업 | 메뉴 탭 |

### effect_type (업그레이드 효과 타입)

| 값                   | 한글          | 효과                                |
| -------------------- | ------------- | ----------------------------------- |
| `"player_speed"`     | 이동속도      | 플레이어 속도 +effect_value%        |
| `"kitchen_capacity"` | 주방 용량     | 동시 조리 수 +effect_value          |
| `"buy_table"`        | 테이블 구매   | 새 테이블 1개 활성화                |
| `"wealthy_bonus"`    | 부유 보너스   | 부유한 손님 등장 확률 +effect_value |
| `"hire_waiter"`      | 종업원 고용   | AI 종업원 1명 추가                  |
| `"hire_bartender"`   | 바텐더 고용   | 음료 서비스 활성화                  |
| `"hire_delivery"`    | 배달기사 고용 | 배달 서비스 활성화                  |

### beverages.json 필드

| 필드                    | 타입   | 한글           | 설명                         |
| ----------------------- | ------ | -------------- | ---------------------------- |
| `unlock_profit`         | int    | 전체 해금 조건 | 바텐더 고용 가능 최소 순이익 |
| `items[].id`            | string | 식별자         | 음료 고유 키                 |
| `items[].name`          | string | 표시명         | 화면 표시 이름               |
| `items[].prep_time`     | float  | 제조 시간      | 바에서 만드는 데 걸리는 초   |
| `items[].price`         | int    | 판매가         | 음료 판매 가격 ($)           |
| `items[].unlock_profit` | int    | 개별 해금      | 이 순이익 이상이면 등장      |

### traits.json 필드

| 필드                   | 타입   | 한글      | 설명                             |
| ---------------------- | ------ | --------- | -------------------------------- |
| `offer_interval_days`  | int    | 제안 주기 | 몇 일마다 특성 선택 이벤트       |
| `choices_per_offer`    | int    | 선택지 수 | 한 번에 보여주는 선택 수         |
| `traits[].id`          | string | 식별자    | 특성 고유 키                     |
| `traits[].name`        | string | 한글명    | 화면 표시 이름                   |
| `traits[].description` | string | 설명      | 효과 한줄 설명                   |
| `traits[].effect`      | string | 효과 유형 | 코드 분기용 (아래 참조)          |
| `traits[].value`       | number | 효과 값   | 수치                             |
| `traits[].max_stacks`  | int    | 최대 중첩 | 같은 특성을 최대 몇 번 선택 가능 |

### trait effect (특성 효과 유형)

| 값                      | 한글             | 효과                           |
| ----------------------- | ---------------- | ------------------------------ |
| `"food_price_bonus"`    | 음식 가격 보너스 | 모든 음식 가격 +value ($)      |
| `"cook_time_reduction"` | 조리시간 감소    | 모든 요리 조리시간 -value (초) |
| `"carry_capacity"`      | 운반 용량        | 동시 운반 가능 수 +value       |
| `"tip_bonus"`           | 팁 보너스        | 전체 팁 +value (비율)          |
| `"speed_bonus"`         | 이동속도 보너스  | 플레이어 속도 +value (비율)    |
| `"spawn_rate"`          | 손님 방문률      | 손님 등장 빈도 +value (비율)   |
| `"patience_bonus"`      | 인내심 보너스    | 모든 손님 인내심 +value (초)   |
| `"base_tip"`            | 기본 팁          | 기본 팁 +value ($)             |

### delivery.json 필드

| 필드               | 타입       | 한글      | 설명                                 |
| ------------------ | ---------- | --------- | ------------------------------------ |
| `unlock_profit`    | int        | 해금 조건 | 배달 서비스 활성화 최소 순이익       |
| `order_interval`   | float      | 주문 간격 | 배달 주문이 들어오는 간격 (초)       |
| `delivery_time`    | float      | 배달 시간 | 조리 완료 후 배달 완료까지 (초)      |
| `price_multiplier` | float      | 가격 배율 | 원래 메뉴 가격 × 이 값 (수수료 차감) |
| `tip_range`        | [int, int] | 팁 범위   | 배달 팁 [최소, 최대]                 |

### map_default.json 필드

| 필드                 | 타입     | 한글             | 설명                              |
| -------------------- | -------- | ---------------- | --------------------------------- |
| `name`               | string   | 맵 이름          | 표시용                            |
| `width`              | int      | 너비             | 그리드 가로 칸 수                 |
| `height`             | int      | 높이             | 그리드 세로 칸 수                 |
| `layout`             | int[][]  | 타일 배열        | 2D 배열, 각 값은 타일 코드        |
| `tile_legend`        | object   | 타일 범례        | 숫자→타일명 매핑                  |
| `tables`             | object[] | 초기 테이블      | 게임 시작 시 활성 테이블 좌표     |
| `kitchen_counters`   | object[] | 주방 위치        | 주방 카운터 그리드 좌표           |
| `bar_counters`       | object[] | 바 위치          | 바 카운터 그리드 좌표             |
| `player_start`       | object   | 시작 위치        | 플레이어 초기 그리드 좌표         |
| `purchasable_tables` | object[] | 구매 가능 테이블 | 업그레이드로 활성화 가능한 테이블 |

### 타일 코드 (layout 배열 값)

| 코드 | 영어              | 한글        | 설명                |
| ---- | ----------------- | ----------- | ------------------- |
| 0    | `floor`           | 바닥        | 이동 가능 영역      |
| 1    | `wall`            | 벽          | 이동 불가           |
| 2    | `table`           | 테이블      | 손님 좌석           |
| 3    | `kitchen_counter` | 주방 카운터 | 주문·음식 수거 위치 |
| 4    | `bar_counter`     | 바 카운터   | 음료 수거 위치      |

---

## 5. 색상 키(Color Key) 매핑

코드에서 스프라이트가 없을 때 사용하는 도형 색상입니다.
디자인팀은 이 색상을 참고하여 각 엔티티의 시각적 정체성을 유지하세요.

| 색상 키 (settings.py) | RGB             | 대상               | 설명           |
| --------------------- | --------------- | ------------------ | -------------- |
| `"background"`        | (40, 40, 40)    | 배경               | 화면 바깥      |
| `"floor"`             | (200, 200, 180) | 바닥 타일          | 이동 가능 영역 |
| `"wall"`              | (80, 80, 80)    | 벽 타일            | 이동 불가      |
| `"grid_line"`         | (60, 60, 60)    | 격자선             | 타일 경계      |
| `"player"`            | (50, 120, 220)  | 플레이어 (빈손)    | 파란색         |
| `"player_carry"`      | (80, 160, 255)  | 플레이어 (운반 중) | 밝은 파란      |
| `"customer"`          | (220, 180, 50)  | 일반 손님          | 노란색         |
| `"customer_angry"`    | (220, 80, 50)   | 화난 손님          | 빨간-주황      |
| `"customer_wealthy"`  | (180, 120, 220) | 부유한 손님        | 보라           |
| `"customer_vip"`      | (255, 215, 0)   | VIP 손님           | 금색           |
| `"customer_family"`   | (180, 200, 100) | 가족 손님          | 연두           |
| `"customer_tourist"`  | (100, 180, 220) | 관광객 손님        | 하늘색         |
| `"customer_critic"`   | (220, 50, 50)   | 평론가 손님        | 진한 빨강      |
| `"table"`             | (139, 90, 43)   | 빈 테이블          | 나무색         |
| `"table_occupied"`    | (160, 110, 60)  | 점유 테이블        | 밝은 나무색    |
| `"kitchen"`           | (180, 60, 60)   | 주방 (빈)          | 빨간 계열      |
| `"kitchen_cooking"`   | (220, 100, 50)  | 주방 (조리 중)     | 주황           |
| `"kitchen_ready"`     | (80, 220, 80)   | 주방 (완성)        | 초록           |
| `"bar"`               | (100, 60, 140)  | 바 (빈)            | 보라 계열      |
| `"bar_ready"`         | (160, 100, 220) | 바 (음료 완성)     | 밝은 보라      |
| `"employee"`          | (80, 200, 160)  | 종업원 (빈손)      | 민트색         |
| `"employee_carry"`    | (120, 240, 190) | 종업원 (운반 중)   | 밝은 민트      |
| `"delivery"`          | (200, 140, 60)  | 배달               | 주황-갈색      |
| `"text"`              | (255, 255, 255) | 텍스트             | 흰색           |
| `"ui_bg"`             | (30, 30, 50)    | UI 배경            | 진한 남색      |
| `"money"`             | (255, 215, 0)   | 돈 표시            | 금색           |
| `"satisfaction"`      | (100, 220, 100) | 만족도 표시        | 초록           |
| `"timer"`             | (200, 200, 200) | 타이머 표시        | 밝은 회색      |

---

## 6. 스프라이트 파일명 ↔ 코드 매핑

`AssetManager`가 `assets/sprites/<entity>/<state>.png` 경로를 자동 탐색합니다.
아래 표의 **엔티티명**과 **상태명**이 폴더/파일명이 됩니다.

### Player 스프라이트

| 파일명                  | 코드에서 조회 시점        | 설명                  |
| ----------------------- | ------------------------- | --------------------- |
| `player/idle.png`       | 키 입력 없을 때           | 대기 포즈             |
| `player/move_up.png`    | ↑/W 이동 중               | 위쪽 걷기             |
| `player/move_down.png`  | ↓/S 이동 중               | 아래쪽 걷기           |
| `player/move_left.png`  | ←/A 이동 중               | 왼쪽 걷기             |
| `player/move_right.png` | →/D 이동 중               | 오른쪽 걷기           |
| `player/carry.png`      | carrying 비어있지 않을 때 | 무언가 들고 있는 상태 |

### Customer 스프라이트

| 파일명                                        | 코드에서 조회 시점       | 설명                |
| --------------------------------------------- | ------------------------ | ------------------- |
| `customer/idle.png`                           | 기본 상태                | 앉아 있는 일반 손님 |
| `customer/angry.png`                          | state == "leaving_angry" | 화난 표정           |
| `customer/eating.png`                         | state == "eating"        | 먹는 모션           |
| `customer/budget.png` ~ `customer/critic.png` | 유형별 (선택)            | 손님 유형별 외형    |

### Employee 스프라이트

| 파일명                | 코드에서 조회 시점   | 설명      |
| --------------------- | -------------------- | --------- |
| `employee/idle.png`   | state == "idle"      | 대기 포즈 |
| `employee/moving.png` | state == "moving"    | 이동 모션 |
| `employee/carry.png`  | carrying is not None | 운반 중   |

### 시설 스프라이트

| 파일명                | 조회 시점            | 설명                |
| --------------------- | -------------------- | ------------------- |
| `table/empty.png`     | is_occupied == False | 빈 테이블           |
| `table/occupied.png`  | is_occupied == True  | 손님 착석 테이블    |
| `kitchen/idle.png`    | 대기                 | 주방 빈 상태        |
| `kitchen/cooking.png` | cooking > 0          | 조리 중 (불꽃 등)   |
| `kitchen/ready.png`   | ready > 0            | 음식 완성 (초록 빛) |
| `bar/idle.png`        | 대기                 | 바 빈 상태          |
| `bar/ready.png`       | ready > 0            | 음료 완성           |

### 음식 아이콘 스프라이트

| 파일명                    | 설명               |
| ------------------------- | ------------------ |
| `food/coffee.png`         | 커피 아이콘        |
| `food/sandwich.png`       | 샌드위치 아이콘    |
| `food/pasta.png`          | 파스타 아이콘      |
| `food/steak.png`          | 스테이크 아이콘    |
| `food/sushi_set.png`      | 초밥 세트 아이콘   |
| `food/lobster.png`        | 랍스터 아이콘      |
| `food/wagyu.png`          | 와규 비프 아이콘   |
| `food/truffle_course.png` | 트러플 코스 아이콘 |

### 스프라이트 시트 규격

- **1프레임 크기**: 64×64 px (= TILE_SIZE)
- **시트 형식**: 프레임들을 가로로 나열한 수평 스프라이트 시트
- **프레임 수** = 이미지 너비 ÷ 64
- **예시**: 걷기 4프레임 → 256×64 px PNG
- 프레임이 1개면 정적 이미지 (64×64)

---

## 7. 그리드·좌표 체계

### 그리드 좌표 (Grid)

- **원점**: 좌상단 (0, 0)
- **X축**: 오른쪽으로 증가, range: 0 ~ 11 (width=12)
- **Y축**: 아래쪽으로 증가, range: 0 ~ 9 (height=10)
- **맵 구조**:
  ```
  Y=0  [벽 벽 벽 벽 벽 벽 벽 벽 벽 벽 벽 벽]
  Y=1  [벽 .  T  .  T  .  T  .  T  .  .  벽]  ← 초기 테이블 4개
  Y=2  [벽 .  .  .  .  .  .  .  .  .  .  벽]
  Y=3  [벽 .  (T) . (T) . (T) . (T) .  .  벽]  ← 구매 가능 테이블 4개
  Y=4  [벽 .  .  .  .  .  .  .  .  .  .  벽]
  Y=5  [벽 .  .  .  .  P  .  .  .  .  .  벽]  ← 플레이어 시작
  Y=6  [벽 .  .  .  .  .  .  .  .  .  .  벽]
  Y=7  [벽 .  .  .  .  .  .  .  .  .  .  벽]
  Y=8  [벽 .  K  K  K  .  .  B  B  .  .  벽]  ← 주방(K) + 바(B)
  Y=9  [벽 벽 벽 벽 벽 벽 벽 벽 벽 벽 벽 벽]
       X=0  1  2  3  4  5  6  7  8  9  10 11
  ```

### 픽셀 좌표 (Pixel)

- `pixel_x = grid_x × TILE_SIZE` (TILE_SIZE = 64)
- `pixel_y = grid_y × TILE_SIZE`
- **엔티티 중심**: `center_x = pixel_x + TILE_SIZE / 2`, `center_y = pixel_y + TILE_SIZE / 2`
- **화면 크기**: 768 × 640 (게임) + 120 (UI) = 768 × 760 (총)
- **Versus**: 좌우 분할 → 각 768×760, 중앙 4px 구분선

### 상호작용 범위

- `INTERACT_RANGE = 80 px` ≈ 1.25 타일
- 플레이어/종업원이 대상 중심에서 80px 이내에 있어야 상호작용 가능

---

## 8. 렌더러(Renderer) 그리기 순서

`renderer.py`의 `draw()` 메서드가 아래 순서로 그립니다 (뒤에 그린 것이 위에 표시):

| 순서 | 메서드                     | 그리는 것                                           |
| ---- | -------------------------- | --------------------------------------------------- |
| 1    | `_draw_map`                | 바닥·벽 타일 (배경)                                 |
| 2    | `_draw_purchasable_tables` | 구매 가능 테이블 (고스트)                           |
| 3    | `_draw_tables`             | 활성 테이블 (나무색/점유색)                         |
| 4    | `_draw_kitchen`            | 주방 카운터 + 조리 진행 바 + 레시피 이름            |
| 5    | `_draw_bar`                | 바 카운터 + 제조 상태                               |
| 6    | `_draw_customers`          | 손님 (상태 아이콘, 그룹 뱃지, 음료 표시, 인내심 바) |
| 7    | `_draw_employees`          | 종업원 (원형 + ID 번호)                             |
| 8    | player.render              | 플레이어 (사각형 + 방향 삼각형 + 운반 표시)         |
| 9    | `_draw_ui`                 | 하단 UI (돈, 일수, 타이머, 평점, 운반 상태, 통계)   |
| 10   | `_draw_upgrade_panel`      | (열려 있을 때) 업그레이드 메뉴 오버레이             |
| 11   | `_draw_trait_popup`        | (활성화 시) 특성 선택 팝업                          |

---

## 9. 게임 루프 구조

```
┌─────── Pygame Main Loop (60 FPS) ───────┐
│                                          │
│  handle_events()  ← 키보드/마우스 입력   │
│                                          │
│  tick(dt)         ← 매 프레임 (1/60초)   │
│   └─ 연속 이동 (WASD/화살표)            │
│   └─ 충돌 감지                          │
│                                          │
│  if accumulator >= STEP_INTERVAL (0.2s): │
│    update()       ← 고정 타임스텝        │
│     └─ 상호작용 처리                    │
│     └─ 손님 스폰                        │
│     └─ 조리 타이머                      │
│     └─ 종업원 AI                        │
│     └─ 배달 타이머                      │
│     └─ 특성 체크                        │
│     └─ 게임 종료 체크                   │
│                                          │
│  render()         ← 프레임마다 화면 갱신 │
└──────────────────────────────────────────┘
```

### 핵심: 이동과 로직이 분리되어 있음

- **tick(dt)**: 매 프레임 실행. 부드러운 연속 이동만 처리
- **update()**: 0.2초마다 실행. 게임 로직 (주문, 조리, 만족도 등) 처리
- RL 에이전트는 `shop.step(action)` 호출 → 내부적으로 이동+로직 한방에 처리

---

## 10. 강화학습(RL) 용어 사전

RL 관련 코드·설정에서 사용되는 영어 용어:

| 영어                | 한글                 | 설명                                                     |
| ------------------- | -------------------- | -------------------------------------------------------- |
| **PPO**             | 근위 정책 최적화     | Proximal Policy Optimization, 현재 사용 중인 RL 알고리즘 |
| **MlpPolicy**       | 다층 퍼셉트론 정책   | 완전 연결 신경망으로 된 정책 네트워크                    |
| **observation**     | 관측                 | 에이전트가 환경에서 보는 정보 (49차원 벡터)              |
| **action**          | 행동                 | 에이전트가 선택하는 행동 (7개 중 택1)                    |
| **reward**          | 보상                 | 행동의 결과로 받는 점수 (돈, 팁, 벌점 등)                |
| **episode**         | 에피소드             | 게임 1회 플레이 (시작→게임오버)                          |
| **timestep**        | 타임스텝             | 에이전트가 관측→행동→보상을 한 번 수행                   |
| **total_timesteps** | 총 학습 스텝         | 전체 학습에서 수행할 총 타임스텝 수                      |
| **n_envs**          | 병렬 환경 수         | 동시에 실행하는 게임 인스턴스 수 (학습 속도↑)            |
| **eval_freq**       | 평가 주기            | 몇 스텝마다 모델을 평가하는지                            |
| **learning_rate**   | 학습률               | 신경망 가중치 업데이트 크기                              |
| **n_steps**         | 롤아웃 길이          | 경험 수집 후 한번에 업데이트하는 스텝 수                 |
| **batch_size**      | 미니배치 크기        | 업데이트 시 한 번에 처리하는 경험 수                     |
| **n_epochs**        | 에폭 수              | 수집한 데이터를 몇 번 재사용하여 학습                    |
| **gamma**           | 할인율               | 미래 보상의 현재 가치 할인 (0.99 = 미래 중시)            |
| **gae_lambda**      | GAE 람다             | Generalized Advantage Estimation 파라미터                |
| **clip_range**      | 클리핑 범위          | PPO 정책 업데이트 제한 범위                              |
| **ent_coef**        | 엔트로피 계수        | 탐험(exploration) 장려 정도 (높으면 다양한 행동)         |
| **vf_coef**         | 가치함수 계수        | 가치(Value) 네트워크 손실 가중치                         |
| **max_grad_norm**   | 그래디언트 최대 노름 | 그래디언트 크기 제한 (학습 안정성)                       |
| **net_arch**        | 네트워크 구조        | 히든 레이어 크기 배열 (예: [128, 128])                   |
| **activation_fn**   | 활성화 함수          | 뉴런 출력 함수 (tanh, relu, elu)                         |
| **tensorboard**     | 텐서보드             | 학습 곡선 시각화 도구                                    |
| **seed**            | 랜덤 시드            | 재현성을 위한 난수 시드                                  |
| **reward_shaping**  | 보상 설계            | 특정 행동에 대한 보상 가중치 커스터마이징                |
| **game_overrides**  | 게임 오버라이드      | 학습 시 목표금액/일수제한 변경                           |

---

## 11. 보상(Reward) 함수 상세

`shop.step(action)` 반환값인 reward의 구성:

### 양의 보상 (Positive Rewards)

| 이벤트          | 보상값   | 발생 조건                         |
| --------------- | -------- | --------------------------------- |
| 손님 결제       | +payment | 식사 완료 → 결제 (음식+팁+음료)   |
| 배달 완료       | +payment | 배달 타이머 만료 (가격×0.85 + 팁) |
| 업그레이드 구매 | +3.0     | action=6일 때 구매 성공           |
| 목표 달성       | +200.0   | money ≥ target_money              |

### 음의 보상 (Negative Rewards)

| 이벤트          | 보상값 | 발생 조건                           |
| --------------- | ------ | ----------------------------------- |
| 손님 이탈       | -30.0  | 인내심 0 → 화남 퇴장                |
| 업그레이드 실패 | -0.1   | action=6인데 돈 부족 또는 이미 최대 |

### 보상 흐름 예시

```
1. 에이전트 action=4 (INTERACT) → 테이블 앞에서 주문 받기 → reward = 0
2. 에이전트 action=4 → 주방에 전달 → reward = 0
3. (조리 대기 중... action=5 반복)
4. 에이전트 action=4 → 음식 수거 → reward = 0
5. 에이전트 action=4 → 서빙 → 손님 식사 시작
6. (식사 3초 후) 손님 결제 → reward = +$45 (파스타)
7. 만약 중간에 다른 손님 화남 → reward -= $30
```

---

## 12. 관측(Observation) 공간 상세

`TycoonEnv`의 observation은 **49개 연속값** (모두 0.0~1.0 정규화):

### 인덱스별 상세

| 인덱스 | 이름             | 정규화                      | 의미                           |
| ------ | ---------------- | --------------------------- | ------------------------------ |
| 0      | player_x         | x / (width × 64)            | 플레이어 X 위치                |
| 1      | player_y         | y / (height × 64)           | 플레이어 Y 위치                |
| 2      | player_facing    | facing / 3.0                | 바라보는 방향 (0~3)            |
| 3      | carry_type       | 0/0.33/0.66/1.0             | 빈손/주문/음식/음료            |
| 4      | carry_table_id   | table_id / max_tables       | 운반 중인 아이템의 목적 테이블 |
| 5      | carry_menu_id    | MENU_IDS[id] / 9            | 운반 중인 아이템의 메뉴 종류   |
| 6~9    | table_0          | 아래 참조                   | 테이블 0번 상태                |
| 10~13  | table_1          |                             | 테이블 1번 상태                |
| 14~17  | table_2          |                             | 테이블 2번 상태                |
| 18~21  | table_3          |                             | 테이블 3번 상태                |
| 22~25  | table_4          |                             | 테이블 4번 (구매 후)           |
| 26~29  | table_5          |                             | 테이블 5번                     |
| 30~33  | table_6          |                             | 테이블 6번                     |
| 34~37  | table_7          |                             | 테이블 7번                     |
| 38     | kitchen_cooking  | num_cooking / capacity      | 주방 조리율                    |
| 39     | kitchen_ready    | len(ready) / capacity       | 주방 완성률                    |
| 40     | kitchen_load     | total / capacity            | 주방 총 부하                   |
| 41     | money_ratio      | min(1, money / target)      | 목표 대비 현재 돈              |
| 42     | day_ratio        | current_day / day_limit     | 시간 경과 비율                 |
| 43     | time_remaining   | 1 - elapsed/total           | 남은 시간 비율                 |
| 44     | shop_rating      | 0.0~1.0                     | 매장 평점                      |
| 45     | can_afford       | 0.0 / 1.0                   | 구매 가능 업그레이드 존재 여부 |
| 46     | net_profit_ratio | min(1, net_profit / target) | 순이익 비율                    |
| 47     | employee_count   | len(employees) / 4.0        | 종업원 비율                    |
| 48     | bar_delivery     | 0~1.0                       | 0.5(바) + 0.5(배달) 활성 여부  |

### 테이블별 4차원 (table_i)

| 오프셋 | 이름           | 값                 | 의미                        |
| ------ | -------------- | ------------------ | --------------------------- |
| +0     | occupied       | 0.0 / 1.0          | 손님 유무                   |
| +1     | customer_state | 0.25/0.50/0.75/0.0 | waiting/ordered/eating/없음 |
| +2     | menu_id        | MENU_IDS[id] / 9   | 주문한 메뉴 종류            |
| +3     | patience_ratio | 0.0~1.0            | 남은 인내심 비율            |

---

## 13. config/train_config.json 필드 상세

| 경로                            | 타입      | 기본값        | 설명                          |
| ------------------------------- | --------- | ------------- | ----------------------------- |
| `algorithm`                     | string    | `"PPO"`       | RL 알고리즘 (현재 PPO만 지원) |
| `policy`                        | string    | `"MlpPolicy"` | SB3 정책 클래스               |
| `training.total_timesteps`      | int       | 200000        | 총 학습 스텝                  |
| `training.n_envs`               | int       | 4             | 병렬 환경 수                  |
| `training.seed`                 | int       | 42            | 랜덤 시드                     |
| `training.eval_freq`            | int       | 5000          | 평가 주기                     |
| `hyperparameters.learning_rate` | float     | 3e-4          | 학습률                        |
| `hyperparameters.n_steps`       | int       | 2048          | 롤아웃 길이                   |
| `hyperparameters.batch_size`    | int       | 64            | 미니배치                      |
| `hyperparameters.n_epochs`      | int       | 10            | 에폭 수                       |
| `hyperparameters.gamma`         | float     | 0.99          | 할인율                        |
| `hyperparameters.gae_lambda`    | float     | 0.95          | GAE 람다                      |
| `hyperparameters.clip_range`    | float     | 0.2           | PPO 클립                      |
| `hyperparameters.ent_coef`      | float     | 0.01          | 엔트로피 계수                 |
| `hyperparameters.vf_coef`       | float     | 0.5           | 가치함수 계수                 |
| `hyperparameters.max_grad_norm` | float     | 0.5           | 그래디언트 클립               |
| `network.net_arch`              | int[]     | [128, 128]    | 히든 레이어                   |
| `network.activation_fn`         | string    | `"tanh"`      | 활성화 함수                   |
| `reward_shaping.*`              | float     | 위 참조       | 보상 가중치                   |
| `game_overrides.target_money`   | int\|null | null          | 목표 금액 오버라이드          |
| `game_overrides.day_limit`      | int\|null | null          | 일수 제한 오버라이드          |

### CLI 오버라이드

`ai/train.py`의 CLI 인자는 JSON 설정보다 우선합니다:

```bash
python -m ai.train --timesteps 500000  # training.total_timesteps 오버라이드
python -m ai.train --n-envs 8          # training.n_envs 오버라이드
python -m ai.train --seed 123          # training.seed 오버라이드
python -m ai.train --config other.json # 다른 설정 파일 사용
```

---

## 14. Shop 핵심 메서드 사전

`core/shop.py` (1242줄)의 주요 메서드와 그 역할:

### 핵심 루프

| 메서드                               | 한글      | 설명                                     |
| ------------------------------------ | --------- | ---------------------------------------- |
| `step(action)`                       | 풀 스텝   | RL용: 이동+로직 한번에 실행, reward 반환 |
| `step_logic(action)`                 | 로직만    | 인간 모드용: 이동 없이 게임 로직만 실행  |
| `move_player_continuous(dx, dy, dt)` | 연속 이동 | 인간 모드용: 부드러운 픽셀 이동          |

### 상호작용

| 메서드                   | 한글            | 설명                                   |
| ------------------------ | --------------- | -------------------------------------- |
| `_interact()`            | 상호작용        | Space 키 → 가장 가까운 대상과 상호작용 |
| `_interact_table(table)` | 테이블 상호작용 | 주문 받기 / 음식 서빙 / 음료 서빙      |
| `_interact_kitchen()`    | 주방 상호작용   | 주문 전달 / 음식 수거                  |
| `_interact_bar()`        | 바 상호작용     | 음료 수거                              |

### 시스템

| 메서드                         | 한글             | 설명                                |
| ------------------------------ | ---------------- | ----------------------------------- |
| `_try_spawn_customer()`        | 손님 배치        | 빈 테이블에 랜덤 손님 생성          |
| `_pick_customer_type()`        | 손님 유형 선택   | 평점 기준 가중치 랜덤 선택          |
| `_record_satisfaction(value)`  | 만족도 기록      | 이동 평균에 추가 → shop_rating 갱신 |
| `_update_employees(dt)`        | 종업원 갱신      | 모든 종업원 상태 업데이트           |
| `_assign_employee_task(emp)`   | 종업원 임무 배정 | 우선순위 기반 작업 할당             |
| `_complete_employee_task(emp)` | 종업원 임무 완료 | 작업 실행 (주문/전달/수거/서빙)     |
| `_update_delivery(dt)`         | 배달 갱신        | 배달 주문 생성/완료/수금            |
| `_check_trait_offer()`         | 특성 체크        | 5일마다 특성 팝업 활성화            |
| `select_trait(idx)`            | 특성 선택        | 플레이어가 특성 선택 → 효과 적용    |

### 업그레이드

| 메서드                      | 한글            | 설명                                       |
| --------------------------- | --------------- | ------------------------------------------ |
| `get_upgrade_info()`        | 업그레이드 정보 | 현재 탭의 구매 가능 목록 반환              |
| `buy_upgrade(id)`           | ID로 구매       | 특정 업그레이드 구매                       |
| `buy_upgrade_by_index(idx)` | 번호로 구매     | 1~9 키로 구매                              |
| `_unlock_food(item)`        | 메뉴 해금       | 메뉴 아이템 활성화                         |
| `_apply_upgrade(upg)`       | 효과 적용       | 업그레이드 효과 실제 반영                  |
| `_auto_buy_upgrade()`       | 자동 구매       | RL용: 가장 저렴한 구매 가능 항목 자동 구매 |

---

## 15. 파일별 역할 한줄 요약

| 파일                         | 한줄 역할                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                    | 게임 실행 진입점 (--mode human/versus, --model, --target-money, --day-limit) |
| `requirements.txt`           | 의존성 목록 (pygame, gymnasium, numpy, torch, stable-baselines3)             |
| `config/settings.py`         | 전역 상수 (타일크기, FPS, 색상, 액션코드, 게임규칙) + JSON 로더              |
| `config/menu.json`           | 메뉴 8종 정의 (id, 조리시간, 가격, 해금조건)                                 |
| `config/customers.json`      | 손님 7종 정의 (id, 인내심, 재산배율, 팁, 그룹, 해금평점)                     |
| `config/upgrades.json`       | 업그레이드 7종 정의 (id, 카테고리, 비용, 효과, 최대레벨)                     |
| `config/beverages.json`      | 음료 5종 정의 (id, 제조시간, 가격, 해금조건)                                 |
| `config/traits.json`         | 특성 8종 정의 (id, 효과, 수치, 최대중첩) + 제안주기                          |
| `config/delivery.json`       | 배달 설정 (주기, 시간, 가격배율, 팁범위)                                     |
| `config/map_default.json`    | 기본 맵 (12×10 그리드, 테이블/주방/바 배치, 구매가능 테이블)                 |
| `config/train_config.json`   | RL 학습 설정 (하이퍼파라미터, 네트워크, 보상가중치, 게임오버라이드)          |
| `core/entity.py`             | Entity 기본 클래스 (픽셀좌표, 그리드좌표, 스프라이트 지원)                   |
| `core/player.py`             | Player (이동, 방향, 리스트기반 운반, carry_capacity)                         |
| `core/customer.py`           | Customer (5단계 상태머신, 그룹, 음료, 만족도, 결제 계산)                     |
| `core/station.py`            | Table (점유) + Kitchen (조리큐+배달큐) + BarStation (음료큐)                 |
| `core/employee.py`           | Employee (AI NPC, IDLE→MOVING→ACTING, 우선순위 기반 작업)                    |
| `core/shop.py`               | Shop (레스토랑 전체 엔진, 1242줄, step(), 모든 시스템 통합)                  |
| `rendering/asset_manager.py` | AssetManager (스프라이트 자동탐색, 시트분할, 프레임조회)                     |
| `rendering/renderer.py`      | Renderer (맵/엔티티/UI/업그레이드/특성 통합 렌더링, 639줄)                   |
| `modes/base_mode.py`         | BaseMode (Pygame 루프 템플릿, 고정 타임스텝, 이벤트 처리)                    |
| `modes/human_mode.py`        | HumanMode (키보드 입력, 연속이동, 업그레이드 UI, 특성 선택)                  |
| `modes/versus_mode.py`       | VersusMode (분할화면, 인간 vs AI, 시간동기화, 독립 Shop)                     |
| `ai/gym_env.py`              | TycoonEnv (Gymnasium 래퍼, obs 49차원, act 7개, reward 반환)                 |
| `ai/agent.py`                | RandomAgent + TrainedAgent (모델 로딩, 추론 인터페이스)                      |
| `ai/train.py`                | PPO 학습 (train_config.json 기반, 병렬환경, 평가콜백, TensorBoard)           |
