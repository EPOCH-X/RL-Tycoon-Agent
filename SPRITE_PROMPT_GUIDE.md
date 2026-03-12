# 🎨 스프라이트 생성 AI 프롬프트 가이드

> 이미지 생성 AI(Midjourney, DALL-E, Stable Diffusion 등)에게 전달할 기준 사양과 프롬프트 모음입니다.
> 모든 스프라이트는 **Pygame 기반 탑다운(top-down) 레스토랑 경영 시뮬레이션** 게임용입니다.

---

## 목차

1. [공통 사양](#1-공통-사양)
2. [색상 팔레트](#2-색상-팔레트)
3. [파일 구조 &amp; 네이밍](#3-파일-구조--네이밍)
4. [플레이어 스프라이트](#4-플레이어-스프라이트)
5. [손님 스프라이트 (7종)](#5-손님-스프라이트-7종)
6. [직원 스프라이트](#6-직원-스프라이트)
7. [가구 &amp; 시설물 타일](#7-가구--시설물-타일)
8. [음식 아이콘 (8종)](#8-음식-아이콘-8종)
9. [음료 아이콘 (5종)](#9-음료-아이콘-5종)
10. [바닥 &amp; 벽 타일](#10-바닥--벽-타일)
11. [프롬프트 공통 접미사](#11-프롬프트-공통-접미사)

---

## 1. 공통 사양

| 항목                | 값                                   | 비고                                             |
| ------------------- | ------------------------------------ | ------------------------------------------------ |
| **타일 크기**       | 64 × 64 px                           | 모든 엔티티·타일의 기본 단위                     |
| **스프라이트 시트** | 가로(수평) 스트립                    | 프레임을 왼→오로 나열, 높이 64px 고정            |
| **프레임 크기**     | 64 × 64 px                           | 시트 너비 ÷ 64 = 프레임 수, 자동 인식            |
| **투명 배경**       | PNG (알파 채널)                      | 배경은 반드시 투명                               |
| **화면 해상도**     | 1024 × 760 px                        | 게임 영역 1024×640 + UI 패널 120px               |
| **그리드**          | 16 × 10 타일                         | 가로 16칸 × 세로 10칸                            |
| **애니메이션 속도** | 0.15 초/프레임 (≈ 6.7 FPS)           | 코드 기본값, 너무 빠르거나 느린 애니는 맞지 않음 |
| **아트 스타일**     | 플랫 컬러 / 심플 아이소메트릭 탑다운 | 디포르메(SD체형), 2~3등신 권장                   |
| **윤곽선**          | 1~2px 검정 외곽선 권장               | 작은 크기에서 가독성 확보                        |

### 📐 해상도 규칙

```
캐릭터 스프라이트 시트 = (프레임 수 × 64) × 64 px
  예) 6프레임 걷기  →  384 × 64 px  (6 × 64 = 384)
  예) 4프레임 대기  →  256 × 64 px  (4 × 64 = 256)

음식/음료 아이콘 = 64 × 64 px (단일 프레임)
타일 텍스처    = 64 × 64 px (단일 프레임, 반복 가능하면 좋음)
```

---

## 2. 색상 팔레트

현재 Phase 1(사각형 렌더링)에서 사용 중인 색상입니다. 스프라이트의 **주요 톤**으로 참고하세요.

### 캐릭터 색상

| 대상            | RGB             | HEX       | 용도                |
| --------------- | --------------- | --------- | ------------------- |
| 플레이어        | (50, 120, 220)  | `#3278DC` | 기본 (파란색 계열)  |
| 플레이어 (운반) | (80, 160, 255)  | `#50A0FF` | 아이템 들고 있을 때 |
| 직원            | (80, 200, 160)  | `#50C8A0` | 기본 (청록 계열)    |
| 직원 (운반)     | (120, 240, 190) | `#78F0BE` | 아이템 들고 있을 때 |

### 손님 유형별 색상

| 유형    | 한글   | RGB             | HEX       |
| ------- | ------ | --------------- | --------- |
| budget  | 알뜰   | (220, 180, 50)  | `#DCB432` |
| normal  | 일반   | (220, 180, 50)  | `#DCB432` |
| family  | 가족   | (180, 200, 100) | `#B4C864` |
| tourist | 관광객 | (100, 180, 220) | `#64B4DC` |
| wealthy | 부유한 | (180, 120, 220) | `#B478DC` |
| vip     | VIP    | (255, 215, 0)   | `#FFD700` |
| critic  | 평론가 | (220, 50, 50)   | `#DC3232` |

### 시설물 색상

| 대상            | RGB             | HEX       | 상태      |
| --------------- | --------------- | --------- | --------- |
| 바닥            | (200, 200, 180) | `#C8C8B4` | -         |
| 벽              | (80, 80, 80)    | `#505050` | -         |
| 테이블          | (139, 90, 43)   | `#8B5A2B` | 비어있음  |
| 테이블 (사용중) | (160, 110, 60)  | `#A06E3C` | 손님 착석 |
| 주방 (대기)     | (180, 60, 60)   | `#B43C3C` | 비활성    |
| 주방 (조리중)   | (220, 100, 50)  | `#DC6432` | 조리      |
| 주방 (완성)     | (80, 220, 80)   | `#50DC50` | 음식 완성 |
| 바 (대기)       | (100, 60, 140)  | `#643C8C` | 비활성    |
| 바 (완성)       | (160, 100, 220) | `#A064DC` | 음료 완성 |
| 쓰레기통        | (120, 100, 80)  | `#786450` | -         |

---

## 3. 파일 구조 & 네이밍

```
assets/sprites/
├── player/
│   ├── idle.png              ← 대기 (4~6프레임)
│   ├── up.png                ← 위 이동 (6~8프레임)
│   ├── down.png              ← 아래 이동 (6~8프레임)
│   ├── left.png              ← 왼쪽 이동 (6~8프레임)
│   └── right.png             ← 오른쪽 이동 (6~8프레임)
│
├── customer/
│   ├── walking_to_table.png  ← 테이블로 이동 (6~8프레임)
│   ├── waiting_to_order.png  ← 주문 대기 (4~6프레임)
│   ├── order_taken.png       ← 음식 대기 (4~6프레임)
│   ├── eating.png            ← 식사 중 (4~8프레임)
│   ├── leaving_happy.png     ← 만족 퇴장 (4~6프레임)
│   └── leaving_angry.png     ← 불만 퇴장 (4~6프레임)
│
├── employee/
│   ├── idle.png              ← 대기 (4~6프레임)
│   ├── moving.png            ← 이동 (6~8프레임)
│   └── acting.png            ← 작업 수행 (4~6프레임)
│
├── table/
│   └── idle.png              ← 테이블 (1프레임)
├── kitchen/
│   ├── idle.png              ← 주방 대기 (1프레임)
│   ├── cooking.png           ← 조리 중 (2~4프레임, 불 이펙트)
│   └── ready.png             ← 완성 (1~2프레임, 반짝임)
├── bar/
│   ├── idle.png              ← 바 대기 (1프레임)
│   └── ready.png             ← 음료 완성 (1~2프레임)
├── trash_can/
│   └── idle.png              ← 쓰레기통 (1프레임)
│
├── food/                     ← 음식 아이콘 (각 64×64, 1프레임)
│   ├── coffee.png
│   ├── sandwich.png
│   ├── pasta.png
│   ├── steak.png
│   ├── sushi_set.png
│   ├── lobster.png
│   ├── wagyu.png
│   └── truffle_course.png
│
└── drink/                    ← 음료 아이콘 (각 64×64, 1프레임)
    ├── water.png
    ├── juice.png
    ├── lemonade.png
    ├── cocktail.png
    └── wine.png
```

> **핵심 규칙**: 폴더명 = `sprite_key`, 파일명(확장자 제외) = `animation_state`
> 코드에서 `asset_manager.get_frame("player", "idle", frame_index)` 형태로 호출합니다.

---

## 4. 플레이어 스프라이트

### 사양

| 항목        | 값                              |
| ----------- | ------------------------------- |
| 시트 크기   | (프레임수 × 64) × 64 px         |
| 권장 프레임 | 대기 4~6프레임, 이동 6~8프레임  |
| 이동 속도   | 180 px/초                       |
| 주요 색상   | 파란색 계열 (#3278DC)           |
| 운반 색상   | 밝은 파란 (#50A0FF)             |
| 시점        | 탑다운 (머리 위에서 내려다보기) |

### AI 프롬프트

#### idle (대기)

```
Top-down 2D pixel art sprite sheet of a restaurant waiter character standing idle.
Art style: flat color, 2-3 head-tall chibi/super-deformed proportions.
Primary color: bright blue (#3278DC) uniform/apron.
Character faces downward (toward camera in top-down view).
Subtle idle animation: slight sway or breathing motion.
4 frames, each frame exactly 64×64 pixels.
Arranged in a single horizontal strip: total image size 256×64 pixels.
Transparent PNG background. Black outline 1-2px.
Simple, clean, easily readable at small size.
```

#### up / down / left / right (방향 이동)

```
Top-down 2D pixel art sprite sheet of a restaurant waiter character walking [DIRECTION].
Art style: flat color, 2-3 head-tall chibi/super-deformed proportions.
Primary color: bright blue (#3278DC) uniform/apron.
Walking animation cycle in [up/down/left/right] direction from top-down perspective.
6 frames, each frame exactly 64×64 pixels.
Arranged in a single horizontal strip: total image size 384×64 pixels.
Transparent PNG background. Black outline 1-2px.
Clear leg/arm movement showing walking motion. Smooth loop.
```

> **참고**: 운반 상태는 색상만 밝은 파란(#50A0FF)으로 변하므로, 별도 시트를 만들거나 코드에서 색조 조정 가능.
> 선택적으로 `carrying_down.png` 등을 추가하면 아이템을 들고 있는 모습 표현 가능.

---

## 5. 손님 스프라이트 (7종)

### 사양

| 항목        | 값                                |
| ----------- | --------------------------------- |
| 시트 크기   | (프레임수 × 64) × 64 px           |
| 권장 프레임 | 이동 6~8, 대기/식사 4~6, 퇴장 4~6 |
| 이동 속도   | 80 px/초 (입구 → 테이블)          |
| 시점        | 탑다운                            |
| 변형 수     | 7종 (색상/의상 차이)              |

### 상태별 애니메이션

| 파일명                 | 상태             | 설명                            | 권장 프레임 |
| ---------------------- | ---------------- | ------------------------------- | ----------- |
| `walking_to_table.png` | 입구→테이블 이동 | 걷는 모션, 밝은 표정            | 6~8         |
| `waiting_to_order.png` | 주문 대기        | 앉아서 두리번, 머리 위 ?! 느낌  | 4~6         |
| `order_taken.png`      | 음식 대기        | 앉아서 기다리기, 차분한 모션    | 4~6         |
| `eating.png`           | 식사 중          | 냠냠 씹는 모션, 행복한 표정     | 4~8         |
| `leaving_happy.png`    | 만족 퇴장        | 걸어나가기, 미소/하트 이펙트    | 4~6         |
| `leaving_angry.png`    | 불만 퇴장        | 화난 표정, 빠른 걸음, 분노 표시 | 4~6         |

### 유형별 디자인 가이드

#### budget (알뜰 손님) & normal (일반 손님)

```
Top-down 2D pixel art sprite sheet of a casual restaurant customer [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: warm gold/tan (#DCB432).
Casual everyday clothing (t-shirt, jeans). Friendly, average appearance.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### family (가족)

```
Top-down 2D pixel art sprite sheet of a family group of restaurant customers [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: olive green (#B4C864).
Show a small family group (parent + 1-2 children) within a single 64×64 frame.
Warm, friendly appearance. Children slightly smaller than parent figure.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### tourist (관광객)

```
Top-down 2D pixel art sprite sheet of a tourist restaurant customer [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: sky blue (#64B4DC).
Tourist outfit: bucket hat or sun hat, camera around neck, casual travel wear.
Curious, excited expression. Looking around.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### wealthy (부유한 손님)

```
Top-down 2D pixel art sprite sheet of a wealthy/elegant restaurant customer [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: light purple (#B478DC).
Upscale formal attire: suit or dress, refined posture. Pearl necklace or cufflinks detail.
Distinguished, composed expression.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### vip (VIP)

```
Top-down 2D pixel art sprite sheet of a VIP restaurant customer [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: bright gold (#FFD700).
Premium luxury appearance: golden accessories, sunglasses, high-end fashion.
Confident, exclusive aura. Subtle sparkle or glow effect around character.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### critic (평론가)

```
Top-down 2D pixel art sprite sheet of a food critic restaurant customer [ANIMATION STATE].
Art style: flat color, 2-3 head-tall chibi proportions.
Primary clothing color: red (#DC3232).
Professional appearance: notebook/pen in hand, glasses, stern/analytical expression.
Formal critic style: beret or neat hair, serious face.
[FRAME COUNT] frames, each exactly 64×64 pixels.
Single horizontal strip: total [FRAME COUNT × 64]×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

### 손님 리컬러 팁

> 캐릭터 실루엣은 모든 유형에서 동일하게 유지하고, **의상 색상 + 소품(모자, 안경 등)**만 바꿔서 7종을 만드는 것이 효율적입니다.
> 가족(family)만 실루엣이 다릅니다 (그룹 표현).

---

## 6. 직원 스프라이트

### 사양

| 항목        | 값                                  |
| ----------- | ----------------------------------- |
| 시트 크기   | (프레임수 × 64) × 64 px             |
| 권장 프레임 | 대기 4~6, 이동 6~8, 작업 4~6        |
| 이동 속도   | 120 px/초 (기본값, 업그레이드 가능) |
| 주요 색상   | 청록 (#50C8A0)                      |
| 운반 색상   | 밝은 청록 (#78F0BE)                 |
| 시점        | 탑다운                              |

### AI 프롬프트

#### idle (대기)

```
Top-down 2D pixel art sprite sheet of a restaurant staff/employee NPC standing idle.
Art style: flat color, 2-3 head-tall chibi proportions.
Primary color: teal (#50C8A0) apron/uniform.
Distinct from the player character: wears an apron or chef hat, rounder silhouette.
Subtle idle animation: slight sway or arm movement.
4 frames, each exactly 64×64 pixels.
Single horizontal strip: total 256×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### moving (이동)

```
Top-down 2D pixel art sprite sheet of a restaurant staff/employee NPC walking.
Art style: flat color, 2-3 head-tall chibi proportions.
Primary color: teal (#50C8A0) apron/uniform.
Walking animation from top-down view. Purposeful, brisk movement.
6 frames, each exactly 64×64 pixels.
Single horizontal strip: total 384×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

#### acting (작업 수행)

```
Top-down 2D pixel art sprite sheet of a restaurant staff/employee NPC performing a task.
Art style: flat color, 2-3 head-tall chibi proportions.
Primary color: teal (#50C8A0) apron/uniform.
Action pose: bending forward or reaching out (taking order, serving food, picking up dishes).
4 frames, each exactly 64×64 pixels.
Single horizontal strip: total 256×64 pixels.
Transparent PNG background. Black outline 1-2px.
```

---

## 7. 가구 & 시설물 타일

모두 **64 × 64 px**, 단일 프레임(또는 상태별 1~2프레임)입니다.

### AI 프롬프트

#### 테이블 (table)

```
Top-down 2D pixel art of a small wooden restaurant table.
Size: exactly 64×64 pixels. Brown wood tone (#8B5A2B).
Simple 4-seat square table viewed from directly above.
Clean edges, 1-2px black outline. Transparent background.
Flat color style matching a casual restaurant game aesthetic.
```

#### 주방 카운터 (kitchen) — 3가지 상태

```
Top-down 2D pixel art of a restaurant kitchen cooking station, [STATE].
Size: exactly 64×64 pixels.

State variants (create separately):
1. Idle: dark red counter (#B43C3C), no flame, clean surface.
2. Cooking: orange-red (#DC6432), small flame or steam effect, pot/pan visible.
3. Ready: bright green glow (#50DC50), completed dish on counter, sparkle effect.

Flat color style. 1-2px black outline. Transparent background.
```

#### 바 카운터 (bar) — 2가지 상태

```
Top-down 2D pixel art of a restaurant bar/drink counter, [STATE].
Size: exactly 64×64 pixels.

State variants:
1. Idle: purple tone (#643C8C), bottles on shelf, clean counter.
2. Ready: light purple glow (#A064DC), prepared drink on counter, small sparkle.

Flat color style. 1-2px black outline. Transparent background.
```

#### 쓰레기통 (trash_can)

```
Top-down 2D pixel art of a small restaurant trash can/bin.
Size: exactly 64×64 pixels. Brown tone (#786450).
Simple cylindrical or rectangular bin viewed from above.
Flat color style. 1-2px black outline. Transparent background.
```

---

## 8. 음식 아이콘 (8종)

각 **64 × 64 px**, 단일 프레임, 투명 배경. 접시 위에 올려진 모습 권장.

| ID             | 한글          | 프롬프트 핵심 키워드                       |
| -------------- | ------------- | ------------------------------------------ |
| coffee         | 커피          | 작은 커피컵, 김 올라오는 모습, 갈색        |
| sandwich       | 샌드위치      | 삼각/사각 샌드위치, 빵+채소+고기 레이어    |
| pasta          | 파스타        | 흰 접시 위 스파게티, 토마토 소스, 포크     |
| steak          | 스테이크      | 접시 위 소고기 스테이크, 구운 자국, 허브   |
| sushi_set      | 초밥 세트     | 나무 접시 위 초밥 3~4개, 밝은 색상         |
| lobster        | 랍스터        | 빨간 랍스터, 접시 위, 레몬 장식            |
| wagyu          | 와규 스테이크 | 고급 마블링 스테이크, 금색 접시 느낌       |
| truffle_course | 트러플 코스   | 고급 코스 요리, 트러플 슬라이스, 소스 데코 |

### 공통 프롬프트

```
Top-down 2D pixel art food icon: [음식 이름 + 핵심 키워드].
Size: exactly 64×64 pixels. Served on a small plate or dish.
Flat color, appetizing presentation, recognizable at small size.
1-2px black outline. Transparent PNG background.
Game asset style matching a casual restaurant simulation.
```

### 개별 예시

```
[coffee]
Top-down 2D pixel art food icon: a small cup of hot coffee with steam rising.
Brown coffee in a white cup on a small saucer.
Size: exactly 64×64 pixels.
Flat color, 1-2px black outline. Transparent PNG background.

[sushi_set]
Top-down 2D pixel art food icon: a sushi set with 3-4 pieces of nigiri sushi
on a wooden serving board. Colorful fish toppings (salmon pink, tuna red, egg yellow).
Size: exactly 64×64 pixels.
Flat color, 1-2px black outline. Transparent PNG background.

[truffle_course]
Top-down 2D pixel art food icon: an elegant gourmet truffle course dish.
White fine dining plate with truffle slices, sauce drizzle, and herb garnish.
Size: exactly 64×64 pixels.
Flat color, 1-2px black outline. Transparent PNG background.
```

---

## 9. 음료 아이콘 (5종)

각 **64 × 64 px**, 단일 프레임, 투명 배경.

| ID       | 한글       | 프롬프트 핵심 키워드                       |
| -------- | ---------- | ------------------------------------------ |
| water    | 물         | 투명한 유리컵, 물, 얼음 조각               |
| juice    | 주스       | 오렌지색 주스, 유리컵, 빨대                |
| lemonade | 레모네이드 | 노란 레모네이드, 레몬 슬라이스, 빨대       |
| cocktail | 칵테일     | 삼각형 칵테일 글라스, 밝은 색상, 체리 장식 |
| wine     | 와인       | 와인 글라스, 붉은 와인, 우아한 형태        |

### 공통 프롬프트

```
Top-down 2D pixel art drink icon: [음료 이름 + 핵심 키워드].
Size: exactly 64×64 pixels. Glass or cup viewed from slightly above.
Flat color, refreshing and recognizable at small size.
1-2px black outline. Transparent PNG background.
Game asset style matching a casual restaurant simulation.
```

---

## 10. 바닥 & 벽 타일

각 **64 × 64 px**, 단일 프레임, **타일링(반복 배치) 가능**해야 합니다.

#### 바닥 (floor)

```
Top-down 2D pixel art floor tile for a restaurant interior.
Size: exactly 64×64 pixels. Light beige/cream tone (#C8C8B4).
Subtle wood plank or stone texture. Seamlessly tileable in all directions.
Flat color style, minimal detail. No outline needed for seamless tiling.
```

#### 벽 (wall)

```
Top-down 2D pixel art wall tile for a restaurant interior border.
Size: exactly 64×64 pixels. Dark gray (#505050).
Brick or stone wall texture viewed from above. Seamlessly tileable horizontally.
Flat color style. Clear distinction from floor tile.
```

---

## 11. 프롬프트 공통 접미사

모든 프롬프트 뒤에 아래 접미사를 붙여주면 일관성을 유지하기 좋습니다.

```
[공통 접미사 — 모든 에셋에 추가]
Game art asset for a top-down 2D restaurant management simulation game.
Pixel art style with flat colors and clean 1-2px black outlines.
Character proportions: 2-3 head-tall super deformed (chibi) style.
Must be exactly [WIDTH]×64 pixels with transparent PNG background.
Consistent art style across all assets. No gradients, no realistic shading.
Optimized for display at 64×64 pixel tiles on screen.
Korean restaurant tycoon game aesthetic.
```

---

## 부록: 스프라이트 시트 예시 구조

```
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ F1   │ F2   │ F3   │ F4   │ F5   │ F6   │  ← 64px 높이
│64×64 │64×64 │64×64 │64×64 │64×64 │64×64 │
└──────┴──────┴──────┴──────┴──────┴──────┘
  384px 너비 (6프레임 × 64px)
```

- **F1~F6**: 각 프레임은 64×64 px
- 하나의 PNG 파일 = 하나의 애니메이션 상태
- 코드가 자동으로 가로 방향으로 64px씩 잘라서 프레임 배열 생성

---

## 부록: 작업 우선순위 체크리스트

### 🔴 최우선 (핵심 게임플레이)

- [ ] 플레이어: idle, up, down, left, right (5 시트)
- [ ] 손님(일반): walking_to_table, waiting_to_order, order_taken, eating, leaving_happy, leaving_angry (6 시트)
- [ ] 직원: idle, moving, acting (3 시트)

### 🟡 중요 (비주얼 품질)

- [ ] 손님 유형별 리컬러/변형 (6종 추가 = budget, family, tourist, wealthy, vip, critic)
- [ ] 음식 아이콘 8종
- [ ] 음료 아이콘 5종

### 🟢 선택 (환경 완성)

- [ ] 바닥 타일
- [ ] 벽 타일
- [ ] 테이블
- [ ] 주방 카운터 (3 상태)
- [ ] 바 카운터 (2 상태)
- [ ] 쓰레기통
- [ ] 플레이어 운반 상태 변형

---

## 부록: AI 도구별 팁

### Midjourney

- `--ar 6:1` (6프레임 시트), `--ar 4:1` (4프레임 시트) 비율 사용
- `--style raw` 로 일관성 유지
- `--no gradient, shadow, realistic` 추가 권장

### DALL-E 3

- "pixel art sprite sheet" 키워드 필수
- 정확한 픽셀 크기 지정이 어려우므로 생성 후 수동 리사이즈 필요
- 한 번에 한 상태(시트) 생성 후 취합 권장

### Stable Diffusion (LoRA)

- pixel-art LoRA 모델 사용 권장
- `<lora:pixel-art-style:0.8>` 계열
- ControlNet으로 포즈 제어 + img2img로 프레임간 일관성 유지
- ComfyUI 워크플로우로 배치 생성 효율적

### Aseprite (수동 보정)

- AI 생성 결과물의 최종 보정용으로 Aseprite 사용 권장
- 프레임 크기 맞추기, 팔레트 통일, 외곽선 정리
- 스프라이트 시트 Export 기능으로 최종 PNG 출력
