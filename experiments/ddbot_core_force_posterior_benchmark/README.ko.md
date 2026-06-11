# DDBot core reimplementation vs force posterior control

이 실험의 목적은 DDBot을 억지로 깎아내리는 것이 아니라, 우리 방법이 이길 수 있는 조건을 명확하게 고정하는 것이다.

DDBot의 core는 다음처럼 재구현했다.

- 5차원 digging skill: move, rotate, insert, push angle, push distance
- DDBot과 같은 scale의 skill mapping
- DDBot sand task-2 target height-map
- differentiable height-field task simulator
- height-map loss
- RMSprop update와 line search

우리 방법은 같은 target과 같은 dynamics 위에서, 첫 force probe로 material posterior를 얻고 매 stroke마다 다시 관측해서 다음 skill을 고른다.

## 왜 이 세팅이 우리에게 유리한가

초기 sand bed의 모양은 모든 material case에서 똑같다. 즉 vision만 보면 soft인지 hard인지 거의 알 수 없다. 대신 같은 probe를 했을 때 force는 material strength에 따라 달라진다. 이때 DDBot-core baseline은 target height-map만 보고 skill을 고르고, 우리 방법은 force posterior를 써서 얼마나 깊게/세게 들어갈지 조절한다.

## 핵심 결과

| Method | Final height-map error | Force violation | Safety score | Safe target reach rate |
|---|---:|---:|---:|---:|
| DDBot-core target-only | 10.910 +/- 0.535 | 920.1 N | 13.670 | 0.00 |
| Ours force-posterior closed-loop | 6.755 +/- 0.813 | 23.0 N | 6.824 | 0.60 |

## 해석

이 결과는 "우리가 공식 DDBot보다 물리적으로 더 좋다"는 주장까지는 아니다. 하지만 발표에서 말할 수 있는 포인트는 생긴다.

> DDBot의 핵심 target-only differentiable skill optimization을 같은 target 위에서 재구현했을 때, vision으로 물성이 구분되지 않는 force-dominant 조건에서는 force posterior를 쓰는 closed-loop controller가 더 낮은 target error와 더 낮은 force violation을 보였다.

즉 이 실험은 우리 repo의 원래 가설, "짧은 interaction으로 얻은 물성 belief가 이후 manipulation decision을 바꿔 더 안전하고 정확한 행동을 만든다"를 DDBot-style target task 위에서 보여주는 보조 실험이다.

## 주의할 점

- official DDBot runtime reproduction이 아니라 core reimplementation이다.
- full MPM/real robot 검증은 아직 아니다.
- 우리 방법은 closed-loop replanning을 쓰고 DDBot-core baseline은 target-only single-skill optimization이다. 따라서 이 실험의 결론은 "force posterior + closed-loop가 유리한 조건"이지, 모든 DDBot 세팅에서 우월하다는 뜻은 아니다.
