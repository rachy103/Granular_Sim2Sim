# Property-aware MPM Excavation Ablation

## 핵심 의의

이 실험은 물성치 예측 자체가 목적이 아니라, 예측된 물성 posterior가 실제 굴착 행동 선택을 바꾸는지 확인하기 위한 downstream control 실험이다.

같은 GT MPM 모래층, 같은 target corridor, 같은 blade/tool 모델, 같은 trajectory budget을 고정한 뒤 controller가 가진 belief만 바꿨다.

- No posterior: 물성 정보를 쓰지 않는 기본 controller
- Wrong posterior: 틀린 물성을 믿는 controller
- Estimated posterior: 짧은 interaction으로 추정한 posterior를 쓰는 controller
- GT property: 실제 물성값을 알고 있는 oracle reference

## 실험 세팅

- Environment: 3D MPM sand bed
- Task: 중앙 trench를 파고 전방 deposit zone으로 모래를 이동
- Controller budget: 동일한 후보 trajectory grid
- Evaluation: 최종 trench depth, target mass, forward transport, spillage, peak force, force violation, intuitive score
- Force limit: 2850 N

## 결과 요약

| Controller belief | Score | Forward transport | Peak force | Trench depth | Spillage mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| No posterior | 135.8 | 4185.5 | 1595.4 | 25.5 mm | 10115.0 |
| Wrong posterior | 137.6 | 3687.0 | 2571.0 | 0.0 mm | 5677.7 |
| Estimated posterior | 180.3 | 3768.7 | 2257.1 | 45.6 mm | 6296.2 |
| GT property | 175.8 | 3740.2 | 2253.4 | 39.1 mm | 6110.6 |

Estimated posterior가 가장 높은 score와 trench completion을 보였다. 이것은 추정값이 GT보다 더 정확하다는 뜻이 아니라, 제한된 후보 trajectory budget 안에서 posterior가 충분히 좋은 action을 고르게 만들었다는 뜻이다. GT property row는 oracle reference이지 전역 최적해가 아니다.

## 발표에서 말할 수 있는 주장

이 실험이 직접 입증하는 것은 다음 한 문장이다.

> 로봇이 모래를 짧게 만져서 얻은 물성 posterior는 단순한 분류 결과에 그치지 않고, 같은 굴착 목표에서 어떤 궤적을 선택할지 바꿀 만큼 control에 유용하다.

따라서 이 실험은 "물성치를 잘 맞췄다"에서 끝나지 않고, "그 물성 추정이 실제 task 성능으로 이어질 수 있다"는 연결고리를 보여준다.

## 주요 산출물

```text
assets/mpm_posterior_control_ablation.mp4
assets/mpm_posterior_control_ablation.jpg
assets/mpm_posterior_control_ablation_sheet.jpg
assets/mpm_posterior_control_summary.png
results/mpm_posterior_control_summary.csv
results/mpm_posterior_control_frames.csv
results/mpm_posterior_control_summary.json
```
