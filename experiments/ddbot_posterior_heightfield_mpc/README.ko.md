# DDBot 대비 Posterior Height-Field MPC 실험

이 폴더는 DDBot `sand task-2` target height-map을 기준으로, 우리 repo의 강점인 **online material posterior**와 **closed-loop height-map feedback**을 살린 controller를 정리한 실험 패키지다.

핵심 결론은 다음과 같다.

> Full MPM/real validation 전 단계인 abstract height-field digital twin benchmark에서는, 우리 posterior-conditioned closed-loop MPC가 DDBot official seed mean보다 낮은 final height-map error를 달성했다.

## 결과 요약

| Method | Final height-map error | EMD/Hungarian | Target completion | Budget |
|---|---:|---:|---:|---:|
| DDBot official seed artifacts | 3.986 +/- 0.151 | 15.925 +/- 0.416 | N/A | 21 skill optimisation steps |
| Ours target-aware CEM on local MPM | 14.111 +/- 1.250 | 14.099 +/- 1.249 | 0.093 | 21 rollout evals |
| Ours posterior height-field MPC | **2.903 +/- 0.169** | **2.903 +/- 0.169** | **1.000** | 8 executed strokes / 1536 cheap digital-twin evals |

이 결과는 “DDBot을 물리적으로 완전히 이겼다”가 아니다. 정확히는 **공유 height-map target을 대상으로 한 residual height-field controller benchmark에서 DDBot official mean을 넘었다**는 의미다.

## 왜 더 잘 나왔나

DDBot은 differentiable MPM 위에서 하나의 digging skill을 최적화한다. 반면 이 실험은 매 stroke마다 현재 height-map을 다시 보고 target과의 residual을 줄이는 closed-loop 방식이다.

우리 controller가 유리했던 이유:

- target-current residual을 직접 보고 높은 곳은 깎고 낮은 곳은 채운다.
- material posterior를 ensemble로 샘플링해 robust action을 고른다.
- full particle simulation보다 훨씬 싼 height-field digital twin에서 많은 후보를 평가한다.
- DDBot target crop은 flat bed 기준으로 pile volume이 dug volume보다 커서, crop 밖 reservoir를 허용하는 residual controller가 target pile을 더 잘 맞출 수 있다.

## 실험 조건

- Target: DDBot official `sand task-2`
- Height-map: `40 x 40`
- Physical crop: `0.24m x 0.24m`
- Ground level: `0.073m`
- Seeds: `0, 1, 2, 3, 4`
- Executed strokes: `8`
- Candidate actions per stroke: `192`
- Material scenarios: `9`
- Total cheap planning evaluations: `1536` per seed

## 재현 명령

repo root에서 다음 명령으로 재현한다.

```powershell
.\.venv\Scripts\python.exe experiments\ddbot_posterior_heightfield_mpc\run_posterior_heightfield_mpc.py --seeds 0,1,2,3,4 --strokes 8 --candidates 192 --ensemble 9 --write-video
```

이 스크립트는 DDBot target height-map과 official sim height-map을 이 폴더의 `data/`에서 읽는다. 파일이 없을 때는 helper에 들어 있는 GitHub media URL로 다시 받을 수 있다.

## 포함 파일

- `run_posterior_heightfield_mpc.py`: posterior-conditioned height-field MPC controller
- `run_shared_benchmark.py`: metric 계산과 DDBot target download URL을 재사용하는 helper
- `data/ddbot_target_sand_task2_height_map_res40.npy`: DDBot sand task-2 target height-map
- `data/ddbot_official_sim_sand_task2_height_map_res40.npy`: DDBot official sim artifact height-map
- `results/summary.json`: 5-seed aggregate result
- `results/seed_results.csv`: seed별 최종 metric
- `results/action_log.csv`: stroke별 action/result log
- `assets/posterior_heightfield_mpc_comparison.mp4`: 논문 페이지 상단용 비교 영상
- `assets/posterior_heightfield_mpc_poster.jpg`: 영상 poster
- `assets/convergence.png`: stroke별 height-map error convergence
- `assets/ours_mean_height_map.png`: ours mean final height-map
- `assets/ddbot_target_height_map.png`: DDBot target height-map
- `assets/ddbot_official_sim_height_map.png`: DDBot official sim artifact height-map
- `RESEARCH_NOTE.ko.md`: 관련 연구와 의의 요약

## 논문에서 쓸 수 있는 표현

강하게 말할 수 있는 표현:

> On the shared DDBot sand task-2 height-map target, our posterior-conditioned closed-loop height-field MPC achieves lower final height-map error than the DDBot official seed mean in an abstract digital-twin benchmark.

아직 말하면 안 되는 표현:

> Our full physical excavation controller outperforms DDBot.

그 주장을 하려면 이 controller를 local MPM 또는 DDBot DOMA runtime에 넣고, 같은 initial bed와 target point cloud에서 다시 검증해야 한다.

## 다음 단계

1. Height-field MPC가 고른 macro-action을 local Warp MPM tool trajectory로 변환한다.
2. Full MPM에서 same target height-map error가 DDBot official mean 아래로 내려가는지 본다.
3. DDBot DOMA runtime이 가능해지면, 우리 posterior를 DDBot parameter prior로 넣는 bridge 실험을 추가한다.
