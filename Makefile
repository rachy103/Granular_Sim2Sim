.PHONY: install install-lite test smoke smoke-bridge demo demo-no-bridge experiment-smoke experiment pipeline-smoke pipeline sweep-smoke sweep analyze-sweep render-density-eef sim2sim-property sim2sim-wedge excavation-policy property-aware-mpm ddbot-core-force aalto-real-force wild-robustness wild-robustness-stress wild-review-audit artifacts clean-artifacts

install:
	./install.sh

install-lite:
	./install.sh --lite --no-menagerie

test:
	python -m pytest -q
	python -m compileall -q src scripts tests

smoke:
	./scripts/reproduce_demo_bundle.sh --smoke --skip-bridge

smoke-bridge:
	./scripts/reproduce_demo_bundle.sh --smoke

demo:
	./scripts/reproduce_demo_bundle.sh --full

demo-no-bridge:
	./scripts/reproduce_demo_bundle.sh --full --skip-bridge

experiment-smoke:
	python scripts/run_experiment_sequence.py --quick --skip-bridge

experiment:
	python scripts/run_experiment_sequence.py

pipeline-smoke:
	python scripts/run_experiment_sequence.py --quick --skip-bridge

pipeline:
	python scripts/run_experiment_sequence.py --config configs/experiments/reference_heightfield_intrusion.json

sweep-smoke:
	python scripts/run_property_sweep.py --quick --skip-bridge --count 2 --actions-per-material 1 --sweep-name smoke_lhs_sweep

sweep:
	python scripts/run_property_sweep.py --config configs/sweeps/lhs_property_sweep.json

analyze-sweep:
	python scripts/analyze_sweep_scatter.py --sweep-root outputs/sweeps/lhs_phi_cohesion_action_v001

render-density-eef:
	python scripts/render_density_mujoco_eef_render.py --config configs/rendering/density_mujoco_eef_render_fixed.json

sim2sim-property:
	python scripts/render_sim2sim_property_compare.py --config configs/rendering/sim2sim_property_compare.json

sim2sim-wedge:
	python scripts/render_sim2sim_property_compare.py --config configs/rendering/sim2sim_bulldozing_wedge.json

excavation-policy:
	python scripts/render_excavation_policy_compare.py --config configs/rendering/excavation_policy_compare.json

property-aware-mpm:
	python experiments/property_aware_mpm_excavation/run_mpm_posterior_control_ablation.py --write-video

ddbot-core-force:
	python experiments/ddbot_core_force_posterior_benchmark/run_benchmark.py --write-video

aalto-real-force:
	python experiments/aalto_real_force_classification/run_experiment.py

wild-robustness:
	python scripts/run_wild_material_robustness.py --config configs/learning/wild_material_robustness.json --output-dir outputs/wild_material_robustness

wild-robustness-stress:
	python scripts/run_wild_material_robustness.py --config configs/learning/wild_material_robustness_stress.json --output-dir outputs/wild_material_robustness_stress

wild-review-audit:
	python scripts/run_wild_review_audit.py --config configs/learning/wild_material_robustness_stress.json --quick --output-dir outputs/wild_review_audit

artifacts:
	python scripts/package_demo_artifacts.py

clean-artifacts:
	rm -rf dist outputs/smoke_density_render
