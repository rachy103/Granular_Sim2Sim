.PHONY: install install-lite test smoke smoke-bridge demo demo-no-bridge experiment-smoke experiment pipeline-smoke pipeline sweep-smoke sweep analyze-sweep artifacts clean-artifacts

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

artifacts:
	python scripts/package_demo_artifacts.py

clean-artifacts:
	rm -rf dist outputs/smoke_density_render
