.PHONY: install install-lite test smoke demo demo-no-bridge artifacts clean-artifacts

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

artifacts:
	python scripts/package_demo_artifacts.py

clean-artifacts:
	rm -rf dist outputs/smoke_density_render
