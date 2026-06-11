# Posterior height-field MPC 연구 메모

이번 controller는 DDBot을 그대로 따라 하는 대신, 우리 repo의 강점인 빠른 material posterior와 closed-loop height-map observation을 앞세운다.

붙인 아이디어와 근거:

- DDBot: unknown granular material에서 differentiable simulator와 skill-to-action 최적화를 쓴다. https://arxiv.org/abs/2510.17335
- Interactive Shaping of Granular Media using RL: granular state를 compact height-map으로 보고, 목표 height-map과 현재 height-map의 차이를 policy 입력으로 쓴다. https://arxiv.org/html/2509.06469v2
- ParticleFormer / Particle-Grid Neural Dynamics: action-conditioned world model을 MPC 안에 넣는 방향이 최신 흐름이다. https://arxiv.org/abs/2506.23126, https://arxiv.org/abs/2506.15680
- Particle MPC: material posterior 같은 불확실성을 scenario ensemble로 샘플링해 action을 고른다. https://stanfordasl.github.io/wp-content/papercite-data/pdf/Dyro.Harrison.ea.IROS21.pdf

이 스크립트는 위 아이디어를 가벼운 height-field digital twin으로 구현한다. 단, 현재 결과는 full MPM/real validation이 아니므로 물리적 우월성 주장은 아직 하면 안 된다.
