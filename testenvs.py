from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html
import jax
from jax import numpy as jp
import mujoco

class TestEnv(PipelineEnv):

  def __init__(
    self,
    path,
    backend='generalized',
    **kwargs,
  ):
    sys = mjcf.load(path)
    # with open(path, "r") as f:
    #   xml_string = f.read()
    #   sys = mjcf.load_model(mujoco.MjModel.from_xml_string(xml_string))

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.0015)
      n_frames = 10
      # sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

  def reset(self, rng: jp.ndarray, init_q = None) -> State:
    # rng, rng1, rng2 = jax.random.split(rng, 3)

    # low, hi = -1e-5, 1e-5
    # qpos = self.sys.init_q + jax.random.uniform(
    #     rng1, (self.sys.q_size(),), minval=low, maxval=hi
    # )
    # qvel = jax.random.uniform(
    #     rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    # )
    if init_q is None: init_q = jp.zeros(self.sys.q_size())
    pipeline_state = self.pipeline_init(init_q, jp.zeros(self.sys.qd_size()))

    # obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
    obs = jp.zeros(10)
    reward, done, zero = jp.zeros(3)
    metrics = {'reward': zero }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    return state.replace(
      pipeline_state=pipeline_state
    )

# path_to_model = "./asset/ibm/panda_backup.urdf"
path_to_model = "./asset/door/door_unlock.xml"
env = TestEnv(path_to_model, backend="positional")
print("finish env create")

# initial state
states = []
q0 = jp.array([1.9022358751989286,
              -1.0758723471705771,
              -1.955482360527604,
              -1.3658844572639135,
              -0.6564028936195097,
              2.575062827740694,
              -0.7479169822940896 - jp.pi/2,
              0.03999999910593034,
              -0.018893523325921393,
              1.555099551599877e-23,
              -7.25692699514752e-09,])
state = env.reset(jax.random.PRNGKey(seed=0), init_q = q0)
states.append(state.pipeline_state)
print("finish state create")
# import ipdb;ipdb.set_trace()
q1 = jp.array([1.8214349166384312,
              -0.9640830815425853,
              -1.9652254391255415,
              -1.4868748694970118,
              -0.6093382086018987,
              2.521681515876036,
              -0.6089795868807693 - jp.pi/2,
              0.03988626911805058,
              -0.020599249876444306,
              0.11194818305456743,
              -0.06401025099715137])
act = (q1 - q0)[:-2]

for i in range(10):
  state = env.step(state, action = act)
  print(i)
  states.append(state.pipeline_state)

html.save("door.html", env.sys, states)