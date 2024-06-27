import os
import gym
import numpy as np
import multiprocessing.pool

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def make_env(env_id, env_kwargs, seed, allow_early_resets=True):
    def _init():
        env = gym.make(env_id, **env_kwargs)
        # アクチュエータの動作の入力範囲の設定
        env.action_space = gym.spaces.Box(low=-1.0, high=1.0,
            shape=env.action_space.shape, dtype=np.float)
        env.seed(seed)
        env = Monitor(env, None, allow_early_resets=True)
        return env
    return _init

def make_vec_envs(env_id, env_kwargs, seed, num_processes, gamma=None, vecnormalize=False, subproc=True, allow_early_resets=True):
    """_summary_

    Args:
        env_id (_type_): タスク名
        env_kwargs (_type_): _description_
        seed (_type_): seed値
        num_processes (_type_): _description_
        gamma (_type_, optional): _description_. Defaults to None.
        vecnormalize (bool, optional): _description_. Defaults to False.
        subproc (bool, optional): _description_. Defaults to True.
        allow_early_resets (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    envs = [make_env(env_id, env_kwargs, seed+i, allow_early_resets=allow_early_resets) for i in range(num_processes)]

    if subproc and num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if vecnormalize:
        if gamma is not None:
            envs = VecNormalize(envs, gamma=gamma)
        else:
            envs = VecNormalize(envs, norm_reward=False)
    
    return envs


from evogym import is_connected, has_actuator, get_full_connectivity

def load_robot(ROOT_DIR, robot_name, task=None):

    if robot_name=='default': # 名前が決まっていない
        robot_name = task
        robot_file = os.path.join(ROOT_DIR, 'envs', 'evogym', 'robot_files', f'{robot_name}.txt')
        assert os.path.exists(robot_file), f'defalt robot is not set on the task {task}'
    else:
        robot_file = os.path.join(ROOT_DIR, 'envs', 'evogym', 'robot_files', f'{robot_name}.txt')

    body = np.loadtxt(robot_file) # ロボット設定ファイルを読み込む
    assert is_connected(body), f'robot {robot_name} is not fully connected'
    assert has_actuator(body), f'robot {robot_name} have not actuator block'

    connections = get_full_connectivity(body)
    robot = {
        'body': body,
        'connections': connections
    }
    return robot
