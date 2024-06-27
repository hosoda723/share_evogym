import os
import numpy as np

from gym_utils import make_vec_envs


class EvogymControllerEvaluator:
    def __init__(self, env_id, robot, num_eval=1):
        self.env_id = env_id # task
        self.robot = robot # 構造とつながり
        self.num_eval = num_eval # 試行回数

    def evaluate_controller(self, key, controller, generation):
        env = make_vec_envs(self.env_id, self.robot, 0, 1) #　１個体に対して複数回試行するために環境を複数個作る

        obs = env.reset() # 環境の初期の情報の取得
        episode_scores = [] # 
        
        while len(episode_scores) < self.num_eval: # エピソードの評価数　<　評価の実行回数
            # obs[0]は座標と速度すべての情報，2を掛けて１引くことでアクアクチュエータに設定した入力範囲に合わせる
            action = np.array(controller.activate(obs[0]))*2 - 1 
            # 
            obs, _, done, infos = env.step([np.array(action)]) #　動作の実行

            if 'episode' in infos[0]: # 辞書型info[0]にキー'episode'があるとき
                score = infos[0]['episode']['r'] # info内のepisode内のrewardをスコアに代入
                episode_scores.append(score) #　リストに追加

        env.close() # 環境を閉じる

        results = {
            'fitness': np.mean(episode_scores),  # 平均をとる
        }
        return results


class EvogymControllerEvaluatorNS:
    def __init__(self, env_id, robot, num_eval=1):
        self.env_id = env_id
        self.robot = robot
        self.num_eval = num_eval

    def evaluate_controller(self, key, controller, generation):
        env = make_vec_envs(self.env_id, self.robot, 0, 1)

        obs = env.reset()

        obs_data = []
        act_data = []

        episode_scores = []
        episode_data = []
        while len(episode_scores) < self.num_eval:
            action = np.array(controller.activate(obs[0]))*2 - 1
            obs_data.append(obs)
            act_data.append(action)
            obs, _, done, infos = env.step([np.array(action)])

            if 'episode' in infos[0]:
                obs_data = np.vstack(obs_data)
                obs_cov = self.calc_covar(obs_data)

                act_data = np.clip(np.vstack(act_data),-1,1)
                act_cov = self.calc_covar(act_data, align=False)

                data = np.hstack([obs_cov,act_cov])
                episode_data.append(data)

                obs_data = []
                act_data = []

                score = infos[0]['episode']['r']
                episode_scores.append(score)

        env.close()

        results = {
            'score': np.mean(episode_scores),
            'data': np.mean(np.vstack(episode_data),axis=0)
        }
        return results

    @staticmethod
    def calc_covar(vec, align=True):
        ave = np.mean(vec,axis=0)
        if align:
            vec_align = (vec-ave).T
        else:
            vec_align = vec.T
        comb_indices = np.tril_indices(vec.shape[1],k=0)
        covar = np.mean(vec_align[comb_indices[0]]*vec_align[comb_indices[1]],axis=1)
        return covar


from run_ppo import run_ppo

class EvogymStructureEvaluator:
    def __init__(self, env_id, save_path, ppo_iters, eval_interval, deterministic=True):
        """_summary_

        Args:
            env_id (_type_): タスク名
            save_path (_type_): _description_
            ppo_iters (_type_): 試行回数
            eval_interval (_type_): 評価間隔
            deterministic (bool, optional): 行動を確立的に（変化させる）選ぶか. Defaults to True.
        """
        self.env_id = env_id
        self.save_path = save_path
        self.robot_save_path = os.path.join(save_path, 'robot')
        self.controller_save_path = os.path.join(save_path, 'controller')
        self.ppo_iters = ppo_iters
        self.eval_interval = eval_interval
        self.deterministic = deterministic

        os.makedirs(self.robot_save_path, exist_ok=True)
        os.makedirs(self.controller_save_path, exist_ok=True)

    def evaluate_structure(self, key, robot, generation):

        file_robot = os.path.join(self.robot_save_path, f'{key}')
        file_controller = os.path.join(self.controller_save_path, f'{key}')
        np.savez(file_robot, **robot)

        reward = run_ppo(
            env_id=self.env_id, # 環境名
            robot=robot,
            train_iters=self.ppo_iters,
            eval_interval=self.eval_interval,
            save_file=file_controller, #コントローラーの保存
            deterministic=self.deterministic
        )

        results = {
            'fitness': reward,
        }
        return results

class EvogymStructureEvaluatorME:
    def __init__(self, env_id, save_path, ppo_iters, eval_interval, bd_dictionary, deterministic=True):
        self.env_id = env_id
        self.save_path = save_path
        self.robot_save_path = os.path.join(save_path, 'robot')
        self.controller_save_path = os.path.join(save_path, 'controller')
        self.ppo_iters = ppo_iters
        self.eval_interval = eval_interval
        self.bd_dictionary = bd_dictionary
        self.deterministic = deterministic

        os.makedirs(self.robot_save_path, exist_ok=True)
        os.makedirs(self.controller_save_path, exist_ok=True)

    def evaluate_structure(self, key, robot, generation):

        file_robot = os.path.join(self.robot_save_path, f'{key}')
        file_controller = os.path.join(self.controller_save_path, f'{key}')
        np.savez(file_robot, **robot)

        reward = run_ppo(
            env_id=self.env_id,
            robot=robot,
            train_iters=self.ppo_iters,
            eval_interval=self.eval_interval,
            save_file=file_controller,
            deterministic=self.deterministic
        )
        bd = {bd_name: bd_func.evaluate(robot) for bd_name,bd_func in self.bd_dictionary.items()}

        results = {
            'fitness': reward,
            'bd': bd
        }
        return results
