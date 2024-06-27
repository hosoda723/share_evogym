

class MazeControllerEvaluator:
    def __init__(self, maze, timesteps):
        self.maze = maze
        self.timesteps = timesteps

    def evaluate_agent(self, key, controller, generation):
        self.maze.reset() # 迷路環境のリセット

        done = False
        for i in range(self.timesteps):
            obs = self.maze.get_observation()
            action = controller.activate(obs)
            done = self.maze.update(action)
            if done:
                break
            
        last_loc = self.maze.get_agent_location() # エージェントの位置をとる

        # 適応度の計算
        if done: #ゴールについた場合
            score = 1.0
        elif last_loc[1] >= 40 and last_loc[1] <= 80: # ゴールにつかなかった場合
            # 距離を求める
            distance = self.maze.get_distance_to_exit()
            # 適応度 = （初期状態の距離 - 評価対象の距離） / 初期状態の距離
            score = (self.maze.initial_distance - distance) / self.maze.initial_distance
        else:
            score = -1.0
            
        results = {
            'fitness': score,
            'data': last_loc
        }
        return results


class MazeControllerEvaluatorNS:
    def __init__(self, maze, timesteps):
        self.maze = maze
        self.timesteps = timesteps

    def evaluate_agent(self, key, controller, generation):
        self.maze.reset()

        done = False
        for i in range(self.timesteps):
            obs = self.maze.get_observation()
            action = controller.activate(obs)
            done = self.maze.update(action)
            if done:
                break

        if done:
            score = 1.0
        else:
            distance = self.maze.get_distance_to_exit()
            score = (self.maze.initial_distance - distance) / self.maze.initial_distance

        last_loc = self.maze.get_agent_location()
        results = {
            'score': score,
            'data': last_loc
        }
        return results
