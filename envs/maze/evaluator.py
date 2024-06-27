

class MazeControllerEvaluator:
    def __init__(self, maze, timesteps):
        self.maze = maze
        self.timesteps = timesteps

    def evaluate_agent(self, key, controller, generation):
        
        self.maze.reset() # 迷路環境のリセット

        done = False
        t=0
        for i in range(self.timesteps):
            obs = self.maze.get_observation()
            action = controller.activate(obs)
            done = self.maze.update(action)
            t +=1
            
            if done:
                break

        # 適応度の計算
        if done: #ゴールについた場合
            #通常
            #score = 1.0
            #正規化
            #score = 1.0 + (self.timesteps - t) / self.timesteps
            #割ったもの
            score = 1.0 / t
            
        else: # ゴールにつかなかった場合
            # 距離を求める
            distance = self.maze.get_distance_to_exit()
            # 適応度 = （初期状態の距離 - 評価対象の距離） / 初期状態の距離
            #score = (self.maze.initial_distance - distance) / (self.maze.initial_distance)
            #timestepを正規化したものを足したもの
            #score = (self.maze.initial_distance - distance) / (self.maze.initial_distance) + 0.0
            #timestepで割ったもの
            score = (self.maze.initial_distance - distance) / (self.maze.initial_distance * t) 
            

        last_loc = self.maze.get_agent_location() # エージェントの位置をとる
        results = {
            'fitness': score,
            'data': last_loc,
            'timestep' : t
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
