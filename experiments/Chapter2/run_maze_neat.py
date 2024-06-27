import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURR_DIR))

LIB_DIR = os.path.join(ROOT_DIR, 'libs')
sys.path.append(LIB_DIR)
import neat_cppn
from experiment_utils import initialize_experiment
from parallel import EvaluatorParallel

ENV_DIR = os.path.join(ROOT_DIR, 'envs', 'maze')
sys.path.append(ENV_DIR)
from evaluator import MazeControllerEvaluator
from maze_drawer import MazeReporterNEAT
from maze_environment_numpy import MazeEnvironment

from arguments.maze_neat import get_args

import numpy as np
import random

np.random.seed(123)
random.seed(123)

def main():
    args = get_args() # コマンドライン引数に関する処理

    save_path = os.path.join(CURR_DIR, 'out', 'maze_neat', args.name) # 実験結果の出力先

    initialize_experiment(args.name, save_path, args) # 実験の出力先の設定

    # MazeEnvironmentクラスのread_environment関数を使って設定ファイル(.txt)を読む
    maze_env = MazeEnvironment.read_environment(ROOT_DIR, args.task) 

    # FeedForwardNNを生成(libs/neat_cppn/feedfoward.py参照)
    decode_function = neat_cppn.FeedForwardNetwork.create

    # エージェントの評価を行う
    evaluator = MazeControllerEvaluator(maze_env, args.timesteps)
    evaluate_function = evaluator.evaluate_agent # エージェントを評価

    # 並列で評価
    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluate_function,
        decode_function=decode_function
    )

    # 設定ファイルの読み込み(.cfg)
    config_file = os.path.join(CURR_DIR, 'config', 'maze_neat.cfg')
    # 引数で渡された設定の上書き
    custom_config = [
        ('NEAT', 'pop_size', args.pop_size),
    ]
    config = neat_cppn.make_config(config_file, custom_config=custom_config)
    config_out_file = os.path.join(save_path, 'maze_neat.cfg')
    config.save(config_out_file)

    # 初期集団の生成
    pop = neat_cppn.Population(config)

    # 保存先の指定
    figure_path = os.path.join(save_path, 'progress')
    
    reporters = [
        neat_cppn.SaveResultReporter(save_path), # 進化結果の保存
        neat_cppn.StdOutReporter(True), # 表示？
        MazeReporterNEAT(maze_env, args.timesteps, figure_path, decode_function, args.generation, no_plot=args.no_plot)
    ]
    for reporter in reporters:
        pop.add_reporter(reporter) 


    try:
        pop.run(fitness_function=parallel.evaluate, n=args.generation) # 進化計算の実行
    finally:
        neat_cppn.figure.make_species(save_path)

if __name__=='__main__':
    main()
