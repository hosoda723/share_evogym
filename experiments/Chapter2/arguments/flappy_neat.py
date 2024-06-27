# コマンドライン引数の処理
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Maze NEAT experiment'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help='experiment name (default: "{task}")'
    )
    parser.add_argument(
        '-t', '--task',
        default='stage', type=str,
        help='maze name (default: stage, built on "envs/flappy/stage_files/")'
    )

    parser.add_argument(
        '-p', '--pop-size',
        default=500, type=int,
        help='population size of NEAT (default: 500)'
    )
    parser.add_argument(
        '-g', '--generation',
        default=500, type=int,
        help='iterations of NEAT (default: 500)'
    )

    parser.add_argument(
        '--timesteps',
        default=400, type=int,
        help='limit of timestep for solving maze (default: 400)'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=4, type=int,
        help='number of parallel evaluation processes (default: 4)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true', default=False,
        help='not open window of progress figure (default: False)'
    )
    args = parser.parse_args()

    if args.name is None:
        args.name = args.task

    return args

def get_figure_args():
    parser = argparse.ArgumentParser(
        description='make circuit figures'
    )

    parser.add_argument(
        'name',
        type=str,
        help='nam of experiment for making figures'
    )
    parser.add_argument(
        '-s', '--specified',
        type=int,
        help='input id, make figure for the only specified circuit (usage: "-s {id}")'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=1, type=int,
        help='number of parallel making processes (default: 1)'
    )
    parser.add_argument(
        '--not-overwrite',
        action='store_true', default=False,
        help='skip process if already gif exists (default: False)'
    )
    parser.add_argument(
        '--no-multi',
        action='store_true', default=False,
        help='do without using multiprocessing. if error occur, try this option. (default: False)'
    )

    args = parser.parse_args()

    assert args.name is not None, 'argumented error: input "{experiment name}"'

    return args
