
from evogym import is_connected, has_actuator, hashable

class EvogymStructureConstraint:
    def __init__(self, decode_function):
        self.decode_function = decode_function
        self.hashes = {} # 辞書の作成

    def eval_constraint(self, genome, config, generation):
        robot = self.decode_function(genome, config) # {'body':,'connection':}
        body = robot['body']
        validity = is_connected(body) and has_actuator(body) # つながりがあるか，アクチュエータがあるかの確認
        if validity:
            robot_hash = hashable(body) # robotを1次元化したものを文字列型にしたもの
            if robot_hash in self.hashes: # 辞書の中に評価したいロボットが既に存在しているかどうか
                validity = False # このロボットは評価しない
            else:
                self.hashes[robot_hash] = True # 辞書にロボット構造の文字列をkey，値をTrueとして追加

        return validity
