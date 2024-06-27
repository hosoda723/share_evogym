import numpy as np
import torch

from evogym import get_full_connectivity

from neat_cppn import BaseCPPNDecoder, BaseHyperDecoder


class EvogymStructureDecoder(BaseCPPNDecoder):
    def __init__(self, size):
        self.size = size # ロボットサイズ

        # 配列座標の作成，[0列目のxまたはｙ座標],[1列], [2列],・・・の形での出力，転置されたイメージ
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
        x = x.flatten() # x座標列の１次元化
        y = y.flatten() # y座標配列の１次元化

        center = (np.array(size) - 1) / 2 # 行列の中心座標を求める
        # 配列のすべての要素と中心の要素との距離を測っている，三平方の定理
        d = np.sqrt(np.square(x - center[0]) + np.square(y - center[1])) 

        self.inputs = np.vstack([x, y, d]).T # [x座標, y座標, 中心からの距離]の形のn行3列の配列

    def decode(self, genome, config):
        """_summary_
        ネットワークの出力からボクセルの特徴を選択し，evogymでロボットとして構成できる形に
        変形しボクセルの位置(ロボットの形状)と特徴，ボクセルのつながりを返す

        Args:
            genome (_type_): 
            config (_type_): 設定

        Returns:
            dict: {'body':形状と特徴, 'conections':ボクセルのつながり}
        """
        # [empty, rigid, soft, vertical, horizontal] * (robot size)
        # ロボットのボクセルの数分[empty, rigid, soft, vertical, horizontal]が返る
        output = self.feedforward(self.inputs, genome, config)
        # 各ボクセルの特徴を選択
        material = np.argmax(output, axis=1) # output配列の行ごとの最大値のインデックスを出力(一番最初の値が出る)

        body = np.reshape(material, self.size) # evogymのロボットの設定ファイルと同じ形に変形
        connections = get_full_connectivity(body) # bodyのつながりを調べる
        return {'body': body, 'connections': connections}


class EvogymHyperDecoder(BaseHyperDecoder):
    def __init__(self, substrate, use_hidden=False, activation='sigmoid'):

        self.activation = activation

        connections = [('input', 'output')]
        downstream_nodes = ['output']

        substrate.make_substrate()
        if use_hidden:
            substrate.add_hidden('hidden', 1)
            connections.extend([('input', 'hidden'), ('hidden', 'output')])
            downstream_nodes.append('hidden')

        substrate.make_vectors()

        self.set_attr(substrate, connections, downstream_nodes)
