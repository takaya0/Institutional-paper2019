# 必要なモジュールのインポート
import numpy as np
import pydotplus


def neuralViz(shape, fileName='NN.png', plot=True, Wlist=None):

    # 形状をnp配列に変換
    args = np.array(shape)

    # 空のmatrixを作成（adjacency_matrix作成用）
    matrix = np.zeros((args.sum(), args.sum()))

    # 結合の強さに関するmatrixを取得したいときに使います。（本記事では未使用）
    if Wlist is not None:
        wmatrix = np.zeros((args.sum(), args.sum()))

    # 結ぶノード間に１を入力
    for i in range(len(args)-1):
        tmp1 = args[:i+1].sum()
        tmp0 = args[:i].sum()
        matrix[tmp1:tmp1+args[i+1], tmp0:tmp0+args[i]] = 1

        if Wlist is not None:
            wmatrix[tmp1:tmp1+args[i+1], tmp0:tmp0+args[i]] = Wlist[i].data

    # GraphVizの機能を使って描画
    g = pydotplus.graph_from_adjacency_matrix(matrix.T.tolist(), node_prefix=0)

    # 何もしないとNodeが作られないので足す。ついでに文字を消す。
    for i in range(args.sum()):
        n = pydotplus.Node(i+1)
        n.set_fontsize(0)
        g.add_node(n)

    return g
    if plot == True:
        # グラフ書き出し（'default.png'）
        g.write_png(fileName, prog='dot')

    else:
        return g, matrix.T, wmatrix.T


g = neuralViz([3, 4, 5, 4, 3, 1])
# ノード一覧を取得
nodeList = g.get_node_list()
# 適当なノードを選択しいろいろ編集
for i in range(len(nodeList)):
    node = nodeList[i]
    node.set_color('royalblue')
    node.set_style('filled')
# 書き出し
g.write_png("../Images/NN.png", prog='dot')
