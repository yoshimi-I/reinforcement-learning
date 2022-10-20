import numpy as np
from collections import defaultdict
import copy
# 環境の実装
class GridWorld:
    def __init__(self):
        #行動は4種類(それぞれを数字に置き換える)
        self.action_list = [0,1,2,3]
        # 辞書型で行動を紐づける
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
        # 以下にマップを作成する
        # それぞれの値はその場所の報酬を指す
        self.reward_map = np.array(
            [[0,0,0,1],
            [0,None,0,-1],
            [0,0,0,0]]
        )
        #以下に障害物を記載し、その座標を保持
        self.goal = (0,3)
        self.start = (2,0)
        self.wall = (1,1)
        #エージェントの場所の配列を保持(スタート値で初期化)
        self.agent_state = self.start

    # 以下にメソッドを実装
    # マップの情報を取得(主に再代入不可能のgetterの役割)
    # 高さを返却
    @property
    def height(self):
        return len(self.reward_map)
    # 横幅を返却
    @property
    def width(self):
        return len(self.reward_map[0])

    #行動の種類を返却
    @property
    def actions(self):
        return self.action_list

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)


    # 以下にエージェントを動かすための関数と報酬関数を実装する。
    def next_step(self,state,action):
        # 上下左右の行動をそれぞれ座標に足し合わせることで実装
        action_list = [(-1,0),(1,0),(0,-1),(0,1)]
        # 0~3までの数字を配列に入れることで行動となる数値を取得
        move = action_list[action]
        # エージェントを移動させる(moveを現在の状態の座標に足し合わせる)
        # next_state = (y座標,x座標)となる
        next_state = (state[0] + move[0],state[1] + move[1])
        new_y,new_x = next_state

        # 移動したのちそれがマップから外れてるかどうかの確認処理(外れていたら更新を中断)
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height or next_state == self.wall:
            next_state = state
        return next_state
    # 報酬関数
    def reward(self,state,action,next_state):
        return self.reward_map[next_state]

# 以下に反復方策評価の実装関数を書く
def eval_step(pi,V,env,gamma=0.9):
    # stateで全ての座標にアクセスして価値関数を更新していく(最後までエピソードが終わった時の報酬がその場所の価値関数になる)
    for state in env.states():#(0,0),(0,1)...
        # アクセスした座標がゴールだった場合は価値関数は0なので0にする
        if state == env.goal:
            V[state] = 0
            continue
        action_probs = pi[state] #{0:0.25,1:0.25,2:0.25,3:0.25}
        new_V = 0

        # 各座標と方策を取得
        for action,action_prob in action_probs.items():# action=0 action_prob=0.25
            next_state = env.next_step(state,action)
            r = env.reward(state,action,next_state)
            # 価値関数を更新していく(動的計画法)
            new_V += action_prob * (r+gamma*V[next_state]) # <-これがベルマン方程式
        V[state] = new_V
    return V


# 以上の関数を繰り返し行うことで(更新する数値がある値よりも少なくなった場合)状態価値関数が決定する
def policy_eval(pi,V,env,gamma,check_value = 0.001):
    while True:
        # まずは以前使った状態価値関数を保持する(更新後と更新前で比較)
        # また参照わたしにならないように値渡しを用いて(copy)保持
        old_V = copy.copy(V)
        V = eval_step(pi,V,env,gamma)

        # 更新された量の最大値を求める
        update_value = 0
        for state in V.keys(): #state=(0,0),(0,1)...
            t = abs(V[state] - old_V[state]) # 更新した値の絶対値を保持
            if update_value < t:
                update_value = t
            # 更新された値がcheck_valueより小さかったらループを抜ける
        if update_value < check_value:
            break
    return V

# 実際に状態価値関数を実装する
env = GridWorld()
gamma = 0.9 #割引率
pi = defaultdict(lambda: {0:0.25,1:0.25,2:0.25,3:0.25})
V = defaultdict(lambda :0)
V = policy_eval(pi,V,env,gamma)
print(V)