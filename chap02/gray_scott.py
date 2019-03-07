#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.cm as cm

# シミュレーションの各パラメタ
SPACE_GRID_SIZE = 256
dx = 0.01
dt = 1
VISUALIZATION_STEP = 172  # 何ステップごとに画面を更新するか。 # どうもimshowの実行が遅いので数値を大きくしている

# モデルの各パラメタ
Du = 2e-5
Dv = 1e-5
f, k = 0.04, 0.06  # amorphous
# f, k = 0.035, 0.065  # spots
# f, k = 0.012, 0.05  # wandering bubbles
# f, k = 0.025, 0.05  # waves
# f, k = 0.022, 0.051 # stripe

# 初期化
u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
v = np.zeros((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
# 中央にSQUARE_SIZE四方の正方形を置く
SQUARE_SIZE = 20
u[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.5
v[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.25
# 対称性を壊すために、少しノイズを入れる
u += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1
v += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE)*0.1

value_range_min=0
value_range_max=1

value_range = (value_range_min, value_range_max)

def frames(*args, **kwargs):
  global u, v, value_range, im
  for i in range(VISUALIZATION_STEP):
    # ラプラシアンの計算
    laplacian_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                    np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx*dx)
    laplacian_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                    np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx*dx)
    # Gray-Scottモデル方程式
    dudt = Du*laplacian_u - u*v*v + f*(1.0-u)
    dvdt = Dv*laplacian_v + u*v*v - (f+k)*v
    u += dt * dudt
    v += dt * dvdt
    matrix = u
    matrix[matrix < value_range[0]] = value_range[0]
    matrix[matrix > value_range[1]] = value_range[1]
  img = ((matrix.astype(np.float64) - value_range[0]) / (value_range[1] - value_range[0]) * 255).astype(np.uint8)
  return [plt.imshow(img, interpolation='none')]


fig = plt.figure()
anim = animation.FuncAnimation(fig, frames, interval=100)
plt.show()
