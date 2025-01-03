## 說明
該程式碼實現了一個深度學習網絡，
旨在處理《星际争霸 II》中多模態（多種輸入格式）的遊戲狀態，
並生成行為策略（policy）和狀態價值（value）。
多模態嵌入：利用卷積、全連接層和 one-hot 編碼處理不同數據模態。
特徵融合：空間特徵與平坦特徵融合，捕捉全局與局部信息。
策略生成：根據遊戲需求輸出空間和非空間行為策略。
價值估計：通過全連接層生成當前狀態的價值評估。

## 解析
### 1.
* 這是一個用來玩《星際爭霸 II》的深度學習網路，
* 負責分析遊戲的狀態（像是地圖和資源）並產生下一步行動的建議。
```
import numpy as np
import tensorflow as tf

import keras
from keras import layers


from pysc2.lib import actions
from pysc2.lib import features

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
```

### FullyConv 類別
* 定義一個卷積神經網絡，用於處理 StarCraft II 的遊戲狀態並生成策略。
```
class FullyConv():
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def __init__(self, data_format='NCHW'):
    self.data_format = data_format
  # 初始化class的數據格式，指定格式為 NCHW [Batch, Channels, Height, Width]

  def embed_obs(self, x, spec, embed_fn):
  # 把遊戲中的特徵（例如：地圖資訊）轉換成適合神經網路的資料。
  
  # x: 輸入數據張量，地圖的數字陣列。[Batch, Features]
  # spec: 特徵規範列表，每個資料的規格清單，說明資料的類型。
  # embed_fn: 嵌入函數，用於處理"類別型"特徵。
  
    feats = tf.split(x, len(spec), -1)
    # 將輸入數據 x 按照特徵數量 len(spec) 進行分割，每列對應一個特徵
    
    out_list = []# 存放每段處理後的資料。。
    for s in spec: # 遍歷每個特徵規範。對每個特徵進行處理。
      f = feats[s.index]
      if s.type == features.FeatureType.CATEGORICAL: 
      # 如果這段資料是分類型（例如：地形類型），我們用 one-hot 編碼來處理。
        dims = np.round(np.log2(s.scale)).astype(np.int32).item()
        dims = max(dims, 1)# 決定嵌入的維度大小。
        indices = tf.one_hot(tf.to_int32(tf.squeeze(f, -1)), s.scale)
        out = embed_fn(indices, dims)# 把分類資訊變成更好理解的高維度數字。
        # dims: 計算嵌入維度（對縮放比例取對數）。
        # indices: 將特徵進行 one-hot 編碼。
        # embed_fn: 將 one-hot 編碼結果進一步嵌入到更高維度表示。
        
      elif s.type == features.FeatureType.SCALAR:
      # 如果是標量數據（像是金礦的數量），就用數學公式縮小範圍，讓它更容易學習。
        out = self.log_transform(f, s.scale)
      # 對標量型特徵應用對數變換。使用對數變換標準化標量數據。
      
      else:
        raise NotImplementedError
      out_list.append(out)# 把處理好的資料存起來。
    return tf.concat(out_list, -1) # 把所有的資料合併成一個張量輸出。
  """
  步驟解析
  使用 tf.split 根據 spec 中特徵數量拆分輸入數據，每部分對應一個特徵。
  初始化 out_list 保存處理後的特徵。
  
  遍歷 spec，根據特徵類型進行處理：
  1.類別型特徵：
  計算嵌入維度 dims，使用 one-hot 編碼生成 indices。
  使用 embed_fn 將 indices 嵌入為高維表示。
  2.標量型特徵：
  使用 self.log_transform 進行對數變換。
  3.其他類型：
  拋出 NotImplementedError 異常。
  
  將嵌入的特徵添加到 out_list。
  使用 tf.concat 合併所有嵌入特徵。
  返回值
  返回嵌入後的特徵張量，形狀為 [Batch, Embedding_Dimensions]。
  """
  
  def log_transform(self, x, scale):
    return tf.log(x + 1.)
  # 對輸入 x 應用對數變換，增強小值的影響，縮小大值的範圍。
  """
  x: 輸入數據張量。
  scale: 標量的尺度參數（未在具體代碼中直接使用）。
  
  步驟解析
  對輸入數據 x 添加常數 1，然後取自然對數。
  返回值
  返回對數變換後的數據，形狀與輸入相同。

  將數據取對數，把大的數字壓縮，小的數字拉大，
  就像是把一條很長的橡皮筋拉成比較平均的樣子，
  讓電腦更容易學習。
  """
  def embed_spatial(self, x, dims):
    x = self.from_nhwc(x)# 把資料格式轉成網路方便處理的樣子
    out = layers.conv2d(
        x, dims,
        kernel_size=1,# 1x1 的小濾鏡，專門提取每個位置的特徵。较大的卷积核（例如 5x5）可以捕获更大范围的信息，适合处理地形特征。
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format)
    return self.to_nhwc(out) # 把資料格式轉回來。

  """
  用卷積處理 "空間" 數據，像是地圖的畫面，
  就像用濾鏡提取圖片中的重要部分。
      
  輸入參數
  x: 輸入的地圖數據，空間特徵張量，形狀通常為 [Batch, Height, Width, Channels]。
  dims: 目標嵌入的通道數。輸出的維度（圖片中要提取多少特徵）
  步驟解析
  將輸入從 NHWC 格式轉換為類內配置的數據格式。
  使用 1x1 卷積進行空間特徵的嵌入，激活函數為 ReLU。
  將卷積輸出的數據轉回 NHWC 格式。
  返回值
  嵌入後的空間特徵，形狀為 [Batch, Height, Width, dims]。

  """
  def embed_flat(self, x, dims):
    return layers.fully_connected(
        x, dims,
        activation_fn=tf.nn.relu)
  # 使用全連接層處理平坦特徵。增加输出维度可以提升策略和价值的表示能力。
  # dims: 全連接層的輸出維度。
  """
  處理 "平坦" 的特徵，像是遊戲中沒有空間感的資訊（例如：礦工的數量）。
  用全連接層把這些數字轉成電腦好處理的高維表示。
  輸入參數
  x: 平坦特徵張量，形狀為 [Batch, Features]。
  dims: 全連接層的輸出維度。
  步驟解析
  應用全連接層將平坦特徵嵌入更高維度，激活函數為 ReLU。
  返回值
  嵌入後的平坦特徵，形狀為 [Batch, dims]。

  """
  def input_conv(self, x, name):
    conv1 = layers.conv2d(
        x, 16,
        kernel_size=5,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv1" % name)
    conv2 = layers.conv2d(
        conv1, 32,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv2" % name)
    return conv2
  # 通過兩層卷積提取輸入的高級特徵。
  # conv1: 16 個輸出通道，5x5 卷積。
  # conv2: 32 個輸出通道，3x3 卷積。
  """
  用兩層卷積提取資料中的重要特徵。
  第一層像是一個大範圍的 "放大鏡"，提取粗略的特徵。
  第二層則是更細膩的 "顯微鏡"，挖掘細節。
  
  輸入參數
  x: 輸入特徵張量，形狀為 [Batch, Channels, Height, Width] 或 [Batch, Height, Width, Channels]。
  name: 卷積層的名稱前綴。
  步驟解析
  應用第一層卷積（16 通道，5x5 核）。
  將第一層輸出傳入第二層卷積（32 通道，3x3 核）。
  卷積層均使用 ReLU 激活函數。
  返回值
  返回經過兩層卷積處理的特徵張量，形狀為 [Batch, 32, Height, Width] 或對應的 NHWC 格式。

  """
  
  def non_spatial_output(self, x, channels):
    logits = layers.fully_connected(x, channels, activation_fn=None)
    return tf.nn.softmax(logits)
  # 使用全連接層生成非空間輸出的概率分佈。
  """
  處理遊戲中非空間的輸出，像是選擇某個動作（例如：進攻、移動）
  它會計算每個動作的可能性，然後用 softmax 讓它變成概率。
  
  輸入參數
  x: 平坦特徵張量，形狀為 [Batch, Features]。
  channels: 輸出概率分佈的通道數。
  步驟解析
  應用全連接層生成 logits。
  使用 softmax 將 logits 轉換為概率分佈。
  返回值
  返回非空間行為的概率分佈，形狀為 [Batch, channels]。
  """
  def spatial_output(self, x):
    logits = layers.conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None,
                           data_format=self.data_format)
    logits = layers.flatten(self.to_nhwc(logits))
    return tf.nn.softmax(logits)
  # 使用 1x1 卷積生成空間行為的概率分佈。
  """
  處理空間輸出，像是選擇地圖上具體的位置（例如：在哪裡放建築）。
  用 1x1 卷積計算每個位置的重要性，再轉成概率。
  """
  def concat2d(self, lst):
    if self.data_format == 'NCHW':
      return tf.concat(lst, axis=1)
    return tf.concat(lst, axis=3)
"""
把多個 2D 的資料拼接在一起，就像拼拼圖一樣，
根據資料格式選擇拼接的方向。

輸入參數

lst: 包含多個 2D 張量的列表，這些張量將被拼接。
該方法基於當前的數據格式（data_format）選擇拼接的維度。
步驟解析

判斷數據格式：
如果 data_format 是 NCHW（[Batch, Channels, Height, Width]），則沿著通道維度（axis=1）進行拼接。
如果 data_format 是 NHWC（[Batch, Height, Width, Channels]），則沿著最後一維（通道維度，axis=3）拼接。
返回值

返回一個新的張量，包含所有輸入張量的拼接結果。
"""
  def broadcast_along_channels(self, flat, size2d):
    if self.data_format == 'NCHW':
      return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                     tf.stack([1, 1, size2d[0], size2d[1]]))
    return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                   tf.stack([1, size2d[0], size2d[1], 1]))
"""
把平坦的數據 "複製" 到整個地圖，
就像把一張平面的圖片鋪滿整個空間，讓它可以和地圖資訊結合。

輸入參數

flat: 1D 或 2D 張量，形狀為 [Batch, Features]，通常是平坦特徵。
size2d: 包含兩個整數值的列表或張量，表示目標 2D 張量的高和寬（[Height, Width]）。
步驟解析

NCHW 格式：
通過 tf.expand_dims 增加兩個新維度（[Batch, Features] -> [Batch, Features, 1, 1]）。
使用 tf.tile 將張量複製，擴展到指定的 size2d，生成形狀為 [Batch, Features, Height, Width] 的張量。
NHWC 格式：
與 NCHW 相似，但增加維度的位置不同，確保最終張量的形狀為 [Batch, Height, Width, Features]。
返回值

返回一個與輸入數據格式匹配的張量，該張量在空間維度上進行了廣播。
"""
  def to_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 2, 3, 1])
    return map2d
"""
把資料格式轉換成 NHWC，這是處理圖像時比較常見的格式。
輸入參數

map2d: 形狀為 [Batch, Channels, Height, Width] 或 [Batch, Height, Width, Channels] 的 4D 張量。
步驟解析

如果 data_format 是 NCHW：
使用 tf.transpose 將數據格式從 NCHW 轉換為 NHWC。
維度順序更改為 [Batch, Height, Width, Channels]。
如果 data_format 是 NHWC：
直接返回輸入張量，無需轉置。
返回值

返回符合 NHWC 格式的張量。
"""
  def from_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 3, 1, 2])
    return map2d
"""
把資料格式轉換成 NCHW，這是這個網路比較好處理的格式。

輸入參數

map2d: 形狀為 [Batch, Height, Width, Channels] 或 [Batch, Channels, Height, Width] 的 4D 張量。
步驟解析

如果 data_format 是 NCHW：
使用 tf.transpose 將數據格式從 NHWC 轉換為 NCHW。
維度順序更改為 [Batch, Channels, Height, Width]。
如果 data_format 是 NHWC：
直接返回輸入張量，無需轉置。
返回值

返回符合 NCHW 格式的張量。
"""
  def build(self, screen_input, minimap_input, flat_input):
  """
  screen_input 表示玩家视野范围内的屏幕信息，包括单位、资源、地形等
  minimap_input 表示小地图的全局视角信息，显示整个地图上单位的分布、资源点的位置等。
  flat_input 表示非空间特征的平面信息，如经济状况（矿物、气体）、单位数量、科技进度。
  建構整個網路，處理來自遊戲的畫面、地圖、以及平坦的數據，
  並生成行為策略（policy）和當前狀態的價值評估（value）。
  """
    size2d = tf.unstack(tf.shape(screen_input)[1:3])# 取得地圖的尺寸。宽度和高度。
    screen_emb = self.embed_obs(screen_input, features.SCREEN_FEATURES,
                                self.embed_spatial)
    # features.SCREEN_FEATURES 描述屏幕中可用的特征种类，比如资源点、单位种类、移动路径
    # 将游戏画面的特征（例如：单位、地形）转化为适合神经网络处理的“嵌入表示”。
    # 用滤镜把照片的颜色、光线特征提取出来，变成机器能理解的语言。
    minimap_emb = self.embed_obs(minimap_input, features.MINIMAP_FEATURES,
                                 self.embed_spatial)
    # features.MINIMAP_FEATURES 描述屏幕中可用的特征种类，比如资源点、单位种类、移动路径
    # 对小地图的数据做同样的特征提取。
    # 从地图上标记资源点、敌人位置，提炼成核心信息。
    flat_emb = self.embed_obs(flat_input, FLAT_FEATURES, self.embed_flat)
    # FLAT_FEATURES 列表中定义了所有非空间特征的范围和类型（标量或类别型）
    #如果需要更高精度的资源管理，可以增加额外的特征，如资源消耗率或战斗时间。
    
    # 对非空间的特征（如经济数据、当前时间）做特征提取。
    # 想象你记下了银行存款余额、账单信息，用数字形式记录下来。
    screen_out = self.input_conv(self.from_nhwc(screen_emb), 'screen')
    # 对画面特征进行深度卷积操作，提取更高级的特征。就像先用放大镜，再用显微镜观察每个细节。
    minimap_out = self.input_conv(self.from_nhwc(minimap_emb), 'minimap')
    # 对小地图特征做同样的操作。仔细分析整个地图，确保不会遗漏重要的资源或敌人。
    broadcast_out = self.broadcast_along_channels(flat_emb, size2d)
    # 将非空间数据（如经济特征）“广播”到整个地图上。
    # 想象你在地形图上标出所有城市的存款余额，把每个城市的资金状况分布到整个地图上。
    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])
    # 将画面、小地图和广播的特征组合在一起，形成完整的游戏状态
    # 像拼拼图，把分散的信息拼成一张完整的战场图。
    flat_out = layers.flatten(self.to_nhwc(state_out))
    # 将二维数据转化为一维，便于后续处理。
    # 把地图上的标记拉成一张清单，按行列列出每个位置的详情。
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)
    # 通过全连接层提炼出更紧凑、更关键的特征。
    # 像从书本中总结出要点，把长篇大论压缩成一个提纲。
    value = layers.fully_connected(fc, 1, activation_fn=None)
    # 预测当前状态的“价值”，即形势好坏的评分。
    # 就像评估棋局中的胜算，分数越高，胜率越大。
    value = tf.reshape(value, [-1])
    # 调整形状，确保输出符合预期。
    # 把评估结果从表格形式转成清单，方便进一步使用。
    fn_out = self.non_spatial_output(fc, NUM_FUNCTIONS)
    # NUM_FUNCTIONS 游戏中可执行的动作函数总数，如移动单位、建造建筑、攻击等
    # 根据全局特征，预测要执行的动作（如攻击、移动）。
    # 就像决定棋子下一步是前进还是攻击。

    args_out = dict()
    # 初始化一个字典，用来存储每个动作的参数。
    # 准备一个清单，用来记录“做什么”和“怎么做”。
    for arg_type in actions.TYPES:
    # 遍历所有动作参数的类型。
    # 逐一检查每种可能的指令（如坐标、目标单位）。
    # 判断这个参数是否需要空间信息。
      if is_spatial_action[arg_type]:# 如果是“在哪里建造”的问题，就需要地图坐标。
        arg_out = self.spatial_output(state_out)# 选择地图上最适合建造的位置
      else:
        arg_out = self.non_spatial_output(fc, arg_type.sizes[0])
        # 如果是“建造什么建筑”，就直接从选项中选一个。
      args_out[arg_type] = arg_out # 将参数的预测结果存入字典，把每个决策记录到清单中。

    policy = (fn_out, args_out)
    # 组合动作和参数，形成完整的策略。
    # 就像确定了棋子的行动和具体操作。

    return policy, value
  
    #返回策略和状态价值。
    #最后输出两项：下一步怎么走(policy)，以及现在形势是否有利(value)。
  
  """
  模型構建方法：build
  
  將屏幕、迷你地圖及平坦數據的輸入進行處理，構建完整的神經網絡。
  返回策略（policy）和價值（value）。
  
  內部步驟簡述:
  嵌入特徵 (embed_obs)。
  空間特徵經卷積 (input_conv)。
  平坦特徵與空間特徵廣播融合 (broadcast_along_channels)。
  平坦化並生成策略和價值。

  """


```
