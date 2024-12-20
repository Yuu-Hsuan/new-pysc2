### 1.匯入
```
# 匯入 namedtuple：是一種可以像普通物件一樣操作的元組，用於定義簡單的不可變數據結構，方便存取其屬性
from collections import namedtuple

# 匯入 NumPy：是一個數值運算庫，支援多維數組操作，常用於機器學習和科學計算
import numpy as np

# 匯入 PySC2 的模組
from pysc2.lib import actions # actions：包含《星海爭霸 II》中可以執行的動作
from pysc2.lib import features # features：包含環境特徵的相關資訊（如畫面特徵、玩家特徵等）
```

### 2.定義 `FlatFeature` 與相關特徵

```
'''
    FlatFeature：一個用來存放特徵資料的簡單結構，包含以下四個屬性：
      1. index：特徵的索引
      2. type：特徵的類型（如標量）
      3. scale：特徵的縮放比例
      4. name：特徵名稱
'''
FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name']) # 定義 FlatFeature 結構，包含索引 (index)、類型 (type)、縮放比例 (scale)、名稱 (name)

#------------------------------------------------------------------------------------
# 計算動作數量：actions.FUNCTIONS 是 PySC2 提供的所有可用動作，此行代碼計算可用動作的總數
NUM_FUNCTIONS = len(actions.FUNCTIONS) 

#------------------------------------------------------------------------------------
# 提取玩家 ID 的縮放比例：features.SCREEN_FEATURES.player_id.scale 定義了畫面中特徵 "玩家 ID" 的最大值（用於縮放）、用於標準化數值
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale

#------------------------------------------------------------------------------------
# 定義玩家的標量特徵：每個 FlatFeature 都表示玩家某一個屬性的特徵，例如資源（礦物、氣體）、人口（食物使用量、空間等）
FLAT_FEATURES = [
  FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),  # 玩家 ID
  FlatFeature(1,  features.FeatureType.SCALAR, 1, 'minerals'),  # 礦物數量
  FlatFeature(2,  features.FeatureType.SCALAR, 1, 'vespene'),  # 氣體數量
  FlatFeature(3,  features.FeatureType.SCALAR, 1, 'food_used'),  # 已用人口
  FlatFeature(4,  features.FeatureType.SCALAR, 1, 'food_cap'),  # 人口上限
  FlatFeature(5,  features.FeatureType.SCALAR, 1, 'food_army'),  # 軍事人口
  FlatFeature(6,  features.FeatureType.SCALAR, 1, 'food_workers'),  # 工人人口
  FlatFeature(7,  features.FeatureType.SCALAR, 1, 'idle_worker_count'),  # 空閒工人數量
  FlatFeature(8,  features.FeatureType.SCALAR, 1, 'army_count'),  # 軍隊數量
  FlatFeature(9,  features.FeatureType.SCALAR, 1, 'warp_gate_count'),  # 傳送門數量
  FlatFeature(10, features.FeatureType.SCALAR, 1, 'larva_count'),  # 幼蟲數量
]
```

### 3.判斷空間動作類型
```
# 建立字典 is_spatial_action：判斷動作是否需要空間參數（如在小地圖或畫面中選擇位置），actions.TYPES 提供了動作的參數類型
is_spatial_action = {} # 建立一個字典來存放動作類型是否為空間動作的標記
for name, arg_type in actions.TYPES._asdict().items(): # 遍歷動作參數類型，將動作名稱和其類型進行映射
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2'] #  # 如果動作涉及小地圖或畫面，標記為空間動作
```

### 4.定義輔助函數
功能：接受一個字典列表 lst，對每個鍵的數值（NumPy 陣列）進行堆疊，返回合併後的字典
```
def stack_ndarray_dicts(lst, axis=0):
  """Concatenate ndarray values from list of dicts
  along new axis.(將多個字典的 ndarray 值按指定軸疊加)"""
  res = {} # 初始化結果字典
  for k in lst[0].keys(): # 遍歷列表中第一個字典的所有鍵
    res[k] = np.stack([d[k] for d in lst], axis=axis) # 將每個字典對應鍵的值按指定軸疊加，存入結果字典
  return res # 返回結果字典
```
### 5.定義 `Preprocessor` 類別
```
class Preprocessor():
  """Compute network inputs from pysc2 observations.

  See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
  for the semantics of the available observations.(負責將 PySC2 的觀察值轉換為神經網路的輸入)
  """
#------------------------------------------------------------------------------------
# 初始化方法
  '''
     1. obs_spec：
          環境的觀察規範，描述了環境輸出的數據結構
     2. 定義輸入特徵的維度：
          畫面特徵（screen_channels）
          小地圖特徵（minimap_channels）
          標量特徵（flat_channels）
          可用動作（available_actions_channels）
  '''
  def __init__(self, obs_spec): # 初始化方法，接收環境的觀察規範 obs_spec
    self.screen_channels = len(features.SCREEN_FEATURES) # 畫面特徵的通道數
    self.minimap_channels = len(features.MINIMAP_FEATURES)  # 小地圖特徵的通道數
    self.flat_channels = len(FLAT_FEATURES)  # 標量特徵的通道數
    self.available_actions_channels = NUM_FUNCTIONS  # 可用動作的通道數

#------------------------------------------------------------------------------------
# 獲取輸入通道數:返回每種特徵的通道數，用於神經網路設計
  def get_input_channels(self):
    """Get static channel dimensions of network inputs.(返回靜態通道數資訊)"""
    return {
        'screen': self.screen_channels,  # 畫面特徵通道數
        'minimap': self.minimap_channels,  # 小地圖特徵通道數
        'flat': self.flat_channels,  # 標量特徵通道數
        'available_actions': self.available_actions_channels}  # 可用動作的通道數

#------------------------------------------------------------------------------------
# 預處理觀察值:接受觀察值列表，逐一預處理後，將結果堆疊起來
  def preprocess_obs(self, obs_list):
    return stack_ndarray_dicts(
        [self._preprocess_obs(o.observation) for o in obs_list])
#------------------------------------------------------------------------------------
# 預處理單個觀察值
  '''
     功能：
          建立可用動作的 One-Hot 向量
          預處理畫面和小地圖數據
          合併玩家的標量特徵
          將這些特徵組裝為字典返回

  def _preprocess_obs(self, obs):
    """Compute screen, minimap and flat network inputs from raw observations.(將單個觀察值轉換為畫面、小地圖、標量及可用動作特徵)
    """
    available_actions = np.zeros(NUM_FUNCTIONS, dtype=np.float32) # 初始化一個全為 0 的 One-Hot 向量，表示可用動作
    available_actions[obs['available_actions']] = 1 # 將觀察值中可用動作的索引設為 1

    screen = self._preprocess_spatial(obs['screen'])  # 預處理畫面特徵
    minimap = self._preprocess_spatial(obs['minimap']) # 預處理小地圖特徵

    flat = np.concatenate([ 
        obs['player']])  # 將玩家的標量特徵組合成一個向量
        # TODO available_actions, control groups, cargo, multi select, build queue(還需處理控制組、建造隊列等其他數據)

    return {
        'screen': screen,  # 返回畫面特徵
        'minimap': minimap,  # 返回小地圖特徵
        'flat': flat,  # 返回標量特徵
        'available_actions': available_actions} # 返回可用動作的 One-Hot 向量
#------------------------------------------------------------------------------------
# 預處理空間數據:將空間數據的維度順序從 [特徵, 高度, 寬度] 轉換為 [高度, 寬度, 特徵]，方便神經網路處理
  def _preprocess_spatial(self, spatial):
    return np.transpose(spatial, [1, 2, 0]) # 使用 NumPy 的轉置操作，將特徵維度移至最後
```
