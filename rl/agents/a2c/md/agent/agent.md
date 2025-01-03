# A2Cagent 說明與逐行解析

此文件詳細說明 `agent.py` 的功能與實現，並逐行解釋每段程式碼的用途和邏輯。

---

## 導入模組
```python
import os

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Categorical

from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.networks.fully_conv import FullyConv
from rl.util import safe_log, safe_div
```
### **解釋:**

#### 1.`import os`:
- 用於進行文件與目錄的操作，如模型保存和加載。
#### 2.`tensorflow`:
- 用於構建和訓練神經網絡。
- `layers` 提供方便的神經網絡層構建工具。
- `Categorical`: 定義離散分佈，用於動作採樣。
#### 3.`pysc2.lib.actions`:
-導入動作參數的類型集合。
#### 4.`rl.networks.fully_conv`:
-導入 `FullyConv` 類別，構建卷積網絡。
#### 5.`rl.util`:
導入工具函數 `safe_log` 和 `safe_div`，分別用於安全取對數和除法。
## A2CAgent 類別
```python
class A2CAgent():
    """A2C agent.

    Run build(...) first, then init() or load(...).
```
### **解釋:**

#### 1.`class A2CAgent`:
- 定義 A2C 強化學習代理。
- 用於執行網絡構建（`build`）、初始化（`init`）或從檔案加載模型（ `load`）。
## 初始化方法
```python
def __init__(self,
             sess,
             network_cls=FullyConv,
             network_data_format='NCHW',
             value_loss_weight=0.5,
             entropy_weight=1e-3,
             learning_rate=7e-4,
             max_gradient_norm=1.0,
             max_to_keep=5):
    self.sess = sess
    self.network_cls = network_cls
    self.network_data_format = network_data_format
    self.value_loss_weight = value_loss_weight
    self.entropy_weight = entropy_weight
    self.learning_rate = learning_rate
    self.max_gradient_norm = max_gradient_norm
    self.train_step = 0
    self.max_to_keep = max_to_keep
```
### **解釋:**

#### 1.功能:
- 初始化 A2C 代理，設置網絡、優化器和訓練相關參數。
#### 2.參數:
- `sess`：TensorFlow 會話，用於執行計算圖。
- `network_cls`：網絡類型，默認為 `FullyConv`。
- `network_data_format`：數據格式，支持 `NCHW` 和 `NHWC`。
- `value_loss_weight`：價值損失的權重。
- `entropy_weight`：策略熵的權重，用於鼓勵探索。。
- `learning_rate`：學習率。
- `max_gradient_norm`：梯度裁剪的最大範圍。
- `max_to_keep`：保存的最大檢查點數量。
## `build` 方法
```python
def build(self, static_shape_channels, resolution, scope=None, reuse=None):
    self._build(static_shape_channels, resolution)
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    self.saver = tf.train.Saver(variables, max_to_keep=self.max_to_keep)
    self.init_op = tf.variables_initializer(variables)
    train_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    self.train_summary_op = tf.summary.merge(train_summaries)
```
### **解釋:**

#### 1.功能:
- 構建 TensorFlow 訓練圖，初始化變量和摘要操作。
#### 2.參數:
- 調用 `_build` 方法構建核心網絡。
- 收集所有全局變量，並使用 `Saver` 進行模型保存和加載。
- 初始化變量。
- 收集訓練摘要，合併為單一操作。
## `build` 方法
```python
def _build(self, static_shape_channels, resolution):
    ...
    policy, value = self.network_cls(data_format=self.network_data_format).build(
        screen, minimap, flat)
    self.policy = policy
    self.value = value
    ...
    log_probs = compute_policy_log_probs(available_actions, policy, actions)
    policy_loss = -tf.reduce_mean(advs * log_probs)
    value_loss = tf.reduce_mean(tf.square(returns - value) / 2.)
    entropy = compute_policy_entropy(available_actions, policy, actions)
    loss = (policy_loss
            + value_loss * self.value_loss_weight
            - entropy * self.entropy_weight)
    ...
    self.train_op = layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=opt,
        clip_gradients=self.max_gradient_norm,
        learning_rate=None,
        name="train_op")
    self.samples = sample_actions(available_actions, policy)
```
### **解釋:**

#### 1.功能:
- 定義核心網絡結構、損失函數和訓練操作。
#### 2.網路購建:
- 通過 `FullyConv` 類構建策略（`policy`）和價值（`value`）網絡。
#### 3.損失計算:
- 策略損失（`policy_loss`）：基於優勢值和動作概率的負對數。
- 價值損失（`value_loss`）：目標回報與估計價值的平方差。
- 熵（`entropy`）：鼓勵策略分佈的多樣性。
- 總損失：
```python
loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight
```
#### 4.優化器:
- 使用 RMSProp 優化器，並對梯度進行裁剪。
#### 5.動作採樣:
- 通過 `sample_actions` 從策略中採樣動作。
## `train` 方法
```python
def train(self, obs, actions, returns, advs, summary=False):
    feed_dict = {
        self.screen: obs["screen"],
        self.minimap: obs["minimap"],
        self.flat: obs["flat"],
        self.available_actions: actions[0],
        self.selected_action: actions[1],
        self.selected_argument: actions[2],
        self.returns: returns,
        self.advs: advs,
    }

    fetches = [self.train_op, self.policy_loss, self.value_loss, self.entropy]
    if summary:
        fetches.append(self.train_summary_op)

    results = self.sess.run(fetches, feed_dict=feed_dict)
    self.train_step += 1
    return results
```
### **解釋:**

#### 1.功能:
- 執行一次訓練步驟，更新模型參數，並返回損失和熵的結果。
#### 2.邏輯:
- 準備數據：
  -  創建 `feed_dict`，將觀測數據 (`obs`)、動作 (`actions`)、回報 (`returns`) 和優勢 (`advs`) 對應到 TensorFlow 圖中的佔位符。
- 定義提取項：
  -  `fetches` 包括訓練操作 (`train_op`)、策略損失 (`policy_loss`)、價值損失 (`value_loss`) 和策略熵 (`entropy`)。
  -  如果需要摘要，將 `train_summary_op` 也加入。
- 執行訓練：
  -  使用 TensorFlow 會話 (`self.sess`) 運行 `fetches`，更新模型參數，並獲取計算結果。
- 更新訓練步驟：
  -  自增 `self.train_step`，表示已完成的訓練迭代次數。
## `step` 方法
```python
def step(self, obs):
    feed_dict = {
        self.screen: obs["screen"],
        self.minimap: obs["minimap"],
        self.flat: obs["flat"],
        self.available_actions: obs["available_actions"],
    }

    fetches = [self.samples, self.value]
    results = self.sess.run(fetches, feed_dict=feed_dict)
    return results
```
### **解釋:**

#### 1.功能:
- 執行一次推斷，基於當前觀測數據生成動作和價值估計。
#### 2.邏輯:
- 準備數據：
  -  創建 `feed_dict`，將當前觀測數據映射到佔位符。
- 提取動作和價值：
  -  使用 TensorFlow 圖中定義的 `self.samples`（動作採樣）和 `self.value`（價值估計）進行推斷。
- 返回結果：
  -  返回採樣的動作和對應的價值估計。
## `save` 方法
```python
def save(self, path, global_step=None):
    self.saver.save(self.sess, path, global_step=global_step)
```
### **解釋:**

#### 1.功能:
- 保存當前模型的參數到檔案。
#### 2.邏輯:
- 調用 `self.saver.save` 方法，將模型參數保存到指定路徑。
- 如果提供了 `global_step`，會將步數附加到檔案名中。
## `load` 方法
```python
def load(self, path):
    self.saver.restore(self.sess, path)
```
### **解釋:**

#### 1.功能:
- 從指定的檔案路徑加載模型參數。
#### 2.邏輯:
- 使用 `self.saver.restore` 方法恢復保存的模型權重到當前 TensorFlow 圖中。
## 工具函數完整解析

## `compute_policy_log_probs` 函數
```python
def compute_policy_log_probs(available_actions, policy, actions):
    mask = tf.cast(available_actions, dtype=tf.float32)
    masked_policy = tf.multiply(policy, mask)
    normalized_policy = tf.divide(masked_policy, tf.reduce_sum(masked_policy, axis=1, keepdims=True))
    selected_policy = tf.gather_nd(normalized_policy, actions)
    log_probs = tf.log(selected_policy + 1e-8)
    return log_probs
```
### **解釋:**

#### 1.功能:
- 計算所選動作的對數概率，用於策略損失計算。
#### 2.邏輯:
- 動作遮罩：
  -  根據 `available_actions`，遮蔽無效動作，確保其概率為零。
- 歸一化：
  -  使用 `tf.divide` 正規化動作概率，確保其和為 1。
- 提取所選動作的概率：
  -  使用 `tf.gather_nd` 提取動作參數對應的概率值。
- 取對數：
  -  使用 `tf.log` 計算對數概率，並加上小常數避免取對數時出現無窮大。
## `sample_actions` 函數
```python
def sample_actions(available_actions, policy):
    mask = tf.cast(available_actions, dtype=tf.float32)
    masked_policy = tf.multiply(policy, mask)
    normalized_policy = tf.divide(masked_policy, tf.reduce_sum(masked_policy, axis=1, keepdims=True))
    sampled_actions = tf.random.categorical(tf.math.log(normalized_policy), num_samples=1)
    return sampled_actions
```
### **解釋:**

#### 1.功能:
- 基於策略分佈採樣動作，用於推斷和環境交互。
#### 2.邏輯:
- 遮罩和歸一化：
  -  與 `compute_policy_log_probs` 相同，遮蔽無效動作並正規化策略分佈。
- 動作採樣：
  -  使用 `tf.random.categorical` 從對數分佈中隨機採樣動作，模擬代理的決策。
