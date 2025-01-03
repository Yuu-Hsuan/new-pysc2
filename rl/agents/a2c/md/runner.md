```
import numpy as np
import tensorflow as tf

from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.pre_processing import Preprocessor
from rl.pre_processing import is_spatial_action, stack_ndarray_dicts


class A2CRunner(): #a2c的agent與env互動
  def __init__(self,
               agent,
               envs,
               summary_writer=None, #紀錄訓練過程得分工具
               train=True,
               n_steps=8,
               discount=0.99): 
    """
    Args:
      agent: A2CAgent instance.
      envs: SubprocVecEnv instance.
      summary_writer: summary writer to log episode scores.
      train: whether to train the agent.
      n_steps: number of agent steps for collecting rollouts.
      discount: future reward discount.
    """

    #初始變數
    self.agent = agent 
    self.envs = envs
    self.summary_writer = summary_writer
    self.train = train
    self.n_steps = n_steps
    self.discount = discount
    
    self.preproc = Preprocessor(self.envs.observation_spec()[0]) #它返回一個觀察規格（observation specification）的結構，描述遊戲環境的觀察數據的格式和內容。
    self.episode_counter = 0 #追蹤遊戲的完成局數
    self.cumulative_score = 0.0#累積的總得分，用於計算平均得分
  #重置所有環境
  def reset(self):
    obs_raw = self.envs.reset()#處理觀察數據，轉換為模型可用格式。    
    self.last_obs = self.preproc.preprocess_obs(obs_raw) #儲存最新的處理後觀察。
  #計算平均分
  def get_mean_score(self):
    return self.cumulative_score / self.episode_counter

#用來記錄、輸出並更新每一局遊戲的分數。
  def _summarize_episode(self, timestep):
    score = timestep.observation["score_cumulative"][0] #提起"score_cumulative"第一個值
    if self.summary_writer is not None:
      #記錄分數到 TensorFlow Summary 
      summary = tf.Summary()#創建一個 TensorFlow 的 Summary 實例，用於記錄數據。
      summary.value.add(tag='sc2/episode_score', simple_value=score)#新增一條記錄tag,simple_value
      self.summary_writer.add_summary(summary, self.episode_counter)#將這條 Summary 記錄加入到 summary_writer 中，並指定其對應的局數

    print("episode %d: score = %f" % (self.episode_counter, score))
    self.episode_counter += 1 #局數+1
    return score


  
  def run_batch(self, train_summary=False):
    """Collect trajectories for a single batch and train (if self.train).

    Args:
      train_summary: return a Summary of the training step (losses, etc.).

    Returns:
      result: None (if not self.train) or the return value of agent.train.
    """
    #一,初始化變數
    shapes = (self.n_steps, self.envs.n_envs)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32) #標記是否某時間步是該局的最後一步（用布林值表示）。
    all_obs = []
    all_actions = []
    all_scores = []

    last_obs = self.last_obs #儲存上一個時間步的觀察，這是下一步的基礎。

    #二,收集數據
      #1.收集數據 
    for n in range(self.n_steps):
      
      #代理模型的步驟:
      actions, value_estimate = self.agent.step(last_obs) 
      #讓代理根據當前觀察（last_obs）選擇動作和計算價值估計。
      #返回值：
      #actions：代理選擇的動作。
      #value_estimate：代理對當前狀態的價值估計。
      actions = mask_unused_argument_samples(actions)#蔽那些動作中未使用的參數（處理格式問題）。
      size = last_obs['screen'].shape[1:3]#獲取螢幕的尺寸，用於處理空間動作（如點擊位置的座標）。

      #2.儲存數據：
      values[n, :] = value_estimate#記錄該時間步的價值估計。
      all_obs.append(last_obs)#保存當前觀察。
      all_actions.append(actions)#保存當前動作。

      #3.執行動作並更新觀察：
      pysc2_actions = actions_to_pysc2(actions, size)#將代理選擇的動作轉換成環境可執行的格式。
      obs_raw = self.envs.step(pysc2_actions)#執行動作並返回下一步的觀察。
      last_obs = self.preproc.preprocess_obs(obs_raw)#對新觀察進行預處理。

      #4.記錄獎勵和結束信息：
      rewards[n, :] = [t.reward for t in obs_raw]#記錄該時間步的獎勵。
      dones[n, :] = [t.last() for t in obs_raw]#記錄是否該時間步是該局的最後一步。

      #5.處理局結束：
      for t in obs_raw:
        if t.last():
          score = self._summarize_episode(t)
          self.cumulative_score += score

    self.last_obs = last_obs

    #三,計算回報和優勢
    next_values = self.agent.get_value(last_obs)#算最後一個時間步之後的狀態價值。

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)
    #compute_returns_advantages 方法計算：
    #returns：折扣回報（用來優化策略）。
    #advs：優勢函數（用來指導策略改進）。


    #四,準備數據
    actions = stack_and_flatten_actions(all_actions) #堆疊並壓平成批次動作數據。
    obs = flatten_first_dims_dict(stack_ndarray_dicts(all_obs)) #將觀察數據堆疊起來。
    returns = flatten_first_dims(returns) #壓平成批次格式
    advs = flatten_first_dims(advs) #壓平成批次格式



    #五,執行訓練
    if self.train:
      return self.agent.train(  #傳入觀察、動作、回報和優勢數據，進行模型更新。
          obs, actions, returns, advs,
          summary=train_summary) #如果 train_summary 是 True，還會返回一些訓練過程的統計數據。

    return None

  #run_batch 的功能可以總結為：
  #在多個時間步中，與環境交互並收集數據。
  #根據收集的數據，計算回報和優勢。
  #如果啟用訓練，則用這些數據更新模型。



def compute_returns_advantages(rewards, dones, values, next_values, discount):

#作用概述
#compute_returns_advantages 的主要功能：

#計算 折扣回報（returns）：
#回報是將未來的獎勵進行折扣後的總和。
#通常使用貝爾曼方程進行計算。
#計算 優勢（advantages）：
#優勢是回報與值函數（value function）的差值，表示某個動作的相對價值。
#用於強化學習中的策略改進。
  """Compute returns and advantages from received rewards and value estimates.

  Args:
    rewards: array of shape [n_steps, n_env] containing received rewards.
    dones: array of shape [n_steps, n_env] indicating whether an episode is
      finished after a time step.
    values: array of shape [n_steps, n_env] containing estimated values.
    next_values: array of shape [n_env] containing estimated values after the
      last step for each environment.
    discount: scalar discount for future rewards.

  Returns:
    returns: array of shape [n_steps, n_env]
    advs: array of shape [n_steps, n_env]
  """
  #1.初始化 returns 陣列
  returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])
  #returns 的大小：
  #大小為 [n_steps + 1, n_env]。
  #這樣可以方便反向計算每個時間步的回報。

  #2. 反向計算回報
  returns[-1, :] = next_values#將 returns 的最後一行設為 next_values，表示每個環境的最終狀態價值估計。
  
  for t in reversed(range(rewards.shape[0])): #使用 reversed(range(...)) 從最後一個時間步開始計算。反向計算是因為未來的回報依賴於之後的時間步。
    future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
    #returns[t + 1, :] 是下一時間步的回報。
    #乘以折扣因子 discount（通常為 0.99）進行折扣。
    #如果該時間步是最後一步（由 dones[t, :] 指示），則將回報設為 0。
    returns[t, :] = rewards[t, :] + future_rewards #將即時獎勵 rewards[t, :] 與折扣未來回報相加。   

  returns = returns[:-1, :]#去掉 returns 最後多出的一行（next_values），使其大小回到 [n_steps, n_env]。

  advs = returns - values
  # 計算優勢:returns - values：
  #returns 表示折扣回報。
  #values 表示模型對狀態價值的估計。
  #差值 advs 表示這個狀態下實際得到的回報比模型的預測多出（或少了）多少，用於指導策略改進。

  return returns, advs



#主要功能是將代理選擇的動作（模型輸出的格式）轉換為 PySC2 環境可執行的格式，讓動作能在遊戲中執行。
def actions_to_pysc2(actions, size):
  """Convert agent action representation to FunctionCall representation."""
  height, width = size #從 size 中提取地圖的高度和寬度，後續用於處理空間動作（如地圖上的點擊位置）
  fn_id, arg_ids = actions
  #fn_id：
  #每個元素是一個動作的 ID，代表 PySC2 中的某個具體動作（例如選擇單位、移動螢幕等）。
  #arg_ids：
  #是一個字典，表示動作所需的參數：
  #鍵是參數類型（例如 screen 或 minimap）。
  #值是一個陣列，表示該參數對應的值。
  
  actions_list = [] #用於存儲轉換後的 PySC2 FunctionCall 動作。
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    a_l = []
    #遍歷每個環境（fn_id.shape[0] 是環境數量），對每個環境執行以下步驟。
    #a_0：取得該環境的動作 ID。
    #a_l：初始化該動作的參數列表。

    for arg_type in FUNCTIONS._func_list[a_0].args: #FUNCTIONS._func_list[a_0].args：根據動作 ID（a_0），從 PySC2 的動作定義中獲取該動作需要的參數類型列表。
      arg_id = arg_ids[arg_type][n] #從代理的動作參數中取出該類型（arg_type）的具體值。
      
      if is_spatial_action[arg_type]: #TRUE的話
        arg = [arg_id % width, arg_id // height]#計算 x,y座標。
      else:
        arg = [arg_id] #false則直接使用參數值
      a_l.append(arg) #計算出的參數（arg）加入參數列表（a_l）。
    action = FunctionCall(a_0, a_l) #使用動作 ID（a_0）和參數列表（a_l）創建一個 PySC2 的動作。
    actions_list.append(action) #將轉換後的動作加入動作列表
  return actions_list #返回所有環境的動作列表（actions_list），供後續執行。



def mask_unused_argument_samples(actions):
#此函式屏蔽那些動作中未使用的參數類型（arg_type），將它們的值設為 -1。
#PySC2 的某些動作可能不需要某些參數，這樣可以避免在模型輸出中包含無效的參數值。

  """Replace sampled argument id by -1 for all arguments not used
  in a steps action (in-place).
  """

  fn_id, arg_ids = actions
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
    for arg_type in unused_types:
      arg_ids[arg_type][n] = -1
  return (fn_id, arg_ids)


def flatten_first_dims(x):
#壓平一個陣列的前兩個維度，將它們合併為一個。
#例如，將形狀 [n_steps, n_envs, ...] 改為 [n_steps * n_envs, ...]。

  new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
  return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
#對字典中的每個值，使用 flatten_first_dims 壓平其前兩個維度。

  return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
#將多個時間步的動作列表進行堆疊並壓平，用於批次訓練。

  fn_id_list, arg_dict_list = zip(*lst)
  fn_id = np.stack(fn_id_list, axis=axis)
  fn_id = flatten_first_dims(fn_id)
  arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
  arg_ids = flatten_first_dims_dict(arg_ids)
  return (fn_id, arg_ids)
```
