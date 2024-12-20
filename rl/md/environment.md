# 概述
這段程式碼是用於建立和管理 StarCraft II 的多環境執行框架，支援同時運行多個強化學習環境以提升訓練效率。程式碼包括單環境（`SingleEnv`）和多進程環境（`SubprocVecEnv`）的實作，並使用了 Python 的多進程工具

# 程式碼解析
### 1.註解
這段程式碼是基於一個開源項目進行修改的，原始碼來源於 `sc2aibot` 的 `common/multienv.py`
```
# Adapted from
# https://github.com/pekaalto/sc2aibot/blob/master/common/multienv.py
```
### 2.匯入庫
* 上行程式碼匯入 Python 的 `multiprocessing` 模組：
  * `multiprocessing.Process`：用於創建新的進程，允許多任務同時運行
  * `multiprocessing.Pipe`：用於進程之間的雙向通訊，允許進程互相傳遞訊息
* 下行匯入了 PySC2（StarCraft II 的強化學習環境）模組：
  * `pysc2.env.sc2_env`：用於創建和管理 StarCraft II 遊戲環境
  * `available_actions_printer`：可以用來打印出目前環境中可用的動作，通常用於除錯或檢查環境
```
from multiprocessing import Process, Pipe #上

from pysc2.env import sc2_env, available_actions_printer #下
```
### 3.`SingleEnv` 類別：單環境類
* 主要功能：

   `SingleEnv` 是一個簡化的環境接口，僅支持單個 StarCraft II 環境運行於主進程
```
class SingleEnv:
  """Same interface as SubprocVecEnv, but runs only one environment in the
  main process.(與 SubprocVecEnv 擁有相同接口，但只在主進程中運行一個環境)
  """
#------------------------------------------------------------------------------------
# 初始化：建立環境實例，並設定環境數量為 1
  def __init__(self, env):
    self.env = env # env: 遊戲環境實例
    self.n_envs = 1 # 環境數量固定為 1
#------------------------------------------------------------------------------------
# 執行動作(step)：接受動作列表，對環境執行第一個動作，並回傳執行結果
  def step(self, actions):
    """
    :param actions: List[FunctionCall]  # 動作列表，每個元素為一個函數調用
    :return: 環境返回的時間步結果
    """
    assert len(actions) == 1  # only 1 environment：檢查動作數量是否為 1，因為只有一個環境
    action = actions[0] # 提取唯一的動作
    return [self.env.step([action])[0]] # 執行動作，並返回環境更新後的結果（列表格式）
#------------------------------------------------------------------------------------
# 重置環境(reset)：重置環境並返回初始觀察值、狀態
  def reset(self):
    return [self.env.reset()[0]]
#------------------------------------------------------------------------------------
# 關閉環境(close)：關閉環境以釋放資源
  def close(self):
    self.env.close() 
#------------------------------------------------------------------------------------
# 獲取觀察規範 (observation_spec)：返回環境的觀察規範，用於了解環境的輸出結構
  def observation_spec(self):
    return [self.env.observation_spec()] # 返回觀察空間的規格（列表格式）
```

### 4.多進程處理工具
```
# below (worker, CloudpickleWrapper, SubprocVecEnv) copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
#------------------------------------------------------------------------------------
# 多進程工人函數 worker 函數：此函數是每個子進程運行的核心代碼，用於處理與主進程的通訊

def worker(remote, env_fn_wrapper):

  """
  Handling the:
  action -> [action] and  [timestep] -> timestep
  single-player conversions here
  用於處理：
  action -> [action] 和 [timestep] -> timestep 的單人模式轉換
  """

# 初始化環境
  env = env_fn_wrapper.x() # 使用 env_fn_wrapper 來創建環境實例

# 無窮迴圈處理命令：不斷從主進程接收命令和參數
  while True:
    cmd, action = remote.recv() # 從主進程接收命令和動作

# 命令處理：執行動作並返回更新後的狀態
    if cmd == 'step':
      timesteps = env.step([action])  # 執行動作並返回時間步
      assert len(timesteps) == 1  # 確保只有一個時間步結果
      remote.send(timesteps[0])  # 發送時間步結果

# 重置環境並返回初始狀態
    elif cmd == 'reset':
      timesteps = env.reset()  # 重置環境
      assert len(timesteps) == 1  # 確保只有一個初始狀態
      remote.send(timesteps[0])  # 發送初始狀態

# 關閉連接並退出迴圈
    elif cmd == 'close':
      remote.close()  # 關閉進程通信管道
      break  # 跳出循環，結束進程

# 返回觀察空間的規格
    elif cmd == 'observation_spec':
      spec = env.observation_spec()  # 獲取環境觀察規範
      remote.send(spec)  # 發送觀察規範

# 對於未知命令，拋出錯誤
    else:
      raise NotImplementedError  # 如果命令未實現，則拋出錯誤
```

### 5.`CloudpickleWrapper` 類別：
使用 `cloudpickle` 序列化環境函數，因為 `multiprocessing` 默認使用的 `pickle` 不支援某些類型的數據
```
class CloudpickleWrapper(object):

  """
  Uses cloudpickle to serialize contents (otherwise multiprocessing tries
  to use pickle).(使用 cloudpickle 進行序列化（否則多處理模組會嘗試使用 pickle）)
  """
# 初始化和序列化
  def __init__(self, x):
    self.x = x  # 儲存需要序列化的對象

  def __getstate__(self): # 負責序列化過程
    import cloudpickle
    return cloudpickle.dumps(self.x)  # 使用 cloudpickle 序列化對象

  def __setstate__(self, ob): # 負責反序列化過程
    import pickle
    self.x = pickle.loads(ob)  # 使用 pickle 反序列化對象
```

### 6.`SubprocVecEnv` 類別：多進程環境管理

* 主要功能：
  `SubprocVecEnv` 提供多進程環境管理，可以同時運行多個 StarCraft II 的環境

```
# 初始化：
class SubprocVecEnv:
  def __init__(self, env_fns):
    n_envs = len(env_fns) # env_fns: 每個環境的生成函數
    # 建立兩組通信管道：主進程端（remotes）和子進程端（work_remotes）
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)]) 

    # 為每個環境創建一個子進程，並執行工人函數 (worker)
    self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
               for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
    for p in self.ps:
      p.start()  # 啟動所有子進程

    self.n_envs = n_envs  # 記錄環境數量
#------------------------------------------------------------------------------------
# 執行步驟或重置：發送命令和動作給工人進程，並收集返回值
  def _step_or_reset(self, command, actions=None):
    actions = actions or [None] * self.n_envs  # 如果未提供動作，設為 None
    for remote, action in zip(self.remotes, actions):
      remote.send((command, action))  # 向每個進程發送命令和對應動作
    timesteps = [remote.recv() for remote in self.remotes]  # 接收所有進程的返回結果
    return timesteps  # 返回時間步列表

# 執行動作
  def step(self, actions):
    return self._step_or_reset("step", actions)  # 執行步進操作

# 重置環境
  def reset(self):
    return self._step_or_reset("reset", None)  # 重置所有環境

# 關閉環境：發送關閉命令，並等待所有進程退出
  def close(self):
    for remote in self.remotes:
      remote.send(('close', None))  # 向所有進程發送關閉命令
    for p in self.ps:
      p.join()  # 等待所有子進程結束
#------------------------------------------------------------------------------------
# 獲取觀察空間規格
  def observation_spec(self):
    for remote in self.remotes:
      remote.send(('observation_spec', None))  # 向所有進程請求觀察規範
    specs = [remote.recv() for remote in self.remotes]  # 接收所有進程返回的觀察規範
    return specs  # 返回觀察規範列表
```

### 7.創建遊戲環境 (`make_sc2env`)
使用提供的參數創建 StarCraft II 遊戲環境
```
def make_sc2env(**kwargs):  # 使用給定參數創建 StarCraft II 環境
  env = sc2_env.SC2Env(**kwargs)
  # env = available_actions_printer.AvailableActionsPrinter(env)
  # # （可選）在環境中打印可用動作，便於調試
  return env  # 返回創建的環境
```
