# 概述
這段程式碼是用於建立和管理《星海爭霸 II》的多環境執行框架，支援同時運行多個強化學習環境以提升訓練效率。程式碼包括單環境（`SingleEnv`）和多進程環境（`SubprocVecEnv`）的實作，並使用了 Python 的多進程工具

# 程式碼解析
### 1.註解
這段程式碼是基於一個開源項目進行修改的，原始碼來源於 `sc2aibot` 的 `common/multienv.py`
```
# Adapted from
# https://github.com/pekaalto/sc2aibot/blob/master/common/multienv.py
```
### 2.匯入庫
* 上行程式碼匯入 Python 的 `multiprocessing` 模組：
  * `Process`：用於創建新的進程，允許多任務同時運行
  * `Pipe`：用於進程之間的雙向通訊，允許進程互相傳遞訊息
* 下行匯入了 PySC2（《星海爭霸 II》的強化學習環境）模組：
  * `sc2_env`：用於創建和管理《星海爭霸 II》遊戲環境
  * `available_actions_printer`：可以用來打印出目前環境中可用的動作，通常用於除錯或檢查環境
```
from multiprocessing import Process, Pipe #上

from pysc2.env import sc2_env, available_actions_printer #下
```
### 3.`SingleEnv` 類別
* 主要功能：

  `SingleEnv` 是一個簡化的環境接口，僅支持單個《星海爭霸 II》環境運行於主進程
```
class SingleEnv:
  """Same interface as SubprocVecEnv, but runs only one environment in the
  main process.
  """
#------------------------------------------------------------------------------------
# 初始化：建立環境實例，並設定環境數量為 1
  def __init__(self, env):
    self.env = env
    self.n_envs = 1
#------------------------------------------------------------------------------------
# 執行動作：接受動作列表，對環境執行第一個動作，並回傳執行結果
  def step(self, actions):
    """
    :param actions: List[FunctionCall]
    :return:
    """
    assert len(actions) == 1  # only 1 environment
    action = actions[0]
    return [self.env.step([action])[0]]
#------------------------------------------------------------------------------------
# 重置環境：重置環境並返回初始觀察值
  def reset(self):
    return [self.env.reset()[0]]
#------------------------------------------------------------------------------------
# 關閉環境：
  def close(self):
    self.env.close()
#------------------------------------------------------------------------------------
# 獲取觀察規範：返回環境的觀察規範，用於了解環境的輸出結構
  def observation_spec(self):
    return [self.env.observation_spec()]
```


# below (worker, CloudpickleWrapper, SubprocVecEnv) copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
def worker(remote, env_fn_wrapper):
  """
  Handling the:
  action -> [action] and  [timestep] -> timestep
  single-player conversions here
  """
  env = env_fn_wrapper.x()
  while True:
    cmd, action = remote.recv()
    if cmd == 'step':
      timesteps = env.step([action])
      assert len(timesteps) == 1
      remote.send(timesteps[0])
    elif cmd == 'reset':
      timesteps = env.reset()
      assert len(timesteps) == 1
      remote.send(timesteps[0])
    elif cmd == 'close':
      remote.close()
      break
    elif cmd == 'observation_spec':
      spec = env.observation_spec()
      remote.send(spec)
    else:
      raise NotImplementedError


class CloudpickleWrapper(object):
  """
  Uses cloudpickle to serialize contents (otherwise multiprocessing tries
  to use pickle).
  """

  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)


class SubprocVecEnv:
  def __init__(self, env_fns):
    n_envs = len(env_fns)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
    self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
               for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
    for p in self.ps:
      p.start()

    self.n_envs = n_envs

  def _step_or_reset(self, command, actions=None):
    actions = actions or [None] * self.n_envs
    for remote, action in zip(self.remotes, actions):
      remote.send((command, action))
    timesteps = [remote.recv() for remote in self.remotes]
    return timesteps

  def step(self, actions):
    return self._step_or_reset("step", actions)

  def reset(self):
    return self._step_or_reset("reset", None)

  def close(self):
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()

  def observation_spec(self):
    for remote in self.remotes:
      remote.send(('observation_spec', None))
    specs = [remote.recv() for remote in self.remotes]
    return specs
