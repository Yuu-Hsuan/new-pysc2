```
import sys
import os
import shutil
import sys
import argparse
from functools import partial

import tensorflow as tf

from rl.agents.a2c.runner import A2CRunner
from rl.agents.a2c.agent import A2CAgent
from rl.networks.fully_conv import FullyConv
from rl.environment import SubprocVecEnv, make_sc2env, SingleEnv


# Workaround for pysc2 flags
from absl import flags  #flags是全域旗標物件：可以存取和設定所有旗標。
FLAGS = flags.FLAGS
FLAGS(['run.py'])
#這行會初始化旗標系統並解析命令列參數。
#解決了 pysc2 的旗標在未初始化時導致的錯誤。


parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents') #初始化一個參數解析器，描述為「Starcraft 2 的深度強化學習代理」。

parser.add_argument('experiment_id', type=str,
                    help='identifier to store experiment results')  #必須提供一個實驗 ID，這是用於存儲結果的標識符。

parser.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')  #--eval：如果設定此選項，將切換到「評估模式」而非訓練模式。

parser.add_argument('--ow', action='store_true',
                    help='overwrite existing experiments (if --train=True)') #--ow：允許覆蓋現有的實驗結果，避免與以前的實驗資料衝突。

parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map') #--map：指定 StarCraft II 的地圖名稱，預設為 MoveToBeacon。

parser.add_argument('--vis', action='store_true',
                    help='render with pygame') #--vis：啟用視覺化功能，讓環境渲染過程可見。

parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open') #--max_windows：設定同時開啟的視覺化視窗數量上限

parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')#--res：設定螢幕和小地圖的解析度，預設為 32x32。

parser.add_argument('--envs', type=int, default=32,
                    help='number of environments simulated in parallel')#--envs：並行模擬的環境數量，預設為 32。

parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')#--step_mul：每次代理執行操作時遊戲步數的倍率，預設為 8

parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')#--steps_per_batch：在每批訓練中收集的代理步數，預設為 16。

parser.add_argument('--discount', type=float, default=0.99,
                    help='discount for future rewards')##--discount：折扣因子，用於加權未來獎勵的重要性，預設為 0.99。

parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')#--iters：設定迭代次數，-1 表示無限執行。

parser.add_argument('--seed', type=int, default=123,
                    help='random seed')#--seed：設定隨機數種子，確保結果可重現。

parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')#--gpu：指定 GPU 裝置的 ID，預設為 GPU 0。

parser.add_argument('--nhwc', action='store_true',
                    help='train fullyConv in NCHW mode')#--nhwc：選擇神經網路的資料格式，使用 NHWC（通道在最後）模式。

parser.add_argument('--summary_iters', type=int, default=10,
                    help='record training summary after this many iterations')##--summary_iters：每隔幾次迭代記錄訓練摘要，預設為每 10 次。。

parser.add_argument('--save_iters', type=int, default=5000,
                    help='store checkpoint after this many iterations')#--save_iters：每隔幾次迭代儲存模型檢查點，預設為 5000 次。

parser.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')#--max_to_keep：最多保留多少個模型檢查點，預設為 5。

parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')#--entropy_weight：設定熵損失的權重，預設為 0.001。

parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')#--value_loss_weight：設定價值函數損失的權重，預設為 0.5。

parser.add_argument('--lr', type=float, default=7e-4,
                    help='initial learning rate')#--lr：設定初始學習率，預設為 0.0007。

parser.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                    help='root directory for checkpoint storage')#--save_dir：儲存模型檢查點的目錄，預設為 out/models。

parser.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                    help='root directory for summary storage')#--summary_dir：儲存訓練摘要的目錄，預設為 out/summary。



##命令列參數解析與初始化

args = parser.parse_args() #解析命令列參數：從命令列輸入解析所有選項，並存入 args。
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
args.train = not args.eval #設定模式：如果沒有啟用 --eval，則進行訓練模式 (args.train=True)。
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu #設定 GPU：告訴 TensorFlow 只使用指定的 GPU（由 --gpu 指定的 ID）。

##設定檔案路徑

ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)
#檢查點與摘要路徑：
#ckpt_path：模型檢查點存放的路徑，根據實驗 ID 建立。
#summary_path：訓練或評估摘要存放的路徑。


##定義儲存函數

def _save_if_training(agent, summary_writer):
  if args.train:
    agent.save(ckpt_path)
    summary_writer.flush()
    sys.stdout.flush()
#保存模型檢查點：如果處於訓練模式，儲存代理模型並刷新摘要寫入器。

###

def main():
    ##刪除舊的結果（如果啟用覆蓋）
    if args.train and args.ow:
      shutil.rmtree(ckpt_path, ignore_errors=True)
      shutil.rmtree(summary_path, ignore_errors=True)


    ##環境設定
    size_px = (args.res, args.res)
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px)
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis
    #設置環境參數：
    #使用解析度與地圖等命令列參數初始化環境配置。
    #如果啟用了視覺化（--vis），會添加視覺化標記。

    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
      env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    #建立環境函數列表：
    #根據是否啟用視覺化，將可視化與非可視化的環境函數分配到列表中。

    envs = SubprocVecEnv(env_fns) #初始化並行環境：使用 SubprocVecEnv 創建多執行緒的環境模擬器。


    ##TensorFlow Session 初始化
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(summary_path)
    #建立 TensorFlow Session：
    #sess：代理訓練的 TensorFlow 計算圖會話。
    #summary_writer：用於記錄訓練摘要的寫入器。


    ##初始化代理與執行器

    #創建 A2C 代理：
    #設定代理的參數，例如學習率、損失權重與神經網路的資料格式。
    network_data_format = 'NHWC' if args.nhwc else 'NCHW'

    agent = A2CAgent(
        sess=sess,
        network_data_format=network_data_format,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr,
        max_to_keep=args.max_to_keep)

    #創建 Runner：負責代理與環境的交互，並收集訓練批次。
    runner = A2CRunner(
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch)

    static_shape_channels = runner.preproc.get_input_channels()
    agent.build(static_shape_channels, resolution=args.res)


    ##模型檢查點處理

    if os.path.exists(ckpt_path):
      agent.load(ckpt_path)
    else:
      agent.init()
    #載入或初始化模型：如果有已保存的檢查點則載入，否則初始化代理模型。
    
    
    runner.reset()#重置環境與 Runner：清除所有內部狀態，準備開始訓練或評估。


    ##主訓練/執行迴圈

    i = 0
    try:
      while True:
        write_summary = args.train and i % args.summary_iters == 0

        if i > 0 and i % args.save_iters == 0:
          _save_if_training(agent, summary_writer)

        result = runner.run_batch(train_summary=write_summary)

        if write_summary:
          agent_step, loss, summary = result
          summary_writer.add_summary(summary, global_step=agent_step)
          print('iter %d: loss = %f' % (agent_step, loss))

        i += 1

        if 0 <= args.iters <= i:
          break
    #主迴圈步驟：
    #確定是否記錄摘要（每隔 --summary_iters 次迭代）。
    #在每隔 --save_iters 次迭代時保存檢查點。
    #運行一個訓練批次，並根據結果記錄摘要或打印損失。
    #如果達到設定的迭代次數 (--iters)，結束迴圈。



    except KeyboardInterrupt: #中斷處理：允許用戶在按下 Ctrl+C 時安全停止程式。
        pass


    ##清理與資源釋放
    _save_if_training(agent, summary_writer)

    envs.close()
    summary_writer.close()

    print('mean score: %f' % runner.get_mean_score()) #顯示平均分數：計算並顯示代理在執行過程中的平均得分。


if __name__ == "__main__":
    main()
```
