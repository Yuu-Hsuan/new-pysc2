## 來源
大來源:(https://github.com/google-deepmind/pysc2)

來源:(https://github.com/google-deepmind/pysc2/blob/master/pysc2/env/sc2_env.py)

## 概述
* 位於: `pysc2/env/sc2_env.py`，是 `PySC2` 工具的一部分
* 目的:提供一個環境用來跟 StarCraft II 互動，使進行強化學習 (Reinforcement Learning, RL) 非常有用
* 主要類別: `SC2Env`，它繼承自 `Base` 類別
* 提供與 StarCraft II 進行互動的功能:(例)
  1. 加載地圖
  2. 定義遊戲參數（例如玩家、步數）
  3. 收到指令後讓遊戲運行
     
## 解析
### import
```
import collections
import copy
import random
import time

from absl import logging
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import enums
from pysc2.env import environment
from pysc2.lib import actions as actions_lib
from pysc2.lib import features
from pysc2.lib import metrics
from pysc2.lib import portspicker
from pysc2.lib import renderer_human
from pysc2.lib import run_parallel
from pysc2.lib import stopwatch

from s2clientprotocol import sc2api_pb2 as sc_pb
```
```
sw = stopwatch.sw

possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}


Race = enums.Race
Difficulty = enums.Difficulty
BotBuild = enums.BotBuild


# Re-export these names to make it easy to construct the environment.
ActionSpace = actions_lib.ActionSpace  # pylint: disable=invalid-name
Dimensions = features.Dimensions  # pylint: disable=invalid-name
AgentInterfaceFormat = features.AgentInterfaceFormat  # pylint: disable=invalid-name
parse_agent_interface_format = features.parse_agent_interface_format


def to_list(arg):
  return arg if isinstance(arg, list) else [arg]


def get_default(a, b):
  return b if a is None else a


class Agent(collections.namedtuple("Agent", ["race", "name"])):
  
  def __new__(cls, race, name=None):
    return super(Agent, cls).__new__(cls, to_list(race), name or "<unknown>")


class Bot(collections.namedtuple("Bot", ["race", "difficulty", "build"])):

  def __new__(cls, race, difficulty, build=None):
    return super(Bot, cls).__new__(
        cls, to_list(race), difficulty, to_list(build or BotBuild.random))


_DelayedAction = collections.namedtuple(
    "DelayedAction", ["game_loop", "action"])

REALTIME_GAME_LOOP_SECONDS = 1 / 22.4
MAX_STEP_COUNT = 524000  # The game fails above 2^19=524288 steps.
NUM_ACTION_DELAY_BUCKETS = 10
```
### 類別:SC2Env
```
class SC2Env(environment.Base):
#-------------------------------------------------------------------------------------
#類別與初始化(類別 SC2Env 的定義)
  def __init__(self,
               *,
               map_name=None,   #地圖名稱，告訴遊戲我們要在哪個地圖上玩
               battle_net_map=False,   #
               players=None,   #玩家設定，可以是人類玩家（Agent）或電腦玩家（Bot）
               agent_interface_format=None,   #介面格式，定義我們如何觀察遊戲環境和發送指令
               discount=1.,
               discount_zero_after_timeout=False,
               visualize=False,   #是否在畫面上顯示遊戲
               step_mul=None,   #步進倍率，控制一次步進執行多少遊戲內部步驟
               realtime=False,
               save_replay_episodes=0,
               replay_dir=None,
               replay_prefix=None,
               game_steps_per_episode=None,
               score_index=None,
               score_multiplier=None,
               random_seed=None,
               disable_fog=False,
               ensure_available_actions=True,
               version=None):
    
    if not players:
      raise ValueError("You must specify the list of players.")

    for p in players:
      if not isinstance(p, (Agent, Bot)):
        raise ValueError(
            "Expected players to be of type Agent or Bot. Got: %s." % p)

    num_players = len(players)
    self._num_agents = sum(1 for p in players if isinstance(p, Agent))
    self._players = players

    if not 1 <= num_players <= 2 or not self._num_agents:
      raise ValueError("Only 1 or 2 players with at least one agent is "
                       "supported at the moment.")

    if not map_name:
      raise ValueError("Missing a map name.")

    self._battle_net_map = battle_net_map
    self._maps = [maps.get(name) for name in to_list(map_name)]
    min_players = min(m.players for m in self._maps)
    max_players = max(m.players for m in self._maps)
    if self._battle_net_map:
      for m in self._maps:
        if not m.battle_net:
          raise ValueError("%s isn't known on Battle.net" % m.name)

    if max_players == 1:
      if self._num_agents != 1:
        raise ValueError("Single player maps require exactly one Agent.")
    elif not 2 <= num_players <= min_players:
      raise ValueError(
          "Maps support 2 - %s players, but trying to join with %s" % (
              min_players, num_players))

    if save_replay_episodes and not replay_dir:
      raise ValueError("Missing replay_dir")

    self._realtime = realtime
    self._last_step_time = None
    self._save_replay_episodes = save_replay_episodes
    self._replay_dir = replay_dir
    self._replay_prefix = replay_prefix
    self._random_seed = random_seed
    self._disable_fog = disable_fog
    self._ensure_available_actions = ensure_available_actions
    self._discount = discount
    self._discount_zero_after_timeout = discount_zero_after_timeout
    self._default_step_mul = step_mul
    self._default_score_index = score_index
    self._default_score_multiplier = score_multiplier
    self._default_episode_length = game_steps_per_episode

    self._run_config = run_configs.get(version=version)
    self._parallel = run_parallel.RunParallel()  # Needed for multiplayer.
    self._game_info = None
    self._requested_races = None

    if agent_interface_format is None:
      raise ValueError("Please specify agent_interface_format.")

    if isinstance(agent_interface_format,
                  (AgentInterfaceFormat, sc_pb.InterfaceOptions)):
      agent_interface_format = [agent_interface_format] * self._num_agents

    if len(agent_interface_format) != self._num_agents:
      raise ValueError(
          "The number of entries in agent_interface_format should "
          "correspond 1-1 with the number of agents.")

    self._action_delay_fns = [
        aif.action_delay_fn if isinstance(aif, AgentInterfaceFormat) else None
        for aif in agent_interface_format
    ]
    self._interface_formats = agent_interface_format
    self._interface_options = [
        self._get_interface(interface_format, require_raw=visualize and i == 0)
        for i, interface_format in enumerate(agent_interface_format)]

    self._launch_game()
    self._create_join()

    self._finalize(visualize)

  def _finalize(self, visualize):
    self._delayed_actions = [collections.deque()
                             for _ in self._action_delay_fns]

    if visualize:
      self._renderer_human = renderer_human.RendererHuman()
      self._renderer_human.init(
          self._controllers[0].game_info(),
          self._controllers[0].data())
    else:
      self._renderer_human = None

    self._metrics = metrics.Metrics(self._map_name)
    self._metrics.increment_instance()

    self._last_score = None
    self._total_steps = 0
    self._episode_steps = 0
    self._episode_count = 0
    self._obs = [None] * self._num_agents
    self._agent_obs = [None] * self._num_agents
    self._state = environment.StepType.LAST  # Want to jump to `reset`.
    logging.info("Environment is ready")

  @staticmethod
  def _get_interface(interface_format, require_raw):
    if isinstance(interface_format, sc_pb.InterfaceOptions):
      if require_raw and not interface_format.raw:
        interface_options = copy.deepcopy(interface_format)
        interface_options.raw = True
        return interface_options
      else:
        return interface_format

    aif = interface_format
    interface = sc_pb.InterfaceOptions(
        raw=(aif.use_feature_units or
             aif.use_unit_counts or
             aif.use_raw_units or
             require_raw),
        show_cloaked=aif.show_cloaked,
        show_burrowed_shadows=aif.show_burrowed_shadows,
        show_placeholders=aif.show_placeholders,
        raw_affects_selection=True,
        raw_crop_to_playable_area=aif.raw_crop_to_playable_area,
        score=True)

    if aif.feature_dimensions:
      interface.feature_layer.width = aif.camera_width_world_units
      aif.feature_dimensions.screen.assign_to(
          interface.feature_layer.resolution)
      aif.feature_dimensions.minimap.assign_to(
          interface.feature_layer.minimap_resolution)
      interface.feature_layer.crop_to_playable_area = aif.crop_to_playable_area
      interface.feature_layer.allow_cheating_layers = aif.allow_cheating_layers

    if aif.rgb_dimensions:
      aif.rgb_dimensions.screen.assign_to(interface.render.resolution)
      aif.rgb_dimensions.minimap.assign_to(interface.render.minimap_resolution)

    return interface

  def _launch_game(self):
    # Reserve a whole bunch of ports for the weird multiplayer implementation.
    if self._num_agents > 1:
      self._ports = portspicker.pick_unused_ports(self._num_agents * 2)
      logging.info("Ports used for multiplayer: %s", self._ports)
    else:
      self._ports = []

    # Actually launch the game processes.
    self._sc2_procs = [
        self._run_config.start(extra_ports=self._ports,
                               want_rgb=interface.HasField("render"))
        for interface in self._interface_options]
    self._controllers = [p.controller for p in self._sc2_procs]

    if self._battle_net_map:
      available_maps = self._controllers[0].available_maps()
      available_maps = set(available_maps.battlenet_map_names)
      unavailable = [m.name for m in self._maps
                     if m.battle_net not in available_maps]
      if unavailable:
        raise ValueError("Requested map(s) not in the battle.net cache: %s"
                         % ",".join(unavailable))

  def _create_join(self):
    """Create the game, and join it."""
    map_inst = random.choice(self._maps)
    self._map_name = map_inst.name

    self._step_mul = max(1, self._default_step_mul or map_inst.step_mul)
    self._score_index = get_default(self._default_score_index,
                                    map_inst.score_index)
    self._score_multiplier = get_default(self._default_score_multiplier,
                                         map_inst.score_multiplier)
    self._episode_length = get_default(self._default_episode_length,
                                       map_inst.game_steps_per_episode)
    if self._episode_length <= 0 or self._episode_length > MAX_STEP_COUNT:
      self._episode_length = MAX_STEP_COUNT

    # Create the game. Set the first instance as the host.
    create = sc_pb.RequestCreateGame(
        disable_fog=self._disable_fog,
        realtime=self._realtime)

    if self._battle_net_map:
      create.battlenet_map_name = map_inst.battle_net
    else:
      create.local_map.map_path = map_inst.path
      map_data = map_inst.data(self._run_config)
      if self._num_agents == 1:
        create.local_map.map_data = map_data
      else:
        # Save the maps so they can access it. Don't do it in parallel since SC2
        # doesn't respect tmpdir on windows, which leads to a race condition:
        # https://github.com/Blizzard/s2client-proto/issues/102
        for c in self._controllers:
          c.save_map(map_inst.path, map_data)
    if self._random_seed is not None:
      create.random_seed = self._random_seed
    for p in self._players:
      if isinstance(p, Agent):
        create.player_setup.add(type=sc_pb.Participant)
      else:
        create.player_setup.add(
            type=sc_pb.Computer, race=random.choice(p.race),
            difficulty=p.difficulty, ai_build=random.choice(p.build))
    self._controllers[0].create_game(create)

    # Create the join requests.
    agent_players = [p for p in self._players if isinstance(p, Agent)]
    sanitized_names = crop_and_deduplicate_names(p.name for p in agent_players)
    join_reqs = []
    for p, name, interface in zip(agent_players, sanitized_names,
                                  self._interface_options):
      join = sc_pb.RequestJoinGame(options=interface)
      join.race = random.choice(p.race)
      join.player_name = name
      if self._ports:
        join.shared_port = 0  # unused
        join.server_ports.game_port = self._ports[0]
        join.server_ports.base_port = self._ports[1]
        for i in range(self._num_agents - 1):
          join.client_ports.add(game_port=self._ports[i * 2 + 2],
                                base_port=self._ports[i * 2 + 3])
      join_reqs.append(join)

    # Join the game. This must be run in parallel because Join is a blocking
    # call to the game that waits until all clients have joined.
    self._parallel.run((c.join_game, join)
                       for c, join in zip(self._controllers, join_reqs))

    self._game_info = self._parallel.run(c.game_info for c in self._controllers)
    for g, interface in zip(self._game_info, self._interface_options):
      if g.options.render != interface.render:
        logging.warning(
            "Actual interface options don't match requested options:\n"
            "Requested:\n%s\n\nActual:\n%s", interface, g.options)

    self._features = [
        features.features_from_game_info(
            game_info=g, agent_interface_format=aif, map_name=self._map_name)
        for g, aif in zip(self._game_info, self._interface_formats)]

    self._requested_races = {
        info.player_id: info.race_requested
        for info in self._game_info[0].player_info
        if info.type != sc_pb.Observer
    }

  @property
  def map_name(self):
    return self._map_name

  @property
  def game_info(self):
    """A list of ResponseGameInfo, one per agent."""
    return self._game_info

  def static_data(self):
    return self._controllers[0].data()

  def observation_spec(self):
    """Look at Features for full specs."""
    return tuple(f.observation_spec() for f in self._features)

  def action_spec(self):
    """Look at Features for full specs."""
    return tuple(f.action_spec() for f in self._features)

  def action_delays(self):
    """In realtime we track the delay observation -> action executed.

    Returns:
      A list per agent of action delays, where action delays are a list where
      the index in the list corresponds to the delay in game loops, the value
      at that index the count over the course of an episode.

    Raises:
      ValueError: If called when not in realtime mode.
    """
    if not self._realtime:
      raise ValueError("This method is only supported in realtime mode")

    return self._action_delays

  def _restart(self):
    if (len(self._players) == 1 and len(self._players[0].race) == 1 and
        len(self._maps) == 1):
      # Need to support restart for fast-restart of mini-games.
      self._controllers[0].restart()
    else:
      if len(self._controllers) > 1:
        self._parallel.run(c.leave for c in self._controllers)
      self._create_join()


#-----------------------------------------------------------------------------------
#SC2Env核心功能1:
#  重置遊戲，讓它從頭開始。這通常用於開始新的訓練回合
#  假設我們剛剛輸掉一場遊戲，呼叫 reset() 就會重置遊戲環境，並返回起始的觀察資訊
  @sw.decorate
  def reset(self):
    """Start a new episode."""
    self._episode_steps = 0
    if self._episode_count:
      # No need to restart for the first episode.
      self._restart()

    self._episode_count += 1
    races = [Race(r).name for _, r in sorted(self._requested_races.items())]
    logging.info("Starting episode %s: [%s] on %s",
                 self._episode_count, ", ".join(races), self._map_name)
    self._metrics.increment_episode()

    self._last_score = [0] * self._num_agents
    self._state = environment.StepType.FIRST
    if self._realtime:
      self._last_step_time = time.time()
      self._last_obs_game_loop = None
      self._action_delays = [[0] * NUM_ACTION_DELAY_BUCKETS] * self._num_agents

    return self._observe(target_game_loop=0)

#-----------------------------------------------------------------------------------
#SC2Env核心功能2:
#  這是與遊戲環境互動的核心方法！
#  * actions：我們的指令，例如移動單位、攻擊目標等
#  * 遊戲會執行這些指令，然後返回新的觀察資訊，以及一些其他數據，例如：
#    * 遊戲的當前畫面
#    * 回合的得分
#    * 遊戲是否結束
  @sw.decorate("step_env")
  def step(self, actions, step_mul=None):
    """Apply actions, step the world forward, and return observations.

    Args:
      actions: A list of actions meeting the action spec, one per agent, or a
          list per agent. Using a list allows multiple actions per frame, but
          will still check that they're valid, so disabling
          ensure_available_actions is encouraged.
      step_mul: If specified, use this rather than the environment's default.

    Returns:
      A tuple of TimeStep namedtuples, one per agent.
    """
    if self._state == environment.StepType.LAST:
      return self.reset()

    skip = not self._ensure_available_actions
    actions = [[f.transform_action(o.observation, a, skip_available=skip)
                for a in to_list(acts)]
               for f, o, acts in zip(self._features, self._obs, actions)]

    if not self._realtime:
      actions = self._apply_action_delays(actions)

    self._parallel.run((c.actions, sc_pb.RequestAction(actions=a))
                       for c, a in zip(self._controllers, actions))

    self._state = environment.StepType.MID
    return self._step(step_mul)

  def _step(self, step_mul=None):
    step_mul = step_mul or self._step_mul
    if step_mul <= 0:
      raise ValueError("step_mul should be positive, got {}".format(step_mul))

    target_game_loop = self._episode_steps + step_mul
    if not self._realtime:
      # Send any delayed actions that were scheduled up to the target game loop.
      current_game_loop = self._send_delayed_actions(
          up_to_game_loop=target_game_loop,
          current_game_loop=self._episode_steps)

      self._step_to(game_loop=target_game_loop,
                    current_game_loop=current_game_loop)

    return self._observe(target_game_loop=target_game_loop)

  def _apply_action_delays(self, actions):
    """Apply action delays to the requested actions, if configured to."""
    assert not self._realtime
    actions_now = []
    for actions_for_player, delay_fn, delayed_actions in zip(
        actions, self._action_delay_fns, self._delayed_actions):
      actions_now_for_player = []

      for action in actions_for_player:
        delay = delay_fn() if delay_fn else 1
        if delay > 1 and action.ListFields():  # Skip no-ops.
          game_loop = self._episode_steps + delay - 1

          # Randomized delays mean that 2 delay actions can be reversed.
          # Make sure that doesn't happen.
          if delayed_actions:
            game_loop = max(game_loop, delayed_actions[-1].game_loop)

          # Don't send an action this frame.
          delayed_actions.append(_DelayedAction(game_loop, action))
        else:
          actions_now_for_player.append(action)
      actions_now.append(actions_now_for_player)

    return actions_now

  def _send_delayed_actions(self, up_to_game_loop, current_game_loop):
    """Send any delayed actions scheduled for up to the specified game loop."""
    assert not self._realtime
    while True:
      if not any(self._delayed_actions):  # No queued actions
        return current_game_loop

      act_game_loop = min(d[0].game_loop for d in self._delayed_actions if d)
      if act_game_loop > up_to_game_loop:
        return current_game_loop

      self._step_to(act_game_loop, current_game_loop)
      current_game_loop = act_game_loop
      if self._controllers[0].status_ended:
        # We haven't observed and may have hit game end.
        return current_game_loop

      actions = []
      for d in self._delayed_actions:
        if d and d[0].game_loop == current_game_loop:
          delayed_action = d.popleft()
          actions.append(delayed_action.action)
        else:
          actions.append(None)
      self._parallel.run((c.act, a) for c, a in zip(self._controllers, actions))

  def _step_to(self, game_loop, current_game_loop):
    step_mul = game_loop - current_game_loop
    if step_mul < 0:
      raise ValueError("We should never need to step backwards")
    if step_mul > 0:
      with self._metrics.measure_step_time(step_mul):
        if not self._controllers[0].status_ended:  # May already have ended.
          self._parallel.run((c.step, step_mul) for c in self._controllers)

  def _get_observations(self, target_game_loop):
    # Transform in the thread so it runs while waiting for other observations.
    def parallel_observe(c, f):
      obs = c.observe(target_game_loop=target_game_loop)
      agent_obs = f.transform_obs(obs)
      return obs, agent_obs

    with self._metrics.measure_observation_time():
      self._obs, self._agent_obs = zip(*self._parallel.run(
          (parallel_observe, c, f)
          for c, f in zip(self._controllers, self._features)))

    game_loop = _get_game_loop(self._agent_obs[0])
    if (game_loop < target_game_loop and
        not any(o.player_result for o in self._obs)):
      raise ValueError(
          ("The game didn't advance to the expected game loop. "
           "Expected: %s, got: %s") % (target_game_loop, game_loop))
    elif game_loop > target_game_loop and target_game_loop > 0:
      logging.warning(
          "Received observation %d step(s) late: %d rather than %d.",
          game_loop - target_game_loop, game_loop, target_game_loop)

    if self._realtime:
      # Track delays on executed actions.
      # Note that this will underestimate e.g. action sent, new observation
      # taken before action executes, action executes, observation taken
      # with action. This is difficult to avoid without changing the SC2
      # binary - e.g. send the observation game loop with each action,
      # return them in the observation action proto.
      if self._last_obs_game_loop is not None:
        for i, obs in enumerate(self._obs):
          for action in obs.actions:
            if action.HasField("game_loop"):
              delay = action.game_loop - self._last_obs_game_loop
              if delay > 0:
                num_slots = len(self._action_delays[i])
                delay = min(delay, num_slots - 1)  # Cap to num buckets.
                self._action_delays[i][delay] += 1
                break
      self._last_obs_game_loop = game_loop

  def _observe(self, target_game_loop):
    self._get_observations(target_game_loop)

    # TODO(tewalds): How should we handle more than 2 agents and the case where
    # the episode can end early for some agents?
    outcome = [0] * self._num_agents
    discount = self._discount
    episode_complete = any(o.player_result for o in self._obs)

    if episode_complete:
      self._state = environment.StepType.LAST
      discount = 0
      for i, o in enumerate(self._obs):
        player_id = o.observation.player_common.player_id
        for result in o.player_result:
          if result.player_id == player_id:
            outcome[i] = possible_results.get(result.result, 0)

    if self._score_index >= 0:  # Game score, not win/loss reward.
      cur_score = [_get_score(o, self._score_index) for o in self._agent_obs]
      if self._episode_steps == 0:  # First reward is always 0.
        reward = [0] * self._num_agents
      else:
        reward = [cur - last for cur, last in zip(cur_score, self._last_score)]
      self._last_score = cur_score
    else:
      reward = outcome

    if self._renderer_human:
      self._renderer_human.render(self._obs[0])
      cmd = self._renderer_human.get_actions(
          self._run_config, self._controllers[0])
      if cmd == renderer_human.ActionCmd.STEP:
        pass
      elif cmd == renderer_human.ActionCmd.RESTART:
        self._state = environment.StepType.LAST
      elif cmd == renderer_human.ActionCmd.QUIT:
        raise KeyboardInterrupt("Quit?")

    game_loop = _get_game_loop(self._agent_obs[0])
    self._total_steps += game_loop - self._episode_steps
    self._episode_steps = game_loop
    if self._episode_steps >= self._episode_length:
      self._state = environment.StepType.LAST
      if self._discount_zero_after_timeout:
        discount = 0.0
      if self._episode_steps >= MAX_STEP_COUNT:
        logging.info("Cut short to avoid SC2's max step count of 2^19=524288.")

    if self._state == environment.StepType.LAST:
      if (self._save_replay_episodes > 0 and
          self._episode_count % self._save_replay_episodes == 0):
        self.save_replay(self._replay_dir, self._replay_prefix)
      logging.info(("Episode %s finished after %s game steps. "
                    "Outcome: %s, reward: %s, score: %s"),
                   self._episode_count, self._episode_steps, outcome, reward,
                   [_get_score(o) for o in self._agent_obs])

    def zero_on_first_step(value):
      return 0.0 if self._state == environment.StepType.FIRST else value
    return tuple(environment.TimeStep(
        step_type=self._state,
        reward=zero_on_first_step(r * self._score_multiplier),
        discount=zero_on_first_step(discount),
        observation=o) for r, o in zip(reward, self._agent_obs))

  def send_chat_messages(self, messages, broadcast=True):
    """Useful for logging messages into the replay."""
    self._parallel.run(
        (c.chat,
         message,
         sc_pb.ActionChat.Broadcast if broadcast else sc_pb.ActionChat.Team)
        for c, message in zip(self._controllers, messages))

  def save_replay(self, replay_dir, prefix=None):
    if prefix is None:
      prefix = self._map_name
    replay_path = self._run_config.save_replay(
        self._controllers[0].save_replay(), replay_dir, prefix)
    logging.info("Wrote replay to: %s", replay_path)
    return replay_path

#-----------------------------------------------------------------------------------
#SC2Env核心功能3:
#  關閉遊戲環境，釋放資源。如果我們不需要繼續與遊戲互動，就可以呼叫這個方法
  def close(self):
    logging.info("Environment Close")
    if hasattr(self, "_metrics") and self._metrics:
      self._metrics.close()
      self._metrics = None
    if hasattr(self, "_renderer_human") and self._renderer_human:
      self._renderer_human.close()
      self._renderer_human = None

    # Don't use parallel since it might be broken by an exception.
    if hasattr(self, "_controllers") and self._controllers:
      for c in self._controllers:
        c.quit()
      self._controllers = None
    if hasattr(self, "_sc2_procs") and self._sc2_procs:
      for p in self._sc2_procs:
        p.close()
      self._sc2_procs = None

    if hasattr(self, "_ports") and self._ports:
      portspicker.return_ports(self._ports)
      self._ports = None

    if hasattr(self, "_parallel") and self._parallel is not None:
      self._parallel.shutdown()
      self._parallel = None

    self._game_info = None


def crop_and_deduplicate_names(names):
  """Crops and de-duplicates the passed names.

  SC2 gets confused in a multi-agent game when agents have the same
  name. We check for name duplication to avoid this, but - SC2 also
  crops player names to a hard character limit, which can again lead
  to duplicate names. To avoid this we unique-ify names if they are
  equivalent after cropping. Ideally SC2 would handle duplicate names,
  making this unnecessary.

  TODO(b/121092563): Fix this in the SC2 binary.

  Args:
    names: List of names.

  Returns:
    De-duplicated names cropped to 32 characters.
  """
  max_name_length = 32

  # Crop.
  cropped = [n[:max_name_length] for n in names]

  # De-duplicate.
  deduplicated = []
  name_counts = collections.Counter(n for n in cropped)
  name_index = collections.defaultdict(lambda: 1)
  for n in cropped:
    if name_counts[n] == 1:
      deduplicated.append(n)
    else:
      deduplicated.append("({}) {}".format(name_index[n], n))
      name_index[n] += 1

  # Crop again.
  recropped = [n[:max_name_length] for n in deduplicated]
  if len(set(recropped)) != len(recropped):
    raise ValueError("Failed to de-duplicate names")

  return recropped


def _get_game_loop(agent_obs):
  if isinstance(agent_obs, sc_pb.ResponseObservation):
    return agent_obs.observation.game_loop
  else:
    return agent_obs.game_loop[0]


def _get_score(agent_obs, score_index=0):
  if isinstance(agent_obs, sc_pb.ResponseObservation):
    if score_index != 0:
      raise ValueError(
          "Non-zero score index isn't supported for passthrough agents, "
          "currently")
    return agent_obs.observation.score.score
  else:
    return agent_obs["score_cumulative"][score_index]
```
