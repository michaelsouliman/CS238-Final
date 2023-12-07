import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import Union
from PIL import Image, ImageDraw, ImageColor
import copy
import logging

# Used from ma-gym library

class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True

def get_cell_sizes(cell_size: Union[int, list, tuple]):
    """Handle multiple type options of `cell_size`.

    In order to keep the old API of following functions, as well as add
    support for non-square grids we need to check cell_size type and
    extend it appropriately.

    Args:
        cell_size: integer of tuple/list size of two with cell size
            in horizontal and vertical direction.

    Returns:
        Horizontal and vertical cell size.
    """
    if isinstance(cell_size, int):
        cell_size_vertical = cell_size
        cell_size_horizontal = cell_size
    elif isinstance(cell_size, (tuple, list)) and len(cell_size) == 2:
        # Flipping coordinates, because first coordinates coresponds with height (=vertical direction)
        cell_size_vertical, cell_size_horizontal = cell_size
    else:
        raise TypeError("`cell_size` must be integer, tuple or list with length two.")

    return cell_size_horizontal, cell_size_vertical


def draw_grid(rows, cols, cell_size=50, fill='black', line_color='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)

    width = cols * cell_size_x
    height = rows * cell_size_y
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height

    for x in range(0, image.width, cell_size_x):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=line_color)

    x = image.width - 1
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, cell_size_y):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    y = image.height - 1
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y, x_dash, y_dash = row + margin_x, col + margin_y, row + cell_size_x - margin_x, col + cell_size_y - margin_y
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def write_cell_text(image, text, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y = row + margin_x, col + margin_y
    ImageDraw.Draw(image).text((x, y), text=text, fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size_x, col + cell_size_y)], outline=fill, width=3)


def draw_circle(image, pos, cell_size=50, fill='black', radius=0.3):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    gap_x, gap_y = cell_size_x * radius, cell_size_y * radius
    x, y = row + gap_x, col + gap_y
    x_dash, y_dash = row + cell_size_x - gap_x, col + cell_size_y - gap_y
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_border(image, border_width=1, fill='black'):
    width, height = image.size
    new_im = Image.new("RGB", size=(width + 2 * border_width, height + 2 * border_width), color=fill)
    new_im.paste(image, (border_width, border_width))
    return new_im


def draw_score_board(image, score, board_height=30):
    im_width, im_height = image.size
    new_im = Image.new("RGB", size=(im_width, im_height + board_height), color='#e1e4e8')
    new_im.paste(image, (0, board_height))

    _text = ', '.join([str(round(x, 2)) for x in score])
    ImageDraw.Draw(new_im).text((10, board_height // 3), text=_text, fill='black')
    return new_im

from re import X
logger = logging.getLogger(__name__)

class RandomWithTaskAmongUsEnv(gym.Env):
    """
    Just has agents move around randomly and but adds rewards for crewmates finishing tasks
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(11, 11), n_imposters=1, n_crew=3, imposter_view=1,
                 full_observable=True, kill_reward=1, task_reward=0.2, missed_kill_reward=0.5):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'


        self._grid_shape = grid_shape
        self.n_imposters = n_imposters
        self.n_crew = n_crew
        self.crew_kill_reward = kill_reward
        self.task_reward = task_reward
        self.missed_kill_reward = missed_kill_reward
        self.imposter_view = imposter_view

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_imposters)])
        self.imposter_pos = {_: None for _ in range(self.n_imposters)}
        self.crew_pos = {_: None for _ in range(self.n_crew)}
        self.crew_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self.all_dones = [False for _ in range(self.n_imposters + self.n_crew)]
        self.crew_move_probs = (0.2, 0.2, 0.2, 0.2, 0.2)
        self.viewer = None
        self.full_observable = full_observable

        self._obs_high = np.array([1.] * 11 , dtype=np.float32)
        self._obs_low = np.array([1.] * 11, dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_imposters)
            self._obs_low = np.tile(self._obs_low, self.n_imposters)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_imposters)])

        self._total_episode_reward = None
        self.seed()
        self._max_steps = 1000
        self.task_positions = [(3,3), (3,7), (7,3), (7,7)]
        self.task_completion = {crew_id: set() for crew_id in range(self.n_crew)}


    def get_action_meanings(self, imposter_i=None):
        if imposter_i is not None:
            assert imposter_i <= self.n_imposters
            return [ACTION_MEANING[i] for i in range(self.action_space[imposter_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for imposter_i in range(self.n_imposters):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.imposter_pos[imposter_i] = pos
                    break
            self.__update_imposter_view(imposter_i)

        for crew_i in range(self.n_crew):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (pos != self.imposter_pos[0]):
                    self.crew_pos[crew_i] = pos
                    break
            self.__update_crew_view(crew_i)

        self.__draw_base_img()

    def get_imposter_obs(self):
        pos = self.imposter_pos[0]
        obs = []

        
        min_x = pos[0]-self.imposter_view
        max_x = pos[0]+self.imposter_view 
        min_y = pos[1]-self.imposter_view
        max_y =pos[1]+self.imposter_view

        # check if prey is in the view area
        for row in range(min_x, max_x+1):
            for col in range(min_y, max_y+1):
                if row < 0 or row > 10 or col < 0 or col > 10:
                    obs.append(0)
                    continue
                if row == pos[0] and col == pos[1]:
                    continue
                if PRE_IDS['crew'] in self._full_obs[row][col]:
                    obs.append(1)
                else:
                    obs.append(0)
        
        return obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_imposters)]
        self.imposter_pos = {}
        self.crew_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._steps_beyond_done = None
        self._imposter_dones = [False for _ in range(self.n_imposters)]
        self._crew_alive = [True for _ in range(self.n_crew)]

        x = self.get_imposter_obs()
        return x
    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_imposter_pos(self, imposter_i, move):

        curr_pos = copy.copy(self.imposter_pos[imposter_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.imposter_pos[imposter_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_imposter_view(imposter_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_crew_pos(self, crew_i, move):
        curr_pos = copy.copy(self.crew_pos[crew_i])
        reward_deduction = 0
        if self._crew_alive[crew_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.crew_pos[crew_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                next_pos_tuple = tuple(next_pos)
                if next_pos_tuple in self.task_positions and next_pos_tuple not in self.task_completion[crew_i]:
                    self.task_completion[crew_i].add(next_pos_tuple)
                    for imposter_i in range(self.n_imposters):
                        reward_deduction += self.task_reward
                self.__update_crew_view(crew_i)
                return reward_deduction
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __update_imposter_view(self, imposter_i):
        self._full_obs[self.imposter_pos[imposter_i][0]][self.imposter_pos[imposter_i][1]] = PRE_IDS['imposter'] + str(imposter_i + 1)

    def __update_crew_view(self, crew_i):
        self._full_obs[self.crew_pos[crew_i][0]][self.crew_pos[crew_i][1]] = PRE_IDS['crew'] + str(crew_i + 1)

    def imposter_missed(self, imposter_pos, crew_pos):

        min_x = imposter_pos[0]-self.imposter_view
        max_x = imposter_pos[0]+self.imposter_view 
        min_y = imposter_pos[1]-self.imposter_view
        max_y = imposter_pos[1]+self.imposter_view

        for row in range(min_x, max_x+1):
            for col in range(min_y, max_y+1):
                if row == crew_pos[0] and col == crew_pos[1]:
                    return True

        return False

    def step(self, imposters_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        reward = 0

        init_pos = self.imposter_pos[0]
        init_view = self.get_imposter_obs()
        init_state = init_pos + init_view
        for imposter_i, action in enumerate(imposters_action):
            if not (self._imposter_dones[imposter_i]):
                self.__update_imposter_pos(imposter_i, action)

        for crew_i in range(self.n_crew):
            if self._crew_alive[crew_i]:
                pos = self.crew_pos[crew_i]
                if self.imposter_pos[0][0] == pos[0] and self.imposter_pos[0][1] == pos[1]:
                    print('HERE!')
                    reward += self.crew_kill_reward
                    self._crew_alive[crew_i] = False
                if self.imposter_missed(self.imposter_pos[0],pos):
                    reward -= self.missed_kill_reward

                crew_move = self.action_space.sample()[0]
                reward_deduction = self.__update_crew_pos(crew_i, crew_move)
                if reward_deduction != None: reward -= reward_deduction

        if (self._step_count >= self._max_steps) or (True not in self._crew_alive):
            for i in range(self.n_imposters):
                self._imposter_dones[i] = True

        for i in range(self.n_imposters):
            self._total_episode_reward[i] += reward

        # Check for episode overflow
        if all(self._imposter_dones):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1
        
        end_pos = self.imposter_pos[0]
        end_view = self.get_imposter_obs()
        end_state = end_pos + end_view

        return init_state, reward, end_state, self._imposter_dones

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        min_x = pos[0]-self.imposter_view
        max_x = pos[0]+self.imposter_view 
        min_y = pos[1]-self.imposter_view
        max_y =pos[1]+self.imposter_view

        # check if prey is in the view area
        for row in range(min_x, max_x+1):
            for col in range(min_y, max_y+1):
                if self.is_valid([row, col]):
                    neighbours.append([row, col])
        return neighbours

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        img = copy.copy(self._base_img)
        for imposter_i in range(self.n_imposters):
            for neighbour in self.__get_neighbour_coordinates(self.imposter_pos[imposter_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.imposter_pos[imposter_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for imposter_i in range(self.n_imposters):
            draw_circle(img, self.imposter_pos[imposter_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(imposter_i + 1), pos=self.imposter_pos[imposter_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for crew_i in range(self.n_crew):
            if self._crew_alive[crew_i]:
                draw_circle(img, self.crew_pos[crew_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(crew_i + 1), pos=self.crew_pos[crew_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        for task_pos in self.task_positions:
            fill_cell(img, list(task_pos), cell_size=CELL_SIZE, fill='green')

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'imposter': 'I',
    'crew': 'C',
    'wall': 'W',
    'empty': '0',
    'task': 'T'
}

import csv
import time
import threading
def run_game(output_path):
    env = RandomWithTaskAmongUsEnv()
    done_n = [False for _ in range(env.n_imposters)]
    ep_reward = 0

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['s', 'a', 'r', 's\''])
        obs_n = env.reset()
        while not all(done_n):
            action = env.action_space.sample()
            s, reward_n, s_prime, done_n = env.step(action)
            ep_reward += reward_n
            writer.writerow([str(s), str(action[0]), str(reward_n), str(s_prime)])

init_time = time.time()
n_threads = 1000
output_files = [f'./game_{i}.csv' for i in range(n_threads)]

for i in output_files:
    run_game(i)
print(time.time()-init_time)
