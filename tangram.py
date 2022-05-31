import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


class Tangram(object):
    def __init__(self,
                 seed,
                 n_grid=20,  # grid size
                 n_blocks=4,  # how many primitive blocks to use in a silhouette
                 n_possible_blocks=6,  # number of blocks to choose from, maximum 6
                 chunk_type=-1,
                 n_blocks_H=2,  # how many chunks to use for silhouette
                 n_distinct_samples=20,  # 20 distinct samples for 2 chunks, 690 distinct samples for 1 chunk, ~400,000 samples for no chunk
                 n_samples=40
                 ):
        np.random.seed(seed)
        self.__n_grid = n_grid
        self.__n_blocks = n_blocks
        self.__n_possible_blocks = n_possible_blocks
        self.__chunk_type = chunk_type
        self.__n_blocks_H = n_blocks_H
        self.__n_train = int(n_distinct_samples*0.8)
        self.__n_test = n_distinct_samples - self.__n_train

        # specify bounds on grid - to control we are not moving outside of grid
        self.__up_bound = np.arange(0, n_grid**2-(n_grid-1), n_grid)
        self.__low_bound = np.arange((n_grid-1), n_grid**2, n_grid)
        self.__left_bound = np.arange(n_grid)
        self.__right_bound = np.arange(n_grid**2-n_grid, n_grid**2)

        unbiased = False
        thresh = 0.0
        print('Generating an unbiased Tangram environment...')
        while not unbiased:

            self.train = {}
            self.__distinct_train = set()
            self.test = {}
            self.__distinct_test = set()
            
            while len(self.train) + len(self.test) < n_samples:
                final_Form, final_Coord, current_form = self.__instantiate(
                    n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H)
                new_sample = tuple(final_Form.flatten())
                new_sample_cropped = tuple(self.mkCrop(final_Form).flatten())
                if new_sample_cropped not in self.__distinct_train and new_sample_cropped not in self.__distinct_test and len(self.__distinct_train) < self.__n_train:
                    self.train.update({new_sample: [final_Form, final_Coord, current_form]})
                    self.__distinct_train.add(new_sample_cropped)
                elif new_sample_cropped not in self.__distinct_test and new_sample_cropped not in self.__distinct_train and len(self.__distinct_test) < self.__n_test:
                    self.test.update({new_sample: [final_Form, final_Coord, current_form]})
                    self.__distinct_test.add(new_sample_cropped)
                """
                elif new_sample not in self.train.keys() and new_sample_cropped not in self.__distinct_test:
                    self.train.update({new_sample: [final_Form, final_Coord, current_form]})
                elif new_sample not in self.test.keys() and new_sample_cropped not in self.__distinct_train:
                    self.test.update({new_sample: [final_Form, final_Coord, current_form]})
                """

            unbiased = self.is_unbiased(thresh)
            thresh += 0.001

        self.reset('train')
        print("Uniformity threshhold: {:.2f}%".format(100*thresh))

    def get_dims(self):
        return self.__n_grid, self.__n_blocks, self.__n_possible_blocks, self.__chunk_type,  self.__n_blocks_H

    # @title Helper function crop silhouette mkCrop()
    # returns form without additional padding, also size info
    def mkCrop(self, FORM, output='reduced'):

        min_x = np.min(np.where(np.sum(FORM, axis=0) != 0))
        max_x = np.max(np.where(np.sum(FORM, axis=0) != 0))

        min_y = np.min(np.where(np.sum(FORM, axis=1) != 0))
        max_y = np.max(np.where(np.sum(FORM, axis=1) != 0))

        FORM_crop = FORM[min_y:max_y+1, min_x:max_x+1]

        range_x = max_x - min_x + 1
        range_y = max_y - min_y + 1

        if output == 'full':
            return FORM_crop, range_x, range_y, min_x, max_x, min_y, max_y
        elif output == 'reduced':
            return FORM_crop

    def is_unbiased(self, thresh):
        unique_blocks = []
        for sample in self.train.keys():
            [_, final_Coord, _] = self.train[sample]
            unique_blocks.append(np.unique(final_Coord))
        unique_blocks = np.concatenate(unique_blocks)
        unique_blocks = sorted(np.unique(unique_blocks))[1:]
        connectivity = np.zeros((len(unique_blocks), len(unique_blocks)))
        for sample in self.train.keys():
            [_, final_Coord, _] = self.train[sample]
            connectivity_new = np.zeros_like(connectivity)
            for x in range(self.__n_grid):
                for y in range(self.__n_grid):
                    block = final_Coord[x, y]
                    if block != 0:
                        con_id = np.where(unique_blocks == block)[0][0]
                        if x < self.__n_grid - 1:
                            block_x = final_Coord[x+1, y]
                            if block_x != block and block_x != 0:
                                con_id_x = np.where(unique_blocks == block_x)[0][0]
                                connectivity_new[con_id, con_id_x] = 1
                                connectivity_new[con_id_x, con_id] = 1
                        if y < self.__n_grid - 1:
                            block_y = final_Coord[x, y+1]
                            if block_y != block and block_y != 0:
                                con_id_y = np.where(unique_blocks == block_y)[0][0]
                                connectivity_new[con_id, con_id_y] = 1
                                connectivity_new[con_id_y, con_id] = 1
            connectivity += connectivity_new
        
        if self.__n_blocks_H == 2:
            if self.__chunk_type == 2:
                connections = [connectivity[1,0],connectivity[2,1],connectivity[3,0],connectivity[3,2]]
            elif self.__chunk_type == 3:
                connections = [connectivity[1,0],connectivity[2,0],connectivity[3,1],connectivity[3,2]]
            else:
                connections = connectivity[2:,0:2]
            freq = (np.max(connections)-np.min(connections))/np.max(connections)
            if freq > thresh:
                return False
        elif self.__n_blocks_H == 1:
            BB1 = self.__solutions_HBB[0][0]
            BB2 = self.__solutions_HBB[0][1]
            BB3 = self.__solutions_HBB[1][0]
            BB4 = self.__solutions_HBB[1][1]
            # check if chunks are equally likely
            freq1 = abs(connectivity[BB1,BB2] - connectivity[BB3,BB4])/connectivity[BB1,BB2]
            unused_BBs = np.setdiff1d(np.array(unique_blocks)-1, np.array(self.__solutions_HBB)).astype(int)
            # check if the unused BBs are attached to the chunk BBs uniformly
            connections1 = [connectivity[BB1,unused_BBs[0]],connectivity[BB1,unused_BBs[1]],connectivity[BB2,unused_BBs[0]],connectivity[BB2,unused_BBs[1]]]
            connections2 = [connectivity[BB3,unused_BBs[0]],connectivity[BB3,unused_BBs[1]],connectivity[BB4,unused_BBs[0]],connectivity[BB4,unused_BBs[1]]]
            freq2 = (np.max(connections1)-np.min(connections1))/np.max(connections1)
            freq3 = (np.max(connections2)-np.min(connections2))/np.max(connections2)
            # check if the unused BBs are equally likely to be attached to each other as they are to the chunk BBs
            connections3 = [connectivity[BB1,unused_BBs[0]]+connectivity[BB2,unused_BBs[0]],connectivity[unused_BBs[0],unused_BBs[1]]]
            connections4 = [connectivity[BB3,unused_BBs[0]]+connectivity[BB3,unused_BBs[0]],connectivity[unused_BBs[0],unused_BBs[1]]]
            freq4 = (np.max(connections3)-np.min(connections3))/np.max(connections3)
            freq5 = (np.max(connections4)-np.min(connections4))/np.max(connections4)
            if freq1 > thresh or freq2 > thresh or freq3 > thresh or freq4 > thresh or freq5 > thresh:
                return False
        else:
            connections = connectivity + np.max(connectivity)*np.eye(len(connectivity))
            freq = (np.max(connections)-np.min(connections))/np.max(connections)
            if freq > thresh:
                return False
        print("Connectivity matrix:\n", connectivity)
        return True

    def __get_form_block(self, starting_point, n_grid, chunk_type, n_blocks_H):

        form_BB = [
            [starting_point,   starting_point-1,           starting_point+(n_grid-1)],
            [starting_point-1, starting_point +(n_grid-1), starting_point+n_grid],
            [starting_point,   starting_point +n_grid,     starting_point+(n_grid-1)],
            [starting_point,   starting_point -1,          starting_point+n_grid],
            [starting_point,   starting_point +n_grid,     starting_point+n_grid*2],
            [starting_point,   starting_point -1,          starting_point-2]
        ]

        form_BB_H = []

        if chunk_type == 0:
            solutions_HBB = [[0, 1], [3, 5]]
            form_BB_H = [
                form_BB[0] + [x + n_grid*2 for x in form_BB[1]],
                [x - 1 for x in form_BB[3]] + [x + n_grid*2 for x in form_BB[5]]
            ]

        elif chunk_type == 1:
            solutions_HBB = [[5, 4], [2, 3]]
            form_BB_H = [
                form_BB[5] + [x - n_grid - 3 for x in form_BB[4]],
                form_BB[2] + [x + n_grid - 2 for x in form_BB[3]]
            ]

        elif chunk_type == 2:
            solutions_HBB = [[4, 1], [2, 0]]
            form_BB_H = [
                form_BB[4] + [x - 1 for x in form_BB[1]],
                form_BB[2] + [x + n_grid*2 - 1 for x in form_BB[0]]
            ]

        elif chunk_type == 3:
            solutions_HBB = [[5, 1], [4, 3]]
            form_BB_H = [
                form_BB[5] + [x + n_grid for x in form_BB[1]],
                form_BB[4] + [x + n_grid - 1 for x in form_BB[3]]
            ]

        elif chunk_type == 4:
            solutions_HBB = [[0, 1], [5, 4]]
            form_BB_H = [
                form_BB[0] + [x + n_grid*2 for x in form_BB[1]],
                form_BB[5] + [x - n_grid - 3 for x in form_BB[4]]
            ]

        elif chunk_type == 5:
            solutions_HBB = [[3, 0], [4, 5]]
            form_BB_H = [
                form_BB[3] + [x - n_grid-2 + n_grid for x in form_BB[0]],
                form_BB[4] + [x - n_grid-1 + n_grid*2 for x in form_BB[5]]
            ]

        elif chunk_type == 6:
            solutions_HBB = [[2, 1], [5, 4]]
            form_BB_H = [
                form_BB[2] + [x - n_grid-2 + n_grid for x in form_BB[1]],
                form_BB[5] + [x - n_grid-1 + n_grid*2 for x in form_BB[4]]
            ]

        elif chunk_type == 7:
            solutions_HBB = [[3, 2], [5, 4]]
            form_BB_H = [
                form_BB[3] + [x + n_grid*2 for x in form_BB[2]],
                form_BB[5] + [x - n_grid-1 - n_grid*2 for x in form_BB[4]]
            ]

        if n_blocks_H == 1:
            self.__solutions_HBB = solutions_HBB
            use_HBB = np.random.choice(np.size(form_BB_H, 0), 1)
            unused_BBs = np.setdiff1d(
                np.arange(np.size(form_BB, 0)), np.array(solutions_HBB))
            unused_BBs = np.random.permutation(unused_BBs)

            use_form_BB = [form_BB_H[idx] for idx in use_HBB] + \
                [form_BB[idx] for idx in unused_BBs]

            return (use_HBB, unused_BBs, use_form_BB, solutions_HBB)

        elif n_blocks_H == 2:
            self.__solutions_HBB = solutions_HBB
            return (form_BB_H, solutions_HBB)

        return form_BB[:self.__n_possible_blocks]

    def __instantiate(self, n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H):

        # find the middle of the big grid to place the first building block
        starting_point = int(n_grid**2/2-(n_grid/2))

        # define shape of primite building blocks:
        form_block = self.__get_form_block(
            starting_point, n_grid, chunk_type, n_blocks_H)

        # no hierarchies (4 primitive building blocks)
        if n_blocks_H == 0:
            blocks = np.random.choice(
                np.arange(n_possible_blocks), size=n_blocks, replace=False)
            blocks = np.random.permutation(blocks)

            block_code = [(x+1) for x in blocks]  # +1 because 0 = background

        # hierarchies (2 chunks)
        elif n_blocks_H == 2:
            solutions_HBB = form_block[1]
            form_block = form_block[0]
            blocks = np.random.choice(
                np.arange(np.size(form_block, 0)), size=n_blocks_H, replace=False)
            blocks = np.random.permutation(blocks)

            # +1 because 0 = background, *10 identifies hierarchies
            block_code = [(x+1)*10 for x in blocks]

        else:
            # mixed (1 chunk, 2 building blocks)
            use_HBB = form_block[0]
            unused_BBs = form_block[1]
            solutions_HBB = form_block[3]
            form_block = form_block[2]

            blocks = np.concatenate((use_HBB, unused_BBs), axis=0)

            block_code = [(x+1) for x in blocks]  # +1 because 0 = background
            block_code[0] = block_code[0]*10

            blocks = np.arange(np.size(blocks, 0))

        # initialise form
        final_Form = np.zeros((n_grid, n_grid))

        # initialise coordinates
        final_Coord = np.zeros((n_grid, n_grid))

        # start with first building block
        current_form = np.array(form_block[blocks[0]])

        # obtain coordinate information in reduced grid
        # move from linear index into grid (matrix)\
        if len(current_form) > 3:
            if n_blocks_H ==2:
                final_Coord[np.unravel_index(
                    current_form[:3], (n_grid, n_grid), order='F')] = solutions_HBB[blocks[0]][0] + 1
                final_Coord[np.unravel_index(
                    current_form[3:], (n_grid, n_grid), order='F')] = solutions_HBB[blocks[0]][1] + 1
            else:
                final_Coord[np.unravel_index(
                    current_form[:3], (n_grid, n_grid), order='F')] = solutions_HBB[use_HBB[0]][0] + 1
                final_Coord[np.unravel_index(
                    current_form[3:], (n_grid, n_grid), order='F')] = solutions_HBB[use_HBB[0]][1] + 1
        else:
            final_Coord[np.unravel_index(
                current_form, (n_grid, n_grid), order='F')] = block_code[0]

        for idx_BB in np.arange(1, np.size(blocks, 0)):

            # find possible adjacent starting points for next BB
            # all adjacent pixels left, ontop, right, or below
            adj_points = np.unique(np.array(
                [current_form-n_grid, current_form-1, current_form+n_grid, current_form+1]))
            # can't move 'into' silhouette
            adj_points = adj_points[~np.isin(adj_points, current_form)]
            # can't move out of linear grid
            adj_points = adj_points[adj_points >= 0]
            # can't move out of linear grid
            adj_points = adj_points[adj_points <= n_grid**2]

            # now try them in random order as connection points for next building block:
            adj_points = np.random.permutation(adj_points)

            built = False
            idx_adj = 0

            while built == False:

                # necessary because it's all randomised, ideally should go through all possible combinations
                if idx_adj == len(adj_points):
                    idx_adj = 0

                # put left bottom part (random choice) of next BB onto chosen adjacent point
                conn_point_block = np.random.choice(
                    form_block[blocks[idx_BB]], 1)
                #next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - form_block[blocks[idx_BB]][0]
                next_block = np.array(
                    form_block[blocks[idx_BB]]) + adj_points[idx_adj] - int(conn_point_block)

                # check if we didn't move 'around' grid, if not attach
                if (
                        # did we move outside of grid?
                        all(np.isin(next_block, np.arange(0, n_grid**2))) and
                        # is new BB overlapping with prev shape (can happen due to weird shapes of BBs)
                        all(~np.isin(next_block, current_form)) and
                        # did we accidentally move from bottom to top of box (linear idx!)
                        not(any(np.isin(self.__low_bound, next_block)) and any(np.isin(self.__up_bound, next_block))) and
                        # did we accidentally move from left to right of box (linear idx!)
                        not(any(np.isin(self.__left_bound, next_block))
                            and any(np.isin(self.__right_bound, next_block)))
                ):

                    current_form = np.concatenate(
                        (current_form, next_block), axis=0)  # concatenate new block

                    # obtain coordinate information in reduced grid
                    # move from linear index into grid (matrix)
                    if len(next_block) > 3:
                        final_Coord[np.unravel_index(
                            next_block[:3], (n_grid, n_grid), order='F')] = solutions_HBB[blocks[idx_BB]][0] + 1
                        final_Coord[np.unravel_index(
                            next_block[3:], (n_grid, n_grid), order='F')] = solutions_HBB[blocks[idx_BB]][1] + 1  
                    else:
                        final_Coord[np.unravel_index(
                            next_block, (n_grid, n_grid), order='F')] = block_code[idx_BB]

                    built = True

                else:

                    idx_adj += 1

        # move from linear index into grid (matrix)
        final_Form[np.unravel_index(
            current_form, (n_grid, n_grid), order='F')] = 1

        return final_Form, final_Coord, current_form

    def get_state(self):
        return self.__state

    def revert(self, time=1):
        if self.__extra_step:
            time += 1
        deleted_blocks = self.__current_form[-time*3:]
        self.__current_form = self.__current_form[:-time*3]
        self.__used_blocks = self.__used_blocks[:-time]
        for t in range(time):
            deleted_block = deleted_blocks[3*t:3*(t+1)]
            self.__state[1][np.unravel_index(
                deleted_block, (self.__n_grid, self.__n_grid), order='F')] = 0
            self.__state[3][np.unravel_index(
                deleted_block, (self.__n_grid, self.__n_grid), order='F')] = 0
        self.__extra_step = False

    def step(self, action, one_possible_step=False):

        self.__used_blocks = np.concatenate(
            (self.__used_blocks, np.array([action[0]])))

        # +1 because 0 = background
        block_code = [(x+1) for x in self.__used_blocks]

        starting_point = self.__target_form[action[1]]

        if action[0] == 1:
            starting_point += 1

        # define shape of primite building blocks:
        form_block = self.__get_form_block(starting_point, self.__n_grid, 0, 0)

        next_block = np.array(form_block[self.__used_blocks[-1]])

        self.__current_form = np.concatenate(
            (self.__current_form, next_block), axis=0)  # concatenate new block

        # move from linear index into grid (matrix)
        self.__state[1][np.unravel_index(
            self.__current_form, (self.__n_grid, self.__n_grid), order='F')] = 1

        # obtain coordinate information in reduced grid
        # move from linear index into grid (matrix)
        self.__state[3][np.unravel_index(
            next_block, (self.__n_grid, self.__n_grid), order='F')] = block_code[-1]

        reward = 0.0
        done = False

        mask = self.get_mask()
        if np.sum(mask) == 0 or len(self.__used_blocks) >= self.__n_blocks:
            done = True
            dist = distance.cityblock(
                self.__state[0].flatten(), self.__state[1].flatten())
            if dist > 0:
                reward = -1.0
            else:
                reward = 1.0
        """
		elif np.sum(mask) == 1:
			action = np.argmax(mask)
			block = action//(3*self.__n_blocks)
			loc = action - block*3*self.__n_blocks
			_, reward, done = self.step(np.array([block,loc]), one_possible_step=True)
		"""
        if one_possible_step:
            self.__extra_step = True

        return self.__state, reward, done

    def get_mask(self):
        mask = np.zeros(3*self.__n_blocks*self.__n_possible_blocks)
        for block in range(self.__n_possible_blocks):
            if block not in self.__used_blocks:
                for loc in range(3*self.__n_blocks):
                    starting_point = self.__target_form[loc]
                    if block == 1:
                        starting_point += 1
                    form_block = self.__get_form_block(
                        starting_point, self.__n_grid, 0, 0)
                    next_block = form_block[block]
                    if (
                            all(np.isin(next_block, self.__target_form)) and
                            all(~np.isin(next_block, self.__current_form))
                    ):
                        mask[block*3*self.__n_blocks + loc] = 1

        return mask

    def reset(self, mode='train'):

        if mode == 'train':
            random_sample = random.sample(sorted(self.train), 1)[0]
            [final_Form, final_Coord, current_form] = self.train[random_sample]

        if mode == 'test':
            random_sample = random.sample(sorted(self.test), 1)[0]
            [final_Form, final_Coord, current_form] = self.test[random_sample]

        self.__target_form = current_form
        self.__current_form = np.array([], dtype=int)
        self.__state = np.zeros((4, self.__n_grid, self.__n_grid))
        self.__state[0] = final_Form
        self.__state[2] = final_Coord
        self.__used_blocks = np.array([], dtype=int)
        self.__extra_step = False

    def render(self):

        fig = plt.figure()
        fig_count = 1

        for idx_BB in [0, 1]:

            BB = self.__state[idx_BB, :, :]

            fig.add_subplot(1, 2, fig_count)

            plt.imshow(BB, cmap='Greys')
            plt.axis('off')

            fig_count += 1

        plt.show()
