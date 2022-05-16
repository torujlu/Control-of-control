import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

class Tangram(object):
	def __init__(self,
			 	 seed,
		     	 n_grid = 20, # grid size
		     	 n_blocks = 4, # how many primitive blocks to use in a silhouette
				 n_possible_blocks = 6, # number of blocks to choose from, maximum 6
				 chunk_type = -1,
				 n_blocks_H = 2, # how many chunks to use for silhouette
				 n_samples = 40 # 40 samples for 2 chunks, 693 samples for 1 chunk, ~400,000 samples for no chunk
		     	):
		np.random.seed(seed)
		self.__n_grid = n_grid
		self.__n_blocks = n_blocks
		self.__n_possible_blocks = n_possible_blocks
		self.__chunk_type = chunk_type
		self.__n_blocks_H = n_blocks_H
		self.__n_train = int(n_samples*0.8)
		self.__n_test = n_samples - self.__n_train
		
		# specify bounds on grid - to control we are not moving outside of grid
		self.__up_bound = np.arange(0,n_grid**2-(n_grid-1),n_grid)
		self.__low_bound = np.arange((n_grid-1),n_grid**2,n_grid)
		self.__left_bound = np.arange(n_grid)
		self.__right_bound = np.arange(n_grid**2-n_grid,n_grid**2)

		self.__train = {}
		self.__test = {}

		while len(self.__train) < self.__n_train:
			final_Form, final_Coord, current_form = self.__instantiate(n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H)
			new_sample = tuple(final_Form.flatten())
			if new_sample not in self.__train.keys():
				self.__train.update({new_sample: [final_Form, final_Coord, current_form]})

		while len(self.__test) < self.__n_test:
			final_Form, final_Coord, current_form = self.__instantiate(n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H)
			new_sample = tuple(final_Form.flatten())
			if new_sample not in self.__test.keys() and new_sample not in self.__train.keys():
				self.__test.update({new_sample: [final_Form, final_Coord, current_form]})

		self.reset('train')
	
	def get_dims(self):
		return self.__n_grid, self.__n_blocks, self.__n_possible_blocks, self.__chunk_type,  self.__n_blocks_H
	
	def __get_form_block(self, starting_point, n_grid, chunk_type, n_blocks_H):
		
		form_BB = [
				  [starting_point,   starting_point-1,          starting_point+(n_grid-1)],
				  [starting_point-1, starting_point+(n_grid-1), starting_point+n_grid],
				  [starting_point,   starting_point+n_grid,     starting_point+(n_grid-1)],
				  [starting_point,   starting_point-1,          starting_point+n_grid],
				  [starting_point,   starting_point+n_grid,     starting_point+n_grid*2],
				  [starting_point,   starting_point-1,          starting_point-2]
			 	  ]
		
		form_BB_H = []

		if chunk_type==0:
			solutions_HBB  = [[0,1], [3,5]]
			form_BB_H = [
						form_BB[0] + [x + n_grid*2 for x in form_BB[1]],
						[x -1 for x in form_BB[3]] + [x + n_grid*2 for x in form_BB[5]]
						]

		elif chunk_type==1:
			solutions_HBB  = [[4,5], [3,2]]
			form_BB_H = [
						form_BB[5] + [x - n_grid -3 for x in form_BB[4]],
						form_BB[2] + [x + n_grid -2  for x in form_BB[3]]
						]

		elif chunk_type==2:
			solutions_HBB  = [[1,4], [2,0]]
			form_BB_H = [
						form_BB[4] + [x - 1 for x in form_BB[1]],
						form_BB[2] + [x + n_grid*2 -1  for x in form_BB[0]]
						]

		elif chunk_type==3:
			solutions_HBB  = [[5,1], [4,3]]
			form_BB_H = [
						form_BB[5] + [x + n_grid for x in form_BB[1]],
						form_BB[4] + [x + n_grid -1  for x in form_BB[3]]
						]

		elif chunk_type==4:
			solutions_HBB  = [[0,1], [4,5]]
			form_BB_H = [
						form_BB[0] + [x + n_grid*2 for x in form_BB[1]],
						form_BB[5] + [x - n_grid -3 for x in form_BB[4]]
						]

		elif chunk_type==5:
			solutions_HBB  = [[0,3], [4,5]]
			form_BB_H = [
						form_BB[3] + [x - n_grid-2 + n_grid for x in form_BB[0]],
						form_BB[4] + [x - n_grid-1 + n_grid*2 for x in form_BB[5]]
						]

		elif chunk_type==6:
			solutions_HBB  = [[1,2], [4,5]]
			form_BB_H = [
						form_BB[2] + [x - n_grid-2 + n_grid for x in form_BB[1]],
						form_BB[5] + [x - n_grid-1 + n_grid*2 for x in form_BB[4]]
						]

		elif chunk_type==7:
			solutions_HBB  = [[2,3], [4,5]]
			form_BB_H = [
						form_BB[3] + [x + n_grid*2 for x in form_BB[2]],
						form_BB[5] + [x - n_grid-1 - n_grid*2 for x in form_BB[4]]
						]

		if n_blocks_H == 1:
			use_HBB    = np.random.choice(np.size(form_BB_H,0),1)
			unused_BBs = np.setdiff1d(np.arange(np.size(form_BB,0)),np.array(solutions_HBB))
			unused_BBs = np.random.permutation(unused_BBs)

			use_form_BB = [form_BB_H[idx] for idx in use_HBB] + [form_BB[idx] for idx in unused_BBs]

			return (use_HBB, unused_BBs, use_form_BB)

		elif n_blocks_H == 2:
			return form_BB_H
		
		return form_BB[:self.__n_possible_blocks]

	def __instantiate(self, n_grid, n_blocks, n_possible_blocks, chunk_type, n_blocks_H):
		
		# find the middle of the big grid to place the first building block
		starting_point = int(n_grid**2/2-(n_grid/2))
		
		# define shape of primite building blocks:
		form_block = self.__get_form_block(starting_point, n_grid, chunk_type, n_blocks_H)
		
		# no hierarchies (4 primitive building blocks)
		if n_blocks_H == 0:
			blocks = np.random.choice(np.arange(n_possible_blocks), size=n_blocks, replace=False)
			blocks = np.random.permutation(blocks)

			block_code = [(x+1) for x in blocks] # +1 because 0 = background

		# hierarchies (2 chunks)
		elif n_blocks_H == 2:
			blocks = np.random.choice(np.arange(np.size(form_block,0)), size=n_blocks_H, replace=False)
			blocks = np.random.permutation(blocks)

			block_code = [(x+1)*10 for x in blocks] # +1 because 0 = background, *10 identifies hierarchies

		else:
			# mixed (1 chunk, 2 building blocks)
			use_HBB    = form_block[0]
			unused_BBs = form_block[1]
			form_block = form_block[2]

			blocks = np.concatenate((use_HBB, unused_BBs), axis=0)

			block_code = [(x+1) for x in blocks] # +1 because 0 = background
			block_code[0] = block_code[0]*10

			blocks = np.arange(np.size(blocks,0))
		
		# initialise form
		final_Form = np.zeros((n_grid,n_grid))

		# initialise coordinates
		final_Coord = np.zeros((n_grid,n_grid))

		#start with first building block
		current_form = np.array(form_block[blocks[0]])

		# obtain coordinate information in reduced grid
		final_Coord[np.unravel_index(current_form, (n_grid,n_grid), order='F')] = block_code[0] # move from linear index into grid (matrix)

	

		for idx_BB in np.arange(1,np.size(blocks,0)):

			# find possible adjacent starting points for next BB
			adj_points = np.unique(np.array([current_form-n_grid, current_form-1, current_form+n_grid, current_form+1])) # all adjacent pixels left, ontop, right, or below
			adj_points = adj_points[~np.isin(adj_points,current_form)] # can't move 'into' silhouette   
			adj_points = adj_points[adj_points>=0] # can't move out of linear grid  
			adj_points = adj_points[adj_points<=n_grid**2] # can't move out of linear grid    

			# now try them in random order as connection points for next building block:
			adj_points = np.random.permutation(adj_points)

			built = False
			idx_adj = 0

			while built == False:

				# necessary because it's all randomised, ideally should go through all possible combinations
				if idx_adj==len(adj_points):
					idx_adj = 0

				# put left bottom part (random choice) of next BB onto chosen adjacent point
				conn_point_block = np.random.choice(form_block[blocks[idx_BB]],1)
				#next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - form_block[blocks[idx_BB]][0]
				next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - int(conn_point_block)

				# check if we didn't move 'around' grid, if not attach
				if (
					all(np.isin(next_block,np.arange(0,n_grid**2))) and # did we move outside of grid?
					all(~np.isin(next_block,current_form)) and # is new BB overlapping with prev shape (can happen due to weird shapes of BBs)
					not(any(np.isin(self.__low_bound,next_block)) and any(np.isin(self.__up_bound,next_block))) and # did we accidentally move from bottom to top of box (linear idx!)
					not(any(np.isin(self.__left_bound,next_block)) and any(np.isin(self.__right_bound,next_block))) # did we accidentally move from left to right of box (linear idx!)
				):
			
					current_form = np.concatenate((current_form, next_block), axis=0) # concatenate new block

					# obtain coordinate information in reduced grid
					final_Coord[np.unravel_index(next_block, (n_grid,n_grid), order='F')] = block_code[idx_BB] # move from linear index into grid (matrix)

					built = True
			
				else:
			
					idx_adj += 1

		final_Form[np.unravel_index(current_form, (n_grid,n_grid), order='F')] = 1 # move from linear index into grid (matrix)

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
			self.__state[1][np.unravel_index(deleted_block , (self.__n_grid,self.__n_grid), order='F')] = 0
			self.__state[3][np.unravel_index(deleted_block , (self.__n_grid,self.__n_grid), order='F')] = 0
		self.__extra_step = False
	
	def step(self, action, one_possible_step=False):
		
		self.__used_blocks = np.concatenate((self.__used_blocks, np.array([action[0]])))

		block_code = [(x+1) for x in self.__used_blocks] # +1 because 0 = background

		starting_point = self.__target_form[action[1]]

		if action[0] == 1:
			starting_point += 1

		# define shape of primite building blocks:
		form_block = self.__get_form_block(starting_point, self.__n_grid, 0, 0)

		next_block = np.array(form_block[self.__used_blocks[-1]])

		self.__current_form = np.concatenate((self.__current_form, next_block), axis=0) # concatenate new block

		self.__state[1][np.unravel_index(self.__current_form, (self.__n_grid,self.__n_grid), order='F')] = 1 # move from linear index into grid (matrix)

		# obtain coordinate information in reduced grid
		self.__state[3][np.unravel_index(next_block, (self.__n_grid,self.__n_grid), order='F')] = block_code[-1] # move from linear index into grid (matrix)

		reward = 0.0
		done = False

		mask = self.get_mask()
		if np.sum(mask) == 0 or len(self.__used_blocks) >= self.__n_blocks:
			done = True
			dist = distance.cityblock(self.__state[0].flatten(), self.__state[1].flatten())
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
					form_block = self.__get_form_block(starting_point, self.__n_grid, 0, 0)
					next_block = form_block[block]
					if (
						all(np.isin(next_block,self.__target_form)) and
						all(~np.isin(next_block,self.__current_form))
					):
						mask[block*3*self.__n_blocks + loc] = 1
		
		return mask
	
	def reset(self, mode='train'):
		
		if mode=='train':
			random_sample = random.sample(sorted(self.__train), 1)[0]
			[final_Form, final_Coord, current_form] = self.__train[random_sample]
		
		if mode=='test':
			random_sample = random.sample(sorted(self.__test), 1)[0]
			[final_Form, final_Coord, current_form] = self.__test[random_sample]

		self.__target_form = current_form
		self.__current_form = np.array([], dtype=int)
		self.__state = np.zeros((4,self.__n_grid,self.__n_grid))
		self.__state[0] = final_Form
		self.__state[2] = final_Coord
		self.__used_blocks = np.array([], dtype=int)
		self.__extra_step = False
	
	def render(self):

		fig = plt.figure()
		fig_count = 1

		for idx_BB in [0,1]:

			BB = self.__state[idx_BB,:,:]

			fig.add_subplot(1, 2, fig_count)

			plt.imshow(BB, cmap='Greys')
			plt.axis('off')

			fig_count += 1

		plt.show()
