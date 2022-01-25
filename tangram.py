import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Tangram(object):
	def __init__(self,
			 	 seed,
		     	 n_grid = 10, # grid size
		     	 n_blocks = 4, # how many primitive blocks to use in a silhouette
				 n_possible_blocks = 6 # number of blocks to choose from, maximum 6
		     	):
		np.random.seed(seed)
		self.__n_grid = n_grid
		self.__n_blocks = n_blocks
		self.__n_possible_blocks = n_possible_blocks
		
		# specify bounds on grid - to control we are not moving outside of grid
		self.__up_bound = np.arange(0,n_grid**2-(n_grid-1),n_grid)
		self.__low_bound = np.arange((n_grid-1),n_grid**2,n_grid)
		self.__left_bound = np.arange(n_grid)
		self.__right_bound = np.arange(n_grid**2-n_grid,n_grid**2)

		final_Form, final_Coord = self.__instantiate(n_grid, n_blocks, n_possible_blocks)

		self.__current_form = np.array([], dtype=int)
		self.__state = np.zeros((4,n_grid,n_grid))
		self.__state[0] = final_Form
		self.__state[2] = final_Coord
		self.__used_blocks = np.array([], dtype=int)
	
	def get_dims(self):
		return self.__n_grid, self.__n_blocks, self.__n_possible_blocks
	
	def __get_form_block(self, starting_point):

		form_block = [
					 [starting_point-1, starting_point,            			starting_point+(self.__n_grid-1)],
					 [starting_point-1, starting_point+(self.__n_grid-1), 	starting_point+self.__n_grid],
					 [starting_point,   starting_point+self.__n_grid,     	starting_point+(self.__n_grid-1)],
					 [starting_point,   starting_point-1,          			starting_point+self.__n_grid],
					 [starting_point,   starting_point+self.__n_grid,		starting_point+self.__n_grid*2],
					 [starting_point,   starting_point-1,          			starting_point-2]
					 ]

		return form_block[:self.__n_possible_blocks]

	def __instantiate(self, n_grid, n_blocks, n_possible_blocks):
		
		# find the middle of the big grid to place the first building block
		starting_point = int(n_grid**2/2-(n_grid/2))
		
		# define shape of primite building blocks:
		form_block = self.__get_form_block(starting_point)
		
		# no hierarchies (4 primitive building blocks)
		blocks = np.random.choice(np.arange(n_possible_blocks), size=n_blocks, replace=False)
		blocks = np.random.permutation(blocks)

		block_code = [(x+1) for x in blocks] # +1 because 0 = background

		
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

				# put left bottom part (random choice) of next BB onto chosen adjacent point
				next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - form_block[blocks[idx_BB]][0]

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

		return final_Form, final_Coord

	
	def get_state(self):
		return self.__state
	
	def revert(self, time=1):
		if time > 0:
			deleted_blocks = self.__current_form[-time*3:]
			self.__current_form = self.__current_form[:-time*3]
			self.__used_blocks = self.__used_blocks[:-time]
			for t in range(time):
				deleted_block = deleted_blocks[3*t:3*(t+1)]
				self.__state[1][np.unravel_index(deleted_block , (self.__n_grid,self.__n_grid), order='F')] = 0
				self.__state[3][np.unravel_index(deleted_block , (self.__n_grid,self.__n_grid), order='F')] = 0
	
	def step(self, action):
		
		self.__used_blocks = np.concatenate((self.__used_blocks, np.array([action[0]])))

		block_code = [(x+1) for x in self.__used_blocks] # +1 because 0 = background

		starting_point = np.ravel_multi_index((action[1],action[2]), (self.__n_grid,self.__n_grid), order='F')

		# define shape of primite building blocks:
		form_block = self.__get_form_block(starting_point)

		next_block = np.array(form_block[self.__used_blocks[-1]])

		self.__current_form = np.concatenate((self.__current_form, next_block), axis=0) # concatenate new block

		self.__state[1][np.unravel_index(self.__current_form, (self.__n_grid,self.__n_grid), order='F')] = 1 # move from linear index into grid (matrix)

		# obtain coordinate information in reduced grid
		self.__state[3][np.unravel_index(next_block, (self.__n_grid,self.__n_grid), order='F')] = block_code[-1] # move from linear index into grid (matrix)

		reward = 0.0
		done = False

		if ~np.any(self.get_mask()) or len(self.__used_blocks) >= self.__n_blocks:
			done = True
			dist = distance.cityblock(self.__state[0].flatten(), self.__state[1].flatten())
			if dist > 0:
				reward = -1.0
			else:
				reward = 1.0

		return self.__state, reward, done


	def get_mask(self):
		mask = np.zeros(self.__n_possible_blocks*self.__n_grid*self.__n_grid)
		for block in range(self.__n_possible_blocks):
			if block not in self.__used_blocks:
				for y in range(self.__n_grid):
					for x in range(self.__n_grid):
						if block==0 and y>0 and x < self.__n_grid-1:
							if self.__state[0,y-1,x] and self.__state[0,y,x] and self.__state[0,y-1,x+1] and\
								1-self.__state[1,y-1,x] and 1-self.__state[1,y,x] and 1-self.__state[1,y-1,x+1]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
						if block==1 and y>0 and x < self.__n_grid-1:
							if self.__state[0,y-1,x] and self.__state[0,y-1,x+1] and self.__state[0,y,x+1] and\
								1-self.__state[1,y-1,x] and 1-self.__state[1,y-1,x+1] and 1-self.__state[1,y,x+1]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
						if block==2 and y>0 and x < self.__n_grid-1:
							if self.__state[0,y,x] and self.__state[0,y,x+1] and self.__state[0,y-1,x+1] and\
								1-self.__state[1,y,x] and 1-self.__state[1,y,x+1] and 1-self.__state[1,y-1,x+1]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
						if block==3 and y>0 and x < self.__n_grid-1:
							if self.__state[0,y,x] and self.__state[0,y-1,x] and self.__state[0,y,x+1] and\
								1-self.__state[1,y,x] and 1-self.__state[1,y-1,x] and 1-self.__state[1,y,x+1]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
						if block==4 and x < self.__n_grid-2:
							if self.__state[0,y,x] and self.__state[0,y,x+1] and self.__state[0,y,x+2] and\
								1-self.__state[1,y,x] and 1-self.__state[1,y,x+1] and 1-self.__state[1,y,x+2]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
						if block==5 and y > 1:
							if self.__state[0,y,x] and self.__state[0,y-1,x] and self.__state[0,y-2,x] and\
								1-self.__state[1,y,x] and 1-self.__state[1,y-1,x] and 1-self.__state[1,y-2,x]:
								mask[block*self.__n_grid*self.__n_grid + y*self.__n_grid + x] = 1
		
		return mask
	
	def reset(self):

		final_Form, final_Coord = self.__instantiate(self.__n_grid, self.__n_blocks, self.__n_possible_blocks)

		self.__current_form = np.array([], dtype=int)
		self.__state = np.zeros((4,self.__n_grid,self.__n_grid))
		self.__state[0] = final_Form
		self.__state[2] = final_Coord
		self.__used_blocks = np.array([], dtype=int)
	
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
