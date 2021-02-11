import numpy as np
import pdb
import random
import math

import torch
from torch import nn
from torch.autograd import Variable
from memory import EpisodicReplayMemory
from model import ActorCritic

STATE_SPACE = 22 * 5 + 1
ACTION_SPACE = 22 * 3 * 3
NUM_LAYERS = 2


def isFloat(a):
    try:
        float(a)
        return True
    except ValueError:
        return False


class Parser:
	def __init__(self):

		self.questions = []
		self.imageNames = []
		self.option1s = []
		self.option2s = []
		self.body_coords = []
		self.prefixes = []
		self.layer = 1

		self.correctAnswer = 0
		self.rewardsWithoutE = []

		self.states = []
		self.hm = {}
		self.memory = []
		self.output = []

		self.epsilon = 0
		self.minLen = 5
		self.maxLen = 30
		self.numNode = 22   # [nodeId, bubbleSiz, bubbleScale]
		self.bubbleSize = 3 
		self.bubbleScale = 3



	''' 
	input: initial csv files array
	output: none
	side effect: initialize parser parameters with inital csv files
	'''
	def parseInit(self, args):
		for arg in args:
			state = []
			i = 0
			for line in open(arg):
				if (i > 0):
					curRow = line.rstrip().split(',')
					if (not isFloat(curRow[2])):   # last row for other info
						state.append(float(curRow[0]))
						self.imageNames.append(curRow[1])
						self.questions.append(curRow[2])
						self.option1s.append(curRow[3])
						self.option2s.append(curRow[4])
						imageId = 0
						if (isFloat(curRow[1][7])):
							imageId = curRow[1][6: 8]
						else:
							imageId = curRow[1][6]
						self.body_coords.append('task{0}_img{1}_all_layer_coords.csv'.format(curRow[1][4], imageId))
						self.prefixes.append('task{0}_{1}_{2}_0-4_'.format(curRow[1][4], imageId, self.layer))
					else:
						self.hm[i] = curRow[1]
						curRow = curRow[0:1] + curRow[2:] + [0] 
						
						curRow = (list(map(float, curRow)))
						state = state + curRow
				i += 1
			self.states.append(state)

	''' 
	input: number of samples to generate
	output: none
	side effect: generate output array, prepared to write into output files
	'''
	def generateRandomDataset(self,num):
		hidden_size = 32
		model = torch.load('training_cps/training1_2_layer2_1-0_270000.pt')
		# model = ActorCritic(STATE_SPACE, ACTION_SPACE, hidden_size, NUM_LAYERS)
		# pre_policy = np.zeros((1, 198))
		# pre_policy = torch.FloatTensor(pre_policy)
		for index in range(len(self.states)):
			state = self.states[index]
			pre_state = torch.FloatTensor(np.zeros((1, 111)))
			pre_actionVal = 0
			# file = open('policy.txt', 'w')
			for i in range(1, num + 1):
				count = 0
				hx = Variable(torch.zeros(1, hidden_size))
				# print(hx)
				cx = Variable(torch.zeros(1, hidden_size))
				# state = self.state
				rand = random.randint(self.minLen, self.maxLen)
				reward = random.uniform(0,10)
				tensorState = torch.zeros(1, STATE_SPACE)
				minS = float("inf")
				for timestep in range(1, rand):  # each episode

					tensorState = torch.FloatTensor(np.array(state)).view(1, STATE_SPACE) # new state				
					actionSingleVal = random.randint(0, 197)
					policy, _, _, (hx, cx) = model(Variable(tensorState), (hx, cx))
					#pre_policy = policy
					prob = random.uniform(0, 1)
					if (prob > self.epsilon):     # for prob epsilon choose action
						# file.write(str(tensorState - pre_state) + '\n')
						actionSingleVal = policy.max(1)[1].data[0]

					if (actionSingleVal == pre_actionVal):
						count += 1
						if (count > 2):
							randVal = random.randint(1, 22)
							actionSingleVal = (actionSingleVal + 7 * randVal) % 198
					else:
						count = 0	
					pre_actionVal = actionSingleVal	
					action = Parser._convertSingleToArray(actionSingleVal)
					question = action[0]		

					
					# print(question)
					self.output.append([self.prefixes[index] + 'trial_' + str(i) + '_' + str(timestep), 'Show me ' + self.hm[question],
										str(action[0]), str(action[1]), str(action[2]), 
						                self.imageNames[index], self.questions[index], self.option1s[index], self.option2s[index], 
						                self.body_coords[index]])
					# pre_policy = policy.data
					pre_state = tensorState
					# print(action[0])
					state[(action[0] - 1) * 5 + 4] = 1
					
					
				self.output.append([])

	''' 
	input: output file name array
	output: none
	side effect: write output array constructed in generateRandomDataset to output files 
			specified by output file name array
	'''
	def writeToFile(self, arg):
		file = open(arg,'w')
		file.write('timestep,question ,action_node_id, bubble_size, bubble_scale, '\
				   'image_name, question, option1, option2,body_coords\n')
		for item in self.output:
			if (len(item) > 0): # not finished for current episode
				string = ''
				for feature in item:
					string += feature + ','
				file.write(string[:-1])
			else:
				string = ''
				for i in range(10): # fill 50 in the empty line
					string += str(-1) + ','
				file.write(string[:-1])
			# for 4 more columns to indicate 4 types of additional info
			file.write('\n')

	''' 
	input: output file name array
	output: none
	side effect: write output array constructed in generateRandomDataset to storage array
	'''
	def _getStorage(self, several_outputs):
		scalaHm = {0: 1, 1: 9, 2: 15}
		sizeHm = {0: 1.45, 1: 2.15, 2: 3.15}
		hidden_size = 32
		model = ActorCritic(STATE_SPACE, ACTION_SPACE, hidden_size, NUM_LAYERS)
		memory_capacity = 10000
		max_episode_length = 10
		self.memory = EpisodicReplayMemory(memory_capacity, max_episode_length)

		states = self.states


		# storage will have format [[[], [], []], [episode2], [], []]. the last 
		# element of each epsidoe is [tensorState, final reward]
		storage = []   
		storage.append([])
		episodeI = 0
		
		for index in range(len(states)):
			state = states[index]
			hx = Variable(torch.zeros(1, hidden_size))
			cx = Variable(torch.zeros(1, hidden_size))
			tensorState = torch.zeros(1, STATE_SPACE)
			minS = float("inf")
			with open(several_outputs[index]) as f:
				next(f)
				for line in f:
					curRow = line.rstrip().split(',')
					if (curRow[0] == '-1' or curRow[0] == '50'):  # end of an episode
						sigma = minS / (len(storage[episodeI]))
						entropy = 1 + 0.5 * math.log(39.4 * sigma)
						# plugging in same reward for all turns in episode... the ACER code uses lambda return to scale them
						# print(episodeI)
						storage[episodeI].append([tensorState, self.rewardsWithoutE[episodeI] + entropy])
						episodeI += 1
						# print(episodeI)
						if (episodeI >= 1583):
							break
						hx = Variable(torch.zeros(1, hidden_size))
						cx = Variable(torch.zeros(1, hidden_size))
						state = states[index]  # prepare for next episode
						storage.append([])
						minS = float("inf")
					else:

						tensorState = torch.FloatTensor(np.array(state)).view(1, STATE_SPACE)
						[action_node_id, bubble_size, bubble_scale] = list(map(int, (curRow[2:5])))
						actionSingleVal = (action_node_id - 1) * self.bubbleScale * self.bubbleSize +  \
										   bubble_size * self.bubbleScale + bubble_scale

						minS = min(minS, pow(scalaHm[bubble_scale], 2) * pow(sizeHm[bubble_size], 2))
						policy, _, _, (hx, cx) = model(Variable(tensorState), (hx, cx))
						storage[episodeI].append([tensorState, actionSingleVal, policy.data])
						state[(action_node_id - 1) * 5 + 4] = 1
		return storage		



	''' 
	input: output file name array
	output: none
	side effect: write storage array into memory required for RL + RNN training
	'''
	def writeBackMemory(self, several_outputs):
		storage = self._getStorage(several_outputs)
			# pdb.set_trace()
		for episode in storage:
			last = episode[-1] # last one is reward
			reward = last[1]
			for i in range(len(episode) - 1):
				each = episode[i]
				tensorState = each[0]
				actionSingleVal = each[1]
				policyData = each[2]
				self.memory.append(tensorState, actionSingleVal, reward, policyData)
			tensorState = last[0]
			self.memory.append(tensorState, None, None, None)
		# pdb.set_trace()


	''' 
	input: output file name array, reward file name array
	output: none
	side effect: append bubble lengths to the end of each row of corresponding reward files
	'''
	def appendToAMTRewardsAndBubbleLen(self, several_outputs, several_rewards):
		storage = self._getStorage(several_outputs)

		# append to file
		episodeI = 0
		for arg in several_rewards: 

			inputFile = open('AMT_rewards/' + arg, 'r')
			outputFile = open('appended_AMT_rewards/' + arg[:-4] + '_appended.csv', 'w')
			firstLine = inputFile.readline()
			# firstLine.rstrip('\n') + ', "bubbleLen", "reward"\n'
			outputFile.write(firstLine)
			for line in inputFile:
				episode = storage[episodeI]
				episodeLen = len(episode) - 1
				last = episode[-1] # last one is reward
				reward = last[1]
				outputFile.write(('{0},{1},{2}\n').format(line.rstrip('\n'), str(episodeLen), str(reward)))
				episodeI += 1

	''' 
	input: output file name array
	output: none
	side effect: append discourse information and structure to the end of each row of output files
	'''
	def appendDiscourseAndStructure(self, structureFile, several_outputs):
		# storage = self._getStorage(several_outputs)
		# append to file
		parentHm, childrenHm = Parser._extractStructure(structureFile)
		for arg in several_outputs: 
			inputFile = open(arg, 'r')
			outputFile = open('appended_outputs/' + arg[:-4] + '_appended.csv', 'w')
			firstLine = inputFile.readline()
			outputFile.write(firstLine.rstrip() + ', structure, discourse\n')
			pre_acton_node_id, pre_bubble_size, pre_bubble_scale = None, None, None
			index = 0
			for line in inputFile:
				appended = ''
				curRow = line.rstrip().split(',')
				if (curRow[0] == '-1' or curRow[0] == '50'):  # end of an episode
					appended = ', NA, NA'

				else:
					[action_node_id, bubble_size, bubble_scale] = list(map(int, (curRow[2:5])))
					if pre_acton_node_id == None:
						appended = ', NA, NA'
					else:
						if (parentHm.get(pre_acton_node_id) == action_node_id):
							appended += ', bottomUp, '
						elif (childrenHm.get(pre_acton_node_id) != None and 
						      action_node_id in childrenHm[pre_acton_node_id]):
							appended += ', topdown, '
						elif (action_node_id == pre_acton_node_id):
							appended += ', alpha, '
						else:
							appended += ', NA, '

						if (action_node_id == pre_acton_node_id):
							if (pre_bubble_scale == bubble_scale and pre_bubble_size == bubble_size):
								appended += 'recurrence'
							elif (bubble_scale > pre_bubble_scale):
								appended += 'elaboration'
							elif (bubble_scale < pre_bubble_scale and bubble_size > pre_bubble_size):
								appended += 'summary'
							else:
								appended += 'restatement'
						else:
							appended += 'sequence'

				pre_acton_node_id, pre_bubble_size, pre_bubble_scale = action_node_id, bubble_size, bubble_scale
				outputFile.write(line.rstrip('\n') + appended + '\n')

	''' 
	input: reward file name array
	output: none
	side effect: extract reward with entropy to parser internal array
	'''
	def readAMTBatch(self, args):
		i = 0
		for arg in args:
			with open('AMT_rewards/' + arg) as f:
				next(f)
				for line in f:
					i += 1
					curRow = line.rstrip().split(',')
					Qvalues = curRow[-3:]
					[Q1, Q2, Q3] = list(map(Parser._extractNumber, Qvalues))
					# always Q1 should be the correct answer
					rewardWithoutE = (3 - 2 * Q1) * Q2 * Q3 + 3.6 * (2 - 2 * Q1)
					self.rewardsWithoutE.append(rewardWithoutE)
		#pdb.set_trace()

	''' 
	input: reward file
	output: none
	side effect: extract the top20 reward actions and return those (state, action) pair
	'''
	def extractFirst20(self, rewardFile):
		inputFile = open('appended_AMT_rewards/' + rewardFile, 'r')
		outputFile = open(rewardFile[0:-4] + '_extracted.csv', 'w')
		firstLine = inputFile.readline()
		outputFile.write(firstLine)
		lines = []
		index = 0
		for line in inputFile:
			curRow = line.rstrip().split(',')
			lines.append([float(curRow[-1]), index, line])
			index += 1
		lines = sorted(lines, key = lambda x : -x[0])
		retindexes = []
		for i in range(20):
			outputFile.write(lines[i][-1])
			retindexes.append(lines[i][1])
		return retindexes


	''' 
	input: number of (state, action) pair to generate for each fixed length
	output: none
	side effect: extract the top20 reward actions and return those indexes
	'''
	def generateDifferentLenBatches(self, num, fixedLens):
		hidden_size = 32
		# training_cps/supervised_1_1_layer1_0-0_100000.pt
		model = torch.load('training_cps/training1_2_layer1_1-0_980000.pt')
		# model = ActorCritic(STATE_SPACE, ACTION_SPACE, hidden_size, NUM_LAYERS)
		# pre_policy = np.zeros((1, 198))
		# pre_policy = torch.FloatTensor(pre_policy)
		state = self.states[0]
		pre_state = torch.FloatTensor(np.zeros((1, 111)))
		pre_actionVal = 0
		# file = open('policy.txt', 'w')
		index = 0
		for fixedLen in fixedLens:
			for i in range(1, num + 1):
				count = 0
				hx = Variable(torch.zeros(1, hidden_size))
				cx = Variable(torch.zeros(1, hidden_size))
				reward = random.uniform(0,10)
				tensorState = torch.zeros(1, STATE_SPACE)
				minS = float("inf")
				for timestep in range(1, fixedLen + 1):  # each episode
					tensorState = torch.FloatTensor(np.array(state)).view(1, STATE_SPACE) # new state				
					actionSingleVal = random.randint(0, 197)
					policy, _, _, (hx, cx) = model(Variable(tensorState), (hx, cx))
					#pre_policy = policy
					prob = random.uniform(0, 1)
					if (prob > self.epsilon):     # for prob epsilon choose action
						# file.write(str(tensorState - pre_state) + '\n')
						actionSingleVal = policy.max(1)[1].data[0]

					if (actionSingleVal == pre_actionVal):
						count += 1
						if (count > 2):
							randVal = random.randint(1, 22)
							actionSingleVal = (actionSingleVal + 7 * randVal) % 198
					else:
						count = 0	
					pre_actionVal = actionSingleVal	
					# print(actionSingleVal)
					action = Parser._convertSingleToArray(actionSingleVal.numpy())
					question = action[0]		

					
					# print(question)
					self.output.append([self.prefixes[index] + str(fixedLen) + '_trial_' + str(i) + '_' + str(timestep), 'Show me ' + self.hm[question],
										str(action[0]), str(action[1]), str(action[2]), 
						                self.imageNames[index], self.questions[index], self.option1s[index], self.option2s[index], 
						                self.body_coords[index]])
					# pre_policy = policy.data
					pre_state = tensorState
					# print(action[0])
					state[(action[0] - 1) * 5 + 4] = 1
					
					
				self.output.append([])

	''' 
	input: reward file
	output: none
	side effect: write back storage structure to memory required for RNN + RL training
	'''
	def writeBackMemoryExtracted20(self, several_outputs):
		storage = self._getStorage(several_outputs)
			# pdb.set_trace()
		retindexes = self.extractFirst20('AMT1_1_layer2_1-0_appended.csv')
		for index in retindexes:
			episode = storage[index]
			last = episode[-1] # last one is reward
			reward = last[1]
			for i in range(len(episode) - 1):
				each = episode[i]
				tensorState = each[0]
				actionSingleVal = each[1]
				policyData = each[2]
				self.memory.append(tensorState, actionSingleVal, reward, policyData)
			tensorState = last[0]
			self.memory.append(tensorState, None, None, None)
		# pdb.set_trace()


	''' 
	input: node hierachy file
	output: extract structure relationship of each node from input file and return the parent and children map
	'''
	@staticmethod
	def _extractStructure(structureFile):
		file = open(structureFile, 'r')
		childrenHm, parentHm = dict(), dict()
		next(file)
		for line in file:
			curRow = line.rstrip().replace('"', '').split(',')
			node_id = int(curRow[0])
			parent = 'NA' if curRow[2] == 'NA' else int(curRow[2])
			children = 'NA' if curRow[3] == 'NA' else list(map(int, curRow[3:]))
			if (parent != 'NA'):
				parentHm[node_id] = parent
			if (children != 'NA'):
				childrenHm[node_id] = children
		return parentHm, childrenHm


	''' 
	input: action single value
	output: convert single value to action value array
	'''
	@staticmethod
	def _convertSingleToArray(actionSingleVal):
		action = []
		question = actionSingleVal // 9 + 1
		action.append(question)
		actionSingleVal = actionSingleVal % 9
		action.append(actionSingleVal // 3)
		action.append(actionSingleVal % 3)	
		return action


	@staticmethod
	def _extractNumber(val):
		return int(list(filter(str.isdigit, val))[0])



if __name__ == '__main__':
	# parser = Parser()
	# # several_csvs = ['initial_csvs/Task1_6.csv', 'initial_csvs/Task1_8.csv', 
	# # 				  'initial_csvs/Task1_9.csv', 'initial_csvs/Task1_10.csv', 'initial_csvs/Task1_11.csv']
	# several_csvs = ['initial_csvs/Task1_3.csv', 'initial_csvs/Task1_4.csv', 'initial_csvs/Task1_5.csv']
	# parser.parseInit(several_csvs)

	# # parser.generateRandomDataset(100)
	# # parser.writeToFile('outputs/output1_several_layer1_0-4.csv')


	# several_rewards = ['AMT1_345_layer1_0-8.csv']
	# parser.readAMTBatch(several_rewards)
	# several_outputs = ['output1_3_layer1_0-8.csv', 'output1_4_layer1_0-8.csv', 'output1_5_layer1_0-8.csv']
	# # parser.writeBackMemory(several_outputs)

	# # generate for columns with rewards and bubble number
	# parser.appendToAMTRewardsAndBubbleLen(several_outputs, several_rewards)



	# non-functional part
	parser = Parser()
	# several_csvs = ['initial_csvs/Task1_6.csv', 'initial_csvs/Task1_8.csv', 
	# # # 				  'initial_csvs/Task1_9.csv', 'initial_csvs/Task1_10.csv', 'initial_csvs/Task1_11.csv']
	several_csvs = ['initial_csvs/Task1_1.csv']
	parser.parseInit(several_csvs)
	# several_rewards = ['AMT1_1_layer2_1-0.csv']
	# parser.readAMTBatch(several_rewards)
	# # several_outputs = ['outputs/output1_1_layer2_1-0.csv','outputs/output1_3_layer2_0-8.csv', 
	# # 				   'outputs/output1_4_layer2_0-8.csv', 'outputs/output1_5_layer2_0-8.csv', 'outputs/output1_2_layer2_1-0.csv']

	# several_outputs = ['outputs/output1_1_layer2_1-0.csv']
	# parser.writeBackMemoryExtracted20(several_outputs)



	# # generate for columns with rewards and bubble number
	# parser.appendToAMTRewardsAndBubbleLen(several_outputs, several_rewards)
	# # generate plots 
	# # parser.appendDiscourseAndStructure('node_hierarchy.csv', several_outputs)



	# # generate one hundread batches
	fixedLens = [35, 31, 28, 25, 21, 18, 15, 11, 9]
	# fixedLens = [40]
	# fixedLens = [18]
	parser.generateDifferentLenBatches(100, fixedLens)
	# RNN_output1_1_1_Len100_0-0.csv
	parser.writeToFile('output1_1_{0}_severalFixedLen_0-0.csv'.format(parser.layer))

	
