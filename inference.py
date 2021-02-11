import math
import random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
 
from parser import isFloat
import pdb
 
from memory import EpisodicReplayMemory
from model import ActorCritic
from utils import state_to_tensor
from gettingstarted.settings import EXPLANATION_FILES

STATE_SPACE = 22 * 5 + 1
ACTION_SPACE = 22 * 3 * 3
NUM_LAYERS = 2

MODEL = torch.load(EXPLANATION_FILES + '/training_cps/training1_2_layer1_1-0_990000.pt')
MODEL.eval()
hidden_size = MODEL.state_dict()["lstm.weight_hh"].size()[1]
		
def parseCSV(file):
	state = []
	i = 0
	for line in open(file):
		if (i > 0):
			print(line)
			curRow = line.rstrip().split(',')
			if (not isFloat(curRow[2])):   # last row for other info
				state.append(float(curRow[0]))
			else:
				curRow = curRow[0:1] + curRow[2:] + [0] 
				
				curRow = (list(map(float, curRow)))
				state = state + curRow
		i += 1
	return state

def convertSingleToArray(actionSingleVal):
	action = []
	question = actionSingleVal // 9 + 1
	action.append(question)
	actionSingleVal = actionSingleVal % 9
	action.append(actionSingleVal // 3)
	action.append(actionSingleVal % 3)	
	return action

# state as an array
# hx and cx hidden state
# returns dict of actions and resulting states hidden states
def infer(state, hx, cx):
	print("state:")
	print(state)
	print("hx:")
	print(hx)
	print("cx:")
	print(cx)
	next_state = state.copy()
	hx = torch.FloatTensor(np.array(hx)).view(1, hidden_size)
	cx = torch.FloatTensor(np.array(cx)).view(1, hidden_size)
	actions_dict = {}
	best = True
	revealed_nodes = set()
	for i in range(22):
		if state[i*5 + 4] == 1:
			revealed_nodes.add(i+1)
	pre_actionVal = 0
	while len(actions_dict) < 5:
		tensor_state = torch.FloatTensor(np.array(next_state)).view(1, STATE_SPACE)
		policy, _, _, (hx, cx) = MODEL(Variable(tensor_state), (Variable(hx), Variable(cx)))
		actionSingleVal = policy.max(1)[1].data[0]
		if (actionSingleVal == pre_actionVal):
			randVal = random.randint(1, 22)
			actionSingleVal = (actionSingleVal + 7 * randVal) % 198
		action = convertSingleToArray(actionSingleVal.numpy())
		while action[0] in revealed_nodes:
			randVal = random.randint(1, 22)
			actionSingleVal = (actionSingleVal + 7 * randVal) % 198
			action = convertSingleToArray(actionSingleVal.numpy())
		pre_actionVal = actionSingleVal	
		revealed_nodes.add(action[0])
		next_state[(action[0] - 1) * 5 + 4] = 1
		resulting_state = state.copy()
		resulting_state[(action[0] - 1) * 5 + 4] = 1

		if best:
			actions_dict["best"] = [action, resulting_state, hx.detach().numpy().tolist()[0], cx.detach().numpy().tolist()[0]]
			best = False
		actions_dict[str(action[0])] = [action, resulting_state, hx.detach().numpy().tolist()[0], cx.detach().numpy().tolist()[0]]

	return actions_dict






