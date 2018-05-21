import sys
import numpy as np
from time import clock

clothes = {
	0:{"name":"shirt","type":"U","order":1},
	1:{"name":"sweater","type":"U","order":2},
	2:{"name":"socks","type":"F","order":1},
	3:{"name":"shoes","type":"F","order":2}
}


states = [
	"RRRR",
	"URRR",
	"RRFR",
	"UURR",
	"URFR",
	"RRFF",
	"UUFR",
	"URFF",
	"UUFF"
]

'''
This function is used to build the Transition Diagram.(tDiag)
I/P: states variable, O/P: returns transition dictionary.
For reference check HEADER(2)
'''
def buildTransitionDiag(states):
	try:
		tDiag = {}
		numStates = len(states)
		for i in range(0,numStates):
			tDiag[states[i]] = []

		for i in range(0, numStates):
			s1 = states[i]
			for j in range(i, numStates):
				s2 = states[j]
				if isValidTransition(s1, s2):
					tmp1 = s2[:] # create a duplicate
					tmp2 = s1[:]
					tDiag[s1].append(tmp1)
					tDiag[s2].append(tmp2)
		return tDiag
	except IndexError as err:
		print "ERROR in buildTransitionDiag()"
		print sys.exc_info()


def isValidTransition(state1, state2):
	try:
		if state1 == state2:
			return False
		diffCount = 0
		for i in range(0, len(state1)):
			c1 = state1[i]
			c2 = state2[i]
			cInfo = clothes[i]
			# check for right type
			if c1 != cInfo["type"] and c1 != "R":
				return False
			if c2 != cInfo["type"] and c2 != "R":
				return False
			if c1 != c2:
				diffCount += 1
				if diffCount > 1:
					return False
			if i == 1 or i == 3:
				if c1 == cInfo["type"] and state1[i-1] == "R":
					return False
				if c2 == cInfo["type"] and state2[i-1] == "R":
					return False
		#for
		return True
	except Exception as err:
		print("ERROR in isValidTransition()")
		print(sys.exc_info())


def buildRMatrix(tD):
	try:
		R = {}
		for s1 in states:
			reward = {}
			legalMoves = tD[s1]
			for s2 in states:
				value = 0
				if s2 not in legalMoves:
					value = -1
				elif s2 == "UUFF":
					value = 100
				reward[s2] = value
			#for
			R[s1] = reward
		#for
		return R

	except Exception as err:
		print("ERROR in buildRMatrix()")
		print(sys.exc_info())


def getIndexInStates(state):
	try:
		for i in range(0, len(states)):
			if state == states[i]:
				return i
	except IndexError as err:
		print("ERROR in getIndexInStates()")
		print(sys.exc_info())

'''
Used by learn_Q to compute the maximum expected value of being in state s
'''
def computeMaxExpected(QMatrix, R, state):
	try:
		possibleActions = states
		maxSoFar = 0
		i = getIndexInStates(state)
		for action in possibleActions:
			j = getIndexInStates(action)
			value = QMatrix[i][j]
			if value > maxSoFar:
				maxSoFar = value
		#for
		return maxSoFar
	except Exception as err:
		print("ERROR in computeMaxExpected()")
		print(sys.exc_info())

'''
Q-Learning Function.
This function takes as input the R-Matrix, gamma, alpha and Number of Episodes to train Q for.
It returns the Q-Matrix as output.
'''
def learn_Q(R, gamma = 0.8, alpha = 0.0, numEpisodes = 0):
	try:
		print numEpisodes
		statesLength = len(states)
		dim = (statesLength, statesLength)
		Q = np.zeros(dim, dtype=int)
		#print 'DEBUG 1'
		seed = int(clock() * 100)
		np.random.seed(seed) # use current CPU time as seed
		#print "DEBUG 2"

		#print Q
		#simulated execution
		start = np.random.randint(0, len(states))
		state = states[start]
		for i in range(0, numEpisodes):
			reward = R[state]
			#print "DEBUG 3"
			# randomly select action a where R >= 0
			moves = []
			for action in reward.keys():
				if reward[action] >= 0:
					moves.append(action)
			resultingState = moves[0]
			#print "DEBUG 4"
			#print moves
			while True:
				resultingState = moves[np.random.randint(0, len(moves))]
				if R[state][resultingState] >= 0:
					break
			#while
			maxExpected = gamma * computeMaxExpected(Q, R, resultingState)
			i = getIndexInStates(state)
			j = getIndexInStates(resultingState)
			update = (1 - alpha) * Q[i][j] + alpha * (R[state][resultingState] + maxExpected)
			Q[i][j] = update
			state = resultingState
			#print "DEBUG 7"
		#for

		return Q

	except Exception as err:
		print("ERROR in learn_Q()")
		#print "Reward = " + str(reward)
		#print "Possibilities = " + str(possibilities)
		print(sys.exc_info())


def main(argv=sys.argv):
	try:
		#s1 = argv[1]
		#s2 = argv[2]
		#print(isValidTransition(s1, s2))
		#enumerateResultingStates(s1)
		tDiag = buildTransitionDiag(states)
		RMatrix = buildRMatrix(tDiag)
		for state in RMatrix.keys():
			print "State = " + state
			print(RMatrix[state])
		matrix = learn_Q(RMatrix, alpha = 0.8, numEpisodes = 200)
		print matrix

	except Exception as err:
		print("ERROR in main()")
		print(sys.exc_info())


if __name__ == "__main__":
	main()
