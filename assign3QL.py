'''
HEADER:
Introduction to Artificial Intelligence: Assignment 3, Reinforcement Learning(QLearning)
Teach the kid to wear clothes in a proper manner.


(1) Representations of states:
At any given state of the program the state is defined as a 4 character string, where each character
denotes the position of the cloth denoted by that character position.
Clothes Order in string: 0: shirt, 1: sweater, 2: socks, 3: shoes.
Example:
RRRR: Denotes that all the clothes are in the room. (initial state)
UUFF: Denotes that all the clothes have been worn in a proper manner. The shirt and sweater are on
the upper body and the socks and shoes are on the feet.(Final State)


(2) Transition Diagram:
The transition Diagram is stored in a Dictionary where each key denotes a node in the graph and
the value for eack key is a list of all nodes that have a connection from that node.
For example: For the graph (http://www.mrgeek.me/wp-content/uploads/2014/04/directed-graph.png) the dictionary would look like:

tDiag = {
	"A":["B"],
	"B":["C"],
	"C":["E"],
	"D":["B"],
	"E":["D","F"],
	"F":[]
}

'''




#Libraries allowed: Numpy, Matplotlib
#Installed using: pip install numpy matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys # for printing stack state in case of exception
from time import clock


'''
All possible locations for the clothes: "R: Room", "U: Upper Body", "F: Feet"
Clothes to wear along with their type; U: Upper Body, F: Feet:
NOTE: It is "not" required to use this variable.
'''
clothes = {
	0:{"name":"shirt","type":"U","order":1},
	1:{"name":"sweater","type":"U","order":2},
	2:{"name":"socks","type":"F","order":1},
	3:{"name":"shoes","type":"F","order":2}
}

'''
Global variable to store all "possible" states.
Please enter all possible states from part (a) Transition Graph in this variable.
For state reference check HEADER(1)
'''
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
					tDiag[s1].append(tmp1) # add forward transition
					#if s2 != "UUFF":
					tDiag[s2].append(tmp2) # add backward transition
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



'''
This function builds the Reward Matrix R.
Penultimate transition are assigned a high score ~ 100.
Possible transitions are assigned 0.
Transitions not possible are assigned -1.
I/P: transition diagram, O/P: returns R matrix.
About output RMatrix:
	All legal transitions have been assigned an appropriate reward value
	Any illegal transition is not explicitly specified, therefore if the
	caller comes across a transition that is not in RMatrix then it can
	safely assume that such a transition is illegal
'''

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


'''
This function returns the path taken while solving the graph by utilizing the Q-Matrix.
I/P: Q-Matrix. O/P: Steps taken to reach the goal state from the initial state.
NOTE: As you probably infer from the code, the break-off point is 50-traversals.
You'll probably encounter this while finishing this assignment that at the initial stages of training,
it is impossible for the agent to reach the goal stage using Q-Matrix.
This break-off point allows your program to not be stuck in a REALLY-LONG loop.
'''
def solveUsingQ(Q):
	start = initial_state
	steps = [start]
	while start != goal_state:
		start = Q[start,].argmax()
		steps.append(start)
		if len(steps) > 50: break
	return steps

'''

'''
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
		statesLength = len(states)
		dim = (statesLength, statesLength)
		Q = np.zeros(dim, dtype=int)
		seed = int(clock() * 100)
		np.random.seed(seed) # use current CPU time as seed

		#simulated execution
		start = np.random.randint(0, len(states))
		state = states[start]
		for i in range(0, numEpisodes):
			reward = R[state]
			# randomly select action a where R >= 0
			moves = []
			for action in reward.keys():
				if reward[action] >= 0:
					moves.append(action)
			resultingState = moves[0]
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
		#for

		return Q

	except Exception as err:
		print("ERROR in learn_Q()")
		print(sys.exc_info())


#variables that hold returned values from the defined functions.
tDiag = buildTransitionDiag(states)
R = buildRMatrix(tDiag)

#Define the initial and goal state with the corresponding index they hold in variable "states".
initial_state = 0
goal_state = 8

'''
Problem: Perform 500 episodes of training, and after every 2nd iteration,
use the Q Matrix to solve the problem, and save the number of steps taken.
At the end of training, use the saved step-count to plot a graph: training episode vs # of Moves.

NOTE: Do this for 4 alpha values. alpha = 0.1, 0.5, 0.8, 1.0
'''
trainSteps = [] #Variable to save iteration# and step-count.
runs = [i for i in range(10,5000,50)]#List contatining the runs from 10 -> 200, with a jump of 2.
#print runs

for i in runs:
	Q = learn_Q(R, alpha = 0.85, numEpisodes = i)
	stepsTaken = len(solveUsingQ(Q))
	trainSteps.append([i,stepsTaken])

#After Training, plotting diagram.
#NOTE: rename diagram accordingly or it will overwrite previous diagram.
x,y = zip(*trainSteps)
plt.plot(x,y,".-")
plt.xlabel("Training Episode")
plt.ylabel("# of Traversals.")
plt.savefig("output_0.85.png")

#Save the output for the best possible order, as generated by the code in the FOOTER.
path = solveUsingQ(Q)
print("\nThe best possible order to wear clothes is:\n")
tmp = ""
for i in path:
	tmp += states[i] + " -> "
print(tmp.rstrip(" ->"))

'''
FOOTER: Save program output here:
For alpha=0.1: Sadly, the agent never learns to put on clothes
For alpha=0.5: RRRR -> RRFR -> URFR -> UUFR -> UUFF
For alpha=0.8: RRRR -> RRFR -> URFR -> URFF -> UUFF
For alpha=1.0: RRRR -> URRR -> URFR -> UUFR -> UUFF

'''
