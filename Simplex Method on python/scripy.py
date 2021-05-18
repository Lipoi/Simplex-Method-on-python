import heapq
import numpy
import pandas as pd
def identity(numRows, numCols, val=1, rowStart=0):
   return [[(val if i == j else 0) for j in range(numCols)]
               for i in range(rowStart, numRows)]


def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[],
                equalities=[], eqThreshold=[], maximization=True):
   newVars = 0
   numRows = 0
   if gtThreshold != []:
      newVars += len(gtThreshold)
      numRows += len(gtThreshold)
   if ltThreshold != []:
      newVars += len(ltThreshold)
      numRows += len(ltThreshold)
   if eqThreshold != []:
      numRows += len(eqThreshold)

   if not maximization:
      cost = [-x for x in cost]

   if newVars == 0:
      return cost, equalities, eqThreshold

   newCost = list(cost) + [0] * newVars

   constraints = []
   threshold = []

   oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1),
                     (equalities, eqThreshold, 0)]

   offset = 0
   for constraintList, oldThreshold, coefficient in oldConstraints:
      constraints += [c + r for c, r in zip(constraintList,
         identity(numRows, newVars, coefficient, offset))]

      threshold += oldThreshold
      offset += len(oldThreshold)

   return newCost, constraints, threshold


def dot(a,b):
   return sum(x*y for x,y in zip(a,b))

def column(A, j):
   return [row[j] for row in A]

def transpose(A):
   return [column(A, j) for j in range(len(A[0]))]

def isPivotCol(col):
   return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

def variableValueForPivotColumn(tableau, column):
   pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
   return tableau[pivotRow][-1]

def initialTableau(c, A, b):
   tableau = [row[:] + [x] for row, x in zip(A, b)]
   tableau.append([ci for ci in c] + [0])
   return tableau


def primalSolution(tableau):
   columns = transpose(tableau)
   indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
   return [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex]))
            for colIndex in indices]


def objectiveValue(tableau):
   return -(tableau[-1][-1])


def canImprove(tableau):
   lastRow = tableau[-1]
   return any(x < 0 for x in lastRow[:-1])


def moreThanOneMin(L):
   if len(L) <= 1:
      return False

   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
   return x == y


def findPivotIndex(tableau):
   column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x < 0]
   column = min(column_choices, key=lambda a: a[1])[0]

   if all(row[column] <= 0 for row in tableau):
      raise Exception('Linear program is unbounded.')

   quotients = [(i, r[-1] / r[column])
      for i,r in enumerate(tableau[:-1]) if r[column] > 0]

   if moreThanOneMin(quotients):
      raise Exception('Linear program is degenerate.')

   row = min(quotients, key=lambda x: x[1])[0]

   return row, column


def pivotAbout(tableau, pivot):
   i,j = pivot

   pivotDenom = tableau[i][j]
   tableau[i] = [x / pivotDenom for x in tableau[i]]

   for k,row in enumerate(tableau):
      if k != i:
         pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
         tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]


'''
   simplex: [float], [[float]], [float] -> [float], float
   Solve the given standard-form linear program:
      max <c,x>
      s.t. Ax = b
           x >= 0
   providing the optimal solution x* and the value of the objective function
'''
def simplex(c, A, b):
   tableau = initialTableau(c, A, b)
   print("Initial tableau:")
   for row in tableau:
      print(row)
   print()

   while canImprove(tableau):
      pivot = findPivotIndex(tableau)
      print("Next pivot index is=%d,%d \n" % pivot)
      pivotAbout(tableau, pivot)
      print("Tableau after pivot:")
      for row in tableau:
         print(row)
      print()

   return tableau, primalSolution(tableau), objectiveValue(tableau)



   '''
   print('Please input c in row, numbers are split by space:')
   c = [-3, -1, -3]
   #c = [int(n) for n in input().split()]
   print('Please input the number of row of A')

   A = [[2, 1, 1], [1, 2, 3], [2, 2, 1]]
   m = int(input())
   A = [[] for i in range(m)]
   print('Please input A')
   for i in range(m):
      line = input().split(' ')
      for j in range(len(line)):
         A[i].append(int(line[j]))


   print('Please input b in row, numbers are split by space:')
   b = [2, 5, 6]
   b = [int(n) for n in input().split()]
   '''


c = [5, 2, -4, 0, 0, 0]
A = [
      [6, 1, -2, -1, 0, 0],
      [1, 1, 1, 0, 1, 0],
      [6, 4, -2, 0, 0, -1]
   ]
b = [5, 4, 10]

s = len(A)
for i in range(s):
   l = [0]*s
   for j in range(s):
      if(i==j):
         l[i] = 1
   A[i] += l

l = [0]*s
c += l


t, s, v = simplex(c, A, b)
print('The pair of base and its value:',s)
print('Objective value:',v)