# main.py
# -------
# Lexi Ross & Ye Zhao

from dtree import *
import sys

class Globals:
    noisyFlag = False
    pruneFlag = False
    valSetSize = 0
    dataset = None


##Classify
#---------

def classify(decisionTree, example):
    return decisionTree.predict(example)

# 3. AdaBoost
# implements majority voting for a weighted set of predictions
def classify_weighted_set(decisionTrees, example):
    weights = [0, 0]
    print "\n\n"
    for dt in decisionTrees:
        classification = dt.predict(example)
        # the rest of this is bad but yolo
        weights[classification] += dt.weight
    print weights
    if weights[0] > weights[1]:
        return 0
    else:
        return 1


##Learn
#-------
def learn(dataset):
    learner = DecisionTreeLearner()
    learner.train(dataset)
    return learner.dt

def weak_learn(dataset, max_depth):
    learner = DecisionTreeLearner()
    learner.limited_train(dataset, max_depth)
    return learner.dt

##AdaBoost
#---------

# AdaBoost wrapper. Uses the weak_learn function, which
# limits tree depth to max_depth.
def ada_boost(dataset, rounds, max_depth):
    hypotheses = []
    default_weight = 1./len(dataset.examples)
    for e in dataset.examples:
       e.weight = default_weight

    for i in range(0, rounds):
        dt = weak_learn(dataset, max_depth)
        error = 0.
        weight_sum = 0.
        # build error from incorrect examples
        for e in dataset.examples:
            if classify(dt, e) != e.attrs[-1]:
                error += e.weight

        # calculate hypothesis weight
        if error == 0:
            dt.weight = sys.maxint
            hypotheses.append(dt)
            print "perfect tree"
            return hypotheses
        else:
            dt.weight = 0.5 * math.log((1 - error) / error)

        # decrease weight of correct examples
        for e in dataset.examples:
            if classify(dt, e) == e.attrs[-1]:
                #textbook psuedocode: e.weight *= error / (1 - error)
                e.weight *= math.exp(-1. * dt.weight)
            else:
                e.weight *= math.exp(dt.weight)
            weight_sum += e.weight

        # normalize weights
        for e in dataset.examples:
            e.weight = 0 if weight_sum == 0 else e.weight/weight_sum
        hypotheses.append(dt)

    return hypotheses

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)
    valSetSize = 0
    noisyFlag = False
    pruneFlag = False
    boostRounds = -1
    maxDepth = -1
    if '-n' in args_map:
      noisyFlag = True
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    if '-b' in args_map:
      boostRounds = int(args_map['-b'])
    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

def score(classified, instances):
  point = 0
  for i in range(0, len(classified)):
    if classified[i] == instances[i]:
      point += 1
  return point

def cross_validate(data, dataset, K, N):
  performance = 0.
  n = N / K
  for i in range(K):
    training_dataset = DataSet(data[i*n:(K+i-1)*n],values=dataset.values)
    test_dataset = DataSet(data[(K+i-1)*n:N+i*n],values=dataset.values)
    if dataset.use_boosting:
      hypotheses = ada_boost(dataset, dataset.num_rounds, dataset.max_depth)
    else:
      dt = learn(training_dataset)
    score = 0.
    for example in test_dataset.examples:
      classified = classify_weighted_set(hypotheses, example) if dataset.use_boosting else classify(dt, example)
      if classified == example.attrs[-1]:
        score += 1.
    accuracy = score / len(test_dataset.examples)
    performance += accuracy / K
  print performance

def pruning(data, dataset, K, N, V):
  performance = 0.
  n = N / K
  for i in range(K):
    training_dataset = DataSet(data[i*n:(K+i-1)*n-V], values=dataset.values)
    validation_dataset = DataSet(data[(K+i-1)*n-V:(K+i-1)*n], values=dataset.values)
    test_dataset = DataSet(data[(K+i-1)*n:N+i*n], values=dataset.values)
    dt = learn(training_dataset)
    score = 0.
    for example in test_dataset.examples:
      classified = classify(dt, example)
      if classified == example.attrs[-1]:
        score += 1.
    accuracy = score / len(test_dataset.examples)
    performance += accuracy / K
  print performance

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file

    if noisyFlag:
        f = open("noisy.csv")
    else:
        f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)

    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]

    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    # ============================
    # 2a) 10-fold cross validation
    # ============================
    #cross_validate(data, dataset, 10, 100)
    #print learn(dataset).display

    # ====================
    # 2b) Pruning Function
    # ====================


    # ====================
    # 3a) AdaBoost, varying depth of weak learner
    # ====================

    if dataset.use_boosting:
        print "AdaBoost:"
    cross_validate(data, dataset, 10, 100)

    # Use command line args:
    # -b 10 -d 1
    # -b 10 -d 2
    # -b 30 -d 1
    # -b 30 -d 2


    
main()



