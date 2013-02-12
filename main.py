# main.py
# -------
# Lexi Ross & Ye Zhao

from dtree import *
import sys
import matplotlib.pyplot as plt

class Globals:
    noisyFlag = False
    pruneFlag = False
    boostingFlag = False
    valSetSize = 0
    dataset = None
    autoPruneFlag = False


##Classify
#---------

def classify(decisionTree, example):
  if Globals.boostingFlag:
    return classify_weighted_set(decisionTree, example)
  else:
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
    if '-a' in args_map:
      Globals.autoPruneFlag = True

    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

def accuracy(classified, instances):
  point = 0.
  for i in range(0, len(classified)):
    if classified[i] == instances[i]:
      point += 1.
  return point / len(instances)

def K_fold_cross_validate(dataset, K, n_vset=None):
  # dataset = entire dataset
  # K = the number of fold to use for the cross validation
  # n_vset = size of validation set to use

  N = len(dataset.examples) / 2 # length of dataset
  L = int(N / K) # length of 1 fold

  test_accuracies, train_accuracies = [], []

  all_examples = dataset.examples[:]
  for i in range(K):
    training_examples = all_examples[i*L:(K+i-1)*L]
    n_tset = len(training_examples)
    if Globals.pruneFlag:
      validate = training_examples[:n_vset]
      train = training_examples[n_vset:]
      dataset.examples = train
      dt = prune(learn(dataset), train, validate)
    elif dataset.use_boosting:
      print dataset.use_boosting
      dataset.examples = training_examples
      print len(dataset.examples)
      dt = ada_boost(dataset, dataset.num_rounds, dataset.max_depth)
      print dt
    else:
      dataset.examples = training_examples
      dt = learn(dataset)
    # run prediction over the test set of examples
    examples = all_examples[i*L:N+i*L]
    predictions = [classify(dt, example) for example in examples]
    targets = [example.attrs[dataset.target] for example in examples]

    train_accuracies.append(accuracy(predictions[:n_tset], targets[:n_tset]))
    test_accuracies.append(accuracy(predictions[n_tset:], targets[n_tset:]))
  return mean(train_accuracies), mean(test_accuracies)


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

def prune(dt, training_examples, validation_examples):
  # dt = a decision tree instancce
  # training_examples = A list of training examples
  # validation_examples = A list of validation examples

  if dt.nodetype == DecisionTree.LEAF or len(training_examples) * len(validation_examples) == 0:
    return dt

  branches = dt.branches

  # An empty dictionary for mapping branches to reduced data sets
  data = {}

  # Pune the tree recursively
  for branch in branches:
    data[branch] = ([example for example in training_examples if example.attrs[dt.attr] == branch],
                    [example for example in validation_examples if example.attrs[dt.attr] == branch])
    branches[branch] = prune(branches[branch], *data[branch])

  # Replace the subtree with majority class
  for branch in branches:
    instances = [example.attrs[-1] for example in data[branch][1]]
    if len(instances) == 0 or len(data[branch][0]) == 0:
      continue
    prediction_accuracy = accuracy([dt.predict(example) for example in data[branch][1]], instances)
    majority = mode([example.attrs[-1] for example in data[branch][0]])
    majority_accuracy = accuracy([majority for i in range(len(instances))], instances)
    if majority_accuracy > prediction_accuracy:
      branches[branch] = DecisionTree(DecisionTree.LEAF, classification=majority)

  return dt

def main():
    arguments = validateInput(sys.argv)
    #print arguments
    Globals.noisyFlag, Globals.pruneFlag, Globals.valSetSize, maxDepth, boostRounds = arguments
    print Globals.noisyFlag, Globals.pruneFlag, Globals.valSetSize, maxDepth, boostRounds

    # Read in the data file

    if Globals.noisyFlag:
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
      Globals.boostingFlag = True
    
    if Globals.autoPruneFlag:
      train_accuracies, test_accuracies = [], []
      xs = range(1,81)
      for i in xs:
        print i
        train, test = K_fold_cross_validate(DataSet(data), 10, i)
        train_accuracies.append(train)
        test_accuracies.append(test)
      plt.plot(xs, train_accuracies, '-r', xs, test_accuracies, '-b')
      plt.xlabel('Validation set size')
      plt.ylabel('10 fold cross-validation accuracy')
      plt.legend(['Train accuracy', 'Test accuracy'])
      plt.title('Plot of test and training performance for validation set pruning of size [1,80]')
      plt.show()
    else:
      train, test = K_fold_cross_validate(dataset, 10, Globals.valSetSize)
      print "Train accuracy: %f" % train
      print "Test accuracy: %f" % test


    
main()


