import numpy as np

# everything here is for when a quarter note is 4 units long (so the fundamenmtal unit is a 16th)
# the longest measure is 16 units long
# tp stands for transition probabilities

tp = dict()
tp[16] = {None: 15,
          (1, 15): 5,
          (2, 14): 2.5,
          (3, 13): 2.5,
          (4, 12): 50,
          (5, 11): 2.5,
          (6, 10): 5,
          (7, 9): 2.5,
          (8, 8): 100}

tp[15] = {(1, 14): 1,
          (2, 13): 1,
          (3, 12): 1,
          (4, 11): 2,
          (5, 10): 1,
          (6, 9): 5,
          (7, 8): 1}

tp[14] = {(1, 13): 1,
          (2, 12): 20,
          (3, 11): 1,
          (4, 10): 2,
          (5, 9): 1,
          (6, 8): 20,
          (7, 7): 1}

tp[13] = {(1, 12): 2,
          (2, 11): 1,
          (3, 10): 2,
          (4, 9): 1,
          (5, 8): 1,
          (6, 7): 1}

tp[12] = {None: 35,
          (1, 11): 1,
          (2, 10): 1,
          (3, 9): 20,
          (4, 8): 100,
          (5, 7): 1,
          (6, 6): 100}

tp[11] = {(1, 10): 1,
          (2, 9): 1,
          (3, 8): 10,
          (4, 7): 2,
          (5, 6): 5}

tp[10] = {(1, 9): 1,
          (2, 8): 50,
          (3, 7): 5,
          (4, 6): 10,
          (5, 5): 1}

tp[9] = {(1, 8): 1,
         (2, 7): 1,
         (3, 6): 5,
         (4, 5): 1}

tp[8] = {None: 50,
         (1, 7): 1,
         (2, 6): 10,
         (3, 5): 1,
         (4, 4): 50}

tp[7] = {(1, 6): 1,
         (2, 5): 1,
         (3, 4): 10}

tp[6] = {None: 25,
         (1, 5): 1,
         (2, 4): 100,
         (3, 3): 10}

tp[5] = {(1, 4): 10,
         (2, 3): 2}

tp[4] = {None: 200,
         (1, 3): 5,
         (2, 2): 75}

tp[3] = {None: 50,
         (1, 2): 10}

tp[2] = {None: 100,
         (1, 1): 10}

tp[1] = {None: 1}

new = dict()
for div in tp:
    new[div] = dict()
    for subdiv in tp[div]:
        if subdiv and subdiv[0] != subdiv[1]:
            new[div][(subdiv[1], subdiv[0])] = tp[div][subdiv]

for div in tp:
    tp[div] = {**tp[div], **new[div]}

for div in tp:
    total = np.sum([tp[div][subdiv] for subdiv in tp[div]])
    for subdiv in tp[div]:
        tp[div][subdiv] /= total


def produce_subdivision(total):
    # items = list(tp[total].items())
    # probs = [prob for _, prob in items]
    # n = np.random.choice(len(probs), p=probs)
    # split = items[n][0]
    # if not split:
    #     return [total]
    # else:
    #     return produce_subdivision(split[0]) + produce_subdivision(split[1])
    return [4]*int(total/4)

