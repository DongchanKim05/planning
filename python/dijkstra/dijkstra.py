import numpy as np
import math
import matplotlib.pyplot as plt
import random

show_animation  = True

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.f = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True

def collision_check(omap, node):
    nx = node[0]
    ny = node[1]
    ox = omap[0]
    oy = omap[1]

    col = False

    for i in range(len(ox)):
        if nx == ox[i] and ny == oy[i]:
            col = True
            break

    return col

def get_action():
    # dx, dy, cost
    action_set = [[0,-1,1], [0,1,1], [-1,0,1], [1,0,1],
    [1,-1,np.sqrt(2)], [1,1,np.sqrt(2)], [-1,1,np.sqrt(2)], [-1,-1,np.sqrt(2)]]
    return action_set

# a star algorithm
def a_star(start, goal, omap):

    # initialize
    start_node = Node(None, start)
    goal_node = Node(None, goal)

    Open = []
    Closed = []

    Open.append(start_node)

    while Open is not None:
        # find current node with lowest f in 'Open list'
        cur_node = Open[0]
        cur_ind = 0
        for ind, node in enumerate(Open):
            if node.f < cur_node.f:
                cur_node = node
                cur_ind = ind

        # If goal, get optimal path
        if cur_node.position == goal_node.position:
            opt_path = []

            node = cur_node

            while node is not None:
                opt_path.append(node.position)
                node = node.parent
            print("opt path : ", opt_path[::-1])
            print("opt_path_shape : ", np.shape(opt_path))
            return opt_path[::-1]

        # if not goal, delete from 'Open list' and add to 'Closed list'
        Open.pop(cur_ind)
        Closed.append(cur_node)

        # show graph
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(Closed) % 100 == 0:
                plt.pause(0.001)

        # search child nodes
        action_set = get_action()
        for action in action_set:
            # position of child candidate
            child_cand = (cur_node.position[0] + action[0], cur_node.position[1] + action[1])

            if collision_check(omap, child_cand):
                continue

            # create new node
            child = Node(parent=cur_node, position=child_cand)

            # If in 'Closed list', continue
            if child in Closed:
                continue

            child.f = cur_node.f + action[2]

            # if node is not in 'Open list', add it
            if child not in Open:
                Open.append(child)

            if child in Open:
                for node in Open:
                    if node == child and node.f > child.f:
                        node.f = child.f

def main():

    # INITIALIZE - 1)start, 2)goal, 3)map
    start = (10, 5)
    goal = (35, 5)

    ox, oy = [], []

    for i in range(61):
        ox.append(i)
        oy.append(0)
    for i in range(61):
        ox.append(0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60)
    for i in range(60):
        ox.append(60)
        oy.append(i)
    for i in range(51):
        ox.append(30)
        oy.append(i)
    for i in range(21):
        ox.append(30+i)
        oy.append(10)

    omap = [ox, oy]

    if show_animation == True:
        plt.figure(figsize=(10,10))
        plt.plot(start[0], start[1], 'bs',  markersize=7)
        plt.text(start[0], start[1]+0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs',  markersize=7)
        plt.text(goal[0], goal[1]+0.5, 'goal', fontsize=12)
        plt.plot(ox, oy, '.k',  markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Dijkstra algorithm", fontsize=20)

    opt_path = a_star(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)

    if show_animation == True:
        plt.plot(opt_path[:,0], opt_path[:,1], "c.-")
        plt.show()


if __name__ == "__main__":
    main()
