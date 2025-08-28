from game_env import GameEnv
from game_state import GameState

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 1 "Cheese Hunter" Support Code, 2025
"""


class Solver:

    def __init__(self, game_env):
        self.game_env = game_env
        self.actions = list(self.game_env.ACTIONS)
        self.action_costs = dict(self.game_env.ACTION_COST)

        self._h_cache = None


        #
        #
        # TODO: Define any class instance variables you require here (avoid performing any computationally expensive
        #  heuristic preprocessing operations here - use the preprocess_heuristic method below for this purpose).
        #
        #

    @staticmethod
    def get_testcases():
        """
        Select which testcases you wish the autograder to test you on.
        The autograder will not run any excluded testcases.
        e.g. [1, 4, 6] will only run testcases 1, 4, and 6, excluding, 2, 3, and 5.
        :return: a list containing which testcase numbers to run (testcases in 1-6).
        """
        return [1, 2, 3, 4, 5, 6]

    @staticmethod
    def get_search():
        """
        Select which search you wish the autograder to run.
        The autograder will only run the specified search methods.
        e.g. "both" will run both UCS and A*, but "a_star" will only run A* and exclude UCS.
        :return: a string containing which search methods to run ("ucs" to only run UCS, "a_star" to only run A*,
        and "both" to run both).
        """
        return "a_star"
    
    

    def _reconstruct_path(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    # === Uniform Cost Search ==========================================================================================
    def search_ucs(self):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of GameEnv.ACTIONS)
        """

        """
        Return: list of action strings (each in GameEnv.ACTIONS) from start to goal, minimizing total cost.
        """
        start_state: GameState = self.game_env.get_init_state()

        if self.game_env.is_solved(start_state):
            return []

        class Node:
            __slots__ = ("state", "g", "parent", "action")
            def __init__(self, state, g, parent, action):
                self.state = state
                self.g = g
                self.parent = parent
                self.action = action

        import heapq, itertools
        counter = itertools.count()
        frontier = []
        start_node = Node(start_state, 0.0, None, None)
        heapq.heappush(frontier, (0.0, next(counter), start_node))

        best_cost = {start_state: 0.0}

        while frontier:
            g_curr, _, node = heapq.heappop(frontier)
            if g_curr > best_cost.get(node.state, float("inf")) + 1e-12:
                continue
            if self.game_env.is_solved(node.state):
                return self._reconstruct_path(node)

            for action in self.actions:
                success, next_state = self.game_env.perform_action(node.state, action)
                if not success:
                    continue
                step_cost = self.action_costs.get(action, 1.0)
                g_next = node.g + step_cost
                prev_best = best_cost.get(next_state)
                if (prev_best is None) or (g_next + 1e-12 < prev_best):
                    best_cost[next_state] = g_next
                    child = Node(next_state, g_next, node, action)
                    heapq.heappush(frontier, (g_next, next(counter), child))
        return []
    # ===============================================

    






    # === A* Search ====================================================================================================
    def preprocess_heuristic(self):
        """
        Perform pre-processing (e.g. pre-computing repeatedly used values) necessary for your heuristic,
        """

        if self._h_cache is None:
            self._h_cache = {}




        

    def compute_heuristic(self, state: GameState) -> float:
        """
        Compute a heuristic value h(n) for the given state.
        :param state: given state (GameState object)
        :return a real number h(n)
        """

        return 0.0

       



    def search_a_star(self):
        """
        Find a path which solves the environment using A* Search.
        :return: path (list of actions, where each action is an element of GameEnv.ACTIONS)
        """

        self.preprocess_heuristic()

        start_state: GameState = self.game_env.get_init_state()
        if self.game_env.is_solved(start_state):
            return []

        class Node:
            __slots__ = ("state", "g", "parent", "action")
            def __init__(self, state, g, parent, action):
                self.state = state
                self.g = g
                self.parent = parent
                self.action = action

        import heapq, itertools
        counter = itertools.count()

        # f = g + h
        frontier = []
        g0 = 0.0
        h0 = self.compute_heuristic(start_state)
        f0 = g0 + h0
        start_node = Node(start_state, g0, None, None)
        heapq.heappush(frontier, (f0, next(counter), start_node))

        best_cost = {start_state: g0}

        while frontier:
            f_curr, _, node = heapq.heappop(frontier)

            
            if node.g > best_cost.get(node.state, float("inf")) + 1e-12:
                continue

            if self.game_env.is_solved(node.state):
                return self._reconstruct_path(node)

            for action in self.actions:
                success, next_state = self.game_env.perform_action(node.state, action)
                if not success:
                    continue

                step_cost = self.action_costs.get(action, 1.0)
                g_next = node.g + step_cost

                prev_best = best_cost.get(next_state)
                if (prev_best is None) or (g_next + 1e-12 < prev_best):
                    best_cost[next_state] = g_next
                    h_next = self.compute_heuristic(next_state)
                    f_next = g_next + h_next
                    child = Node(next_state, g_next, node, action)
                    heapq.heappush(frontier, (f_next, next(counter), child))

        
        return []

        