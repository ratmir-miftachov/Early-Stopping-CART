
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..utils import noise_level_estimator as noise_est
from collections import deque
from queue import Queue
import heapq
import sys
import random



class Node(object):

    id_counter = 0  # Class variable to assign unique IDs to each node

    def __init__(self):

        self.id = Node.id_counter
        Node.id_counter += 1
        self.split_threshold = None
        self.variable = None
        self.left_child = None
        self.right_child = None
        self.node_prediction = None
        self.is_terminal = False  # Indicator for terminal nodes
        self.level = 0  # Initialize level at creation
        self.indices = None  # This will store the indices of data points at the node
        self.impurity  = None  # Attribute to store the MSE


    def set_params(self, split_threshold: float, variable: int):

        self.split_threshold = split_threshold
        self.variable= variable

    def set_children(self, left, right, level):

        self.left_child = left
        self.right_child = right
        if self.left_child:
            self.left_child.level = level + 1
        if self.right_child:
            self.right_child.level = level + 1



class DecisionTreeRegressor:

    def __init__(self, design: np.array, response: np.array, max_iter: int = None, min_samples_split: int = 1, kappa : [float, int] = None):

        if isinstance(response, (pd.Series, pd.DataFrame)):
            response = response.to_numpy()

        if isinstance(design, (pd.DataFrame, pd.Series)):
            design = design.to_numpy()

        self.design = design

        self.response = response

        self.sample_size = self.design.shape[0]
        self.dimension = self.design.shape[1]

        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.kappa = kappa
        self.queue_snapshots = []  # List to store snapshots of the priority queue
        # Add a counter for leaf nodes
        self.snapshots = []  # List to store tree states at each iteration
        self.snapshot_count = 0  # Initialize snapshot counter to zero

    def take_snapshot(self):
        # Capture the current state of the tree
        current_snapshot = {}
        self._capture_node_state(self.tree, current_snapshot)
        self.snapshots.append(current_snapshot)
        self.snapshot_count += 1

    def _capture_node_state(self, node, snapshot):
        if node is None:
            return
        # Store the state of the current node
        # Only capture the node if it hasn't been split
        if node.split_threshold is None:
            snapshot[node.id] = {
                "node_prediction": node.node_prediction,
                "is_terminal": node.is_terminal,
                "level": node.level,
                "impurity": node.impurity,
                "indices": node.indices

            }
        # Recursively store the state of child nodes
        self._capture_node_state(node.left_child, snapshot)
        self._capture_node_state(node.right_child, snapshot)


    def _process_iteration(self, information):
        # TODO: This is where the music plays:

        # Get the number of observations in each node
        number_obs_temp = {node_id: len(node_info['indices']) for node_id, node_info in information.items()}
        number_obs_temp = list(number_obs_temp.values())

        # Get the mse in each node
        impurity_temp = {node_id: node_info['impurity'] for node_id, node_info in information.items()}
        impurity_all = list(impurity_temp.values())

        # Calculate weighted MSE refering to the iteration:
        weighted_impurity = sum(impurity * (obs/self.sample_size) for impurity, obs in zip(impurity_all, number_obs_temp))

        return weighted_impurity


    def _find_best_split_faster(self, node_indices):

        best_feature = None
        best_split = None
        max_impurity_gain = -np.inf

        response_parent = self.response[node_indices]
        design_parent = self.design[node_indices]

        # Compute parent impurity once
        parent_impurity = self._impurity(response_parent)
        sample_size_parent = len(response_parent)

        # Iterate through features
        for variable in range(self.dimension):
            feature_values = design_parent[:, variable]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_responses = response_parent[sorted_indices]

            # Precompute cumulative sums for fast split evaluations
            left_counts = np.arange(1, sample_size_parent + 1)
            right_counts = sample_size_parent - left_counts

            # Iterate only over unique values as potential split points
            thresholds = np.unique(sorted_values)
            for split_threshold in thresholds:
                # Find the index where split occurs
                split_idx = np.searchsorted(sorted_values, split_threshold, side="right") - 1
                sample_size_left = left_counts[split_idx]
                sample_size_right = right_counts[split_idx]

                # Check min samples split
                if sample_size_left < self.min_samples_split or sample_size_right < self.min_samples_split:
                    continue

                # Compute impurities for left and right
                impurity_left = self._impurity(sorted_responses[:split_idx + 1])
                impurity_right = self._impurity(sorted_responses[split_idx + 1:])

                # Calculate weighted impurity
                weighted_impurity = (
                        (sample_size_left / sample_size_parent) * impurity_left
                        + (sample_size_right / sample_size_parent) * impurity_right
                )
                impurity_gain = parent_impurity - weighted_impurity

                # Update the best split
                if impurity_gain > max_impurity_gain:
                    max_impurity_gain = impurity_gain
                    best_feature = variable
                    best_split = split_threshold

        #TODO: How can a None happen here?
        if best_feature is not None and best_split is not None:
            return best_feature, best_split, max_impurity_gain
        else:
            return None

    def _mark_remaining_nodes_as_terminal(self, priority_queue):

        # Add a small random number to the impurity gains to break ties
        impurity_gains = [t[0] for t in priority_queue]
        epsilon = sys.float_info.epsilon
        updated_impurity_gains = [element + epsilon*random.uniform(0, 0.1) for element in impurity_gains]
        updated_priority_queue = [(updated_impurity_gains[i],) + tup[1:] for i, tup in enumerate(priority_queue)]
        priority_queue = updated_priority_queue

        while priority_queue:
            _, _, _, node,  indices = heapq.heappop(priority_queue)
            node.node_prediction = np.mean(self.response[indices])
            node.is_terminal = True


        self.take_snapshot()


    def iterate(self, max_depth: int):

        self.max_depth = max_depth
        self.tree = Node()

        node = self.tree
        indices = np.arange(self.sample_size)
        priority_queue = []
        initial_impurity = self._impurity(self.response)

        heapq.heappush(priority_queue, (-initial_impurity, 0, random.uniform(0, 1), node, indices))
        node.indices = indices
        node.impurity = initial_impurity

        while priority_queue:
            # Take a snapshot before processing the next node
            self.queue_snapshots.append(priority_queue.copy())
            negative_gain, level, _ , parent_node, parent_indices = heapq.heappop(priority_queue)
            sample_size_node = parent_indices.shape[0]

            response_node = self.response[parent_indices]

            if level >= self.max_depth or self.snapshot_count>=self.max_iter or sample_size_node <= self.min_samples_split:
                parent_node.node_prediction = np.mean(response_node)
                parent_node.is_terminal = True
                continue

            split_params = self._find_best_split_faster(parent_indices)


            if split_params is not None:
                best_feature, best_split, best_gain = split_params

                feature_values = self.design[parent_indices, best_feature]
                left_mask = feature_values <= best_split
                right_mask = ~left_mask

                left_indices = parent_indices[left_mask]
                right_indices = parent_indices[right_mask]
                response_left = self.response[left_indices]
                response_right = self.response[right_indices]



                left_node, right_node = Node(), Node()
                left_node.indices = left_indices
                right_node.indices = right_indices

                left_node.impurity = self._impurity(response_left)
                right_node.impurity = self._impurity(response_right)

                parent_node.set_params(best_split, best_feature)
                parent_node.set_children(left_node, right_node, level)

                left_split_params = self._find_best_split_faster(left_indices)

                if left_split_params:
                    left_best_feature, left_best_split, left_gain = left_split_params
                    priority_key = -left_gain + sys.float_info.epsilon * random.uniform(0, 0.1)
                    heapq.heappush(priority_queue, (priority_key, level + 1, random.uniform(0, 1), left_node, left_indices))
                else: # No further splits possible. Make the node a terminal node and assign the leaf value.
                    left_node.node_prediction = np.mean(response_left)
                    left_node.is_terminal = True

                right_split_params = self._find_best_split_faster(right_indices)
                if right_split_params:
                    right_best_feature, right_best_split, right_gain = right_split_params
                    priority_key = -right_gain + sys.float_info.epsilon * random.uniform(0, 0.1)
                    heapq.heappush(priority_queue, (priority_key, level + 1, random.uniform(0, 1), right_node, right_indices))
                else: # No further splits possible. Make the node a terminal node and assign the leaf value.
                    right_node.node_prediction = np.mean(response_right)
                    right_node.is_terminal = True

            else:
                parent_node.node_prediction = np.mean(response_node)
                parent_node.is_terminal = True

            self.take_snapshot()
            processed_information_iter = self._process_iteration(self.snapshots[-1])

            if processed_information_iter < self.kappa:
                self.stopping_iteration = self.snapshot_count
                self._mark_remaining_nodes_as_terminal(priority_queue)

                break


    def _impurity(self, response_node):

        mean_response = np.mean(response_node)
        squared_diffs = np.square(response_node - mean_response)
        mse = np.sum(squared_diffs) / len(response_node)
        return mse



    def predict(self, Xin: pd.DataFrame | np.ndarray) -> np.array:

        if isinstance(Xin, pd.DataFrame):
            Xin = Xin.to_numpy()

        p = []
        for r in range(Xin.shape[0]):
            p.append(self.__traverse(self.tree, Xin[r, :]))
        # return predictions
        return (np.array(p).flatten())


    def __traverse(self, node: Node, Xrow: np.array) -> int | float:
        """
        Private recursive function to traverse the (trained) tree

        Inputs:
            node -> current node in the tree
            Xrow -> data sample being considered
        Output:
            leaf value corresponding to Xrow
        """
        if node.is_terminal:
            # return the leaf value if it's a terminal node
            return node.node_prediction
        else:
            # continue to traverse based on the split condition
            (s, f) = node.split_threshold, node.variable

            if Xrow[f] <= s:
                return self.__traverse(node.left_child, Xrow)
            else:
                return self.__traverse(node.right_child, Xrow)