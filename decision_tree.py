import numpy as np
import pickle
import os
import utilities
import generate_data_matrix
import svm
import numpy as np
import cv2
# import aggregation
import csv
import min_max_scaler
import decision_tree
from sklearn.model_selection import train_test_split


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Leaf_Node:

    def __init__(self, rows):
        self.future_pred = list(class_counts(rows).keys())[0]

class D_Node:
    def __init__(self,
                 rule,
                 left_br,
                 right_br):
        self.rule = rule
        self.left_br = left_br
        self.right_br = right_br


class rule:
    def __init__(self, attr, value):
        self.attr = attr
        self.value = value

    def match(self, row):
        val = row[self.attr]
        if isinstance(self.value, int) or isinstance(self.value, float):
            return val >= self.value
        else:
            return val == self.value


class DecisionTreeClassifier():

    def __init__(self):
        self.root = None


    def split(self, rows):
        current_gain, uncertainty, counts, good_ques, best_gain = 0, 1, class_counts(rows), None, 0
        for count in counts:
            prob = counts[count] / float(len(rows))
            uncertainty -= prob ** 2

        for col in range(len(rows[0]) - 1):
            values = set([row[col] for row in rows])
            for val in values:
                ques = rule(col, val)

                left_rows, right_rows = [], []
                for row in rows:
                    if ques.match(row):
                        left_rows.append(row)
                    else:
                        right_rows.append(row)

                if len(left_rows) == 0 or len(right_rows) == 0:
                    continue

                gl_p = float(len(left_rows)) / (len(left_rows) + len(right_rows))
                ginileft, giniright = 1, 1
                countsleft, countsright = class_counts(left_rows), class_counts(right_rows)
                for count in countsleft:
                    prob = countsleft[count] / float(len(left_rows))
                    ginileft -= prob ** 2
                for count in countsright:
                    prob1 = countsright[count] / float(len(right_rows))
                    giniright -= prob1 ** 2
                gain = uncertainty - gl_p * ginileft - (1 - gl_p) * giniright

                if gain >= best_gain:
                    best_gain, good_ques = gain, ques

        return best_gain, good_ques


    def make_tree(self, rows):
        gain, rule = self.split(rows)
        if gain == 0:
            return Leaf_Node(rows)


        left_rows, right_rows = [], []


        for row in rows:
            if rule.match(row):
                left_rows.append(row)
            else:
                right_rows.append(row)

        left_br, right_br = self.make_tree(left_rows), self.make_tree(right_rows)
        return D_Node(rule, left_br, right_br)


    def predict(self, row, node):
        if node == None:
            node = self.root
        if isinstance(node, Leaf_Node):
            return node.future_pred

        if node.rule.match(row):
            return self.predict(row, node.left_br)
        else:
            return self.predict(row, node.right_br)

    def fit(self, data):
        self.root = self.make_tree(data)
        return self.root

    def transform(self, data):
        results = []
        for row in data:
            results.append(self.predict(row, self.root))
        return results

