'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import math

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)
    dr = (np.diff(points_X) ** 2 + np.diff(points_Y) ** 2) ** .5  # segment lengths
    r = np.zeros_like(points_X)
    r[1:] = np.cumsum(dr)  # integrate path
    r_int = np.linspace(0, r.max(), 100)  # regular spaced path
    sample_points_X = np.interp(r_int, r, points_X)  # interpolate
    sample_points_Y = np.interp(r_int, r, points_Y)
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 6.0
    # TODO: Do pruning (12 points)
    for i in range(10000):
        dist_start_to_start = math.hypot(gesture_points_X[0] - template_sample_points_X[i][0], gesture_points_Y[0] - template_sample_points_Y[i][0])
        dist_end_to_end = math.hypot(gesture_points_X[99] - template_sample_points_X[i][99], gesture_points_Y[99] - template_sample_points_Y[i][99])
        if dist_start_to_start <= threshold and dist_end_to_end <= threshold:
            valid_words.append(words[i])
            valid_template_sample_points_X.append(template_points_X[i])
            valid_template_sample_points_Y.append(template_points_Y[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 1

    # TODO: Calculate shape scores (12 points)
    #shifting coordinates in position with centroid
    gesture_centroid = (sum(gesture_sample_points_X) / 100, sum(gesture_sample_points_Y) / 100)
    gesture_sample_points_X = [ x - gesture_centroid[0] for x in gesture_sample_points_X]
    gesture_sample_points_Y = [y - gesture_centroid[1] for y in gesture_sample_points_Y]

    for i in range(len(valid_template_sample_points_X)):
        template_centroid = (sum(valid_template_sample_points_X[i]) / 100, sum(valid_template_sample_points_Y[i]) / 100)
        valid_template_sample_points_X[i] = [x - template_centroid[0] for x in valid_template_sample_points_X[i]]
        valid_template_sample_points_Y[i] = [y - template_centroid[1] for y in valid_template_sample_points_Y[i]]

    #scaling coordinates
    gesture_scaling_factor = L/max(max(gesture_sample_points_X), max(gesture_sample_points_Y))
    gesture_sample_points_X = [x * gesture_scaling_factor for x in gesture_sample_points_X]
    gesture_sample_points_Y = [y * gesture_scaling_factor for y in gesture_sample_points_Y]

    for i in range(len(valid_template_sample_points_X)):
        gesture_scaling_factor = L / max(max(valid_template_sample_points_X[i]), max(valid_template_sample_points_Y[i]))
        gesture_sample_points_X = [x * gesture_scaling_factor for x in valid_template_sample_points_X[i]]
        gesture_sample_points_Y = [y * gesture_scaling_factor for y in valid_template_sample_points_Y[i]]

    for i in range(len(valid_template_sample_points_X)):
        euclidean_distance_sum = 0.0
        for j in range(100):
            euclidean_distance_sum += math.hypot(gesture_sample_points_X[j] - valid_template_sample_points_X[i][j], gesture_sample_points_Y[j] - valid_template_sample_points_Y[i][j])
        shape_scores.append(sum/100.0)

    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores (12 points)
    '''
    d_p_q_list = []
    min_d_p_q_list = []
    templatewise_min_d_p_q_list = []

    for i in range(len(valid_template_sample_points_X)):
        min_d_p_q_list = []
        for j in range(100):
            x1 = gesture_sample_points_X[j]
            y1 = gesture_sample_points_Y[j]
            d_p_q_list = []
            for k in range(100):
                x2 = valid_template_sample_points_X[j][k]
                y2 = valid_template_sample_points_Y[j][k]
                d_p_q_list.append(math.hypot(x2-x1, y2-y1))
            #min_d_p_q_list.append(min(d_p_q_list))
        templatewise_min_d_p_q_list.append(min_d_p_q_list)

    D_p_q = []

    for i in range(len(valid_template_sample_points_X)):
        dist_sum = 0.0
        for j  in range(100):
            dist_sum += max(templatewise_min_d_p_q_list[i][j] - radius, 0)
        D_p_q.append(dist_sum)
    '''
    first_half = []
    point = 0.0001
    first_half.append(point)
    for i in range(49):
        point = point + 0.00041
        first_half.append(point)
    alpha = first_half[::-1] + first_half

    for i in range(len(valid_template_sample_points_X)):
        sum_xl = 0.0
        for j in range(100):
            sum_xl += alpha[j] * delta(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[j], valid_template_sample_points_Y[j], radius, j)
        location_scores.append(sum_xl)

    return location_scores

def delta(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y, radius, i):
    euclid_dist = 0.0
    if D_p_q(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y, radius) == 0 and D_p_q(valid_template_sample_points_X, valid_template_sample_points_Y, gesture_sample_points_X, gesture_sample_points_Y, radius) == 0:
        return 0
    else:
        euclid_dist = math.hypot(gesture_sample_points_X[i] - valid_template_sample_points_X[i], gesture_sample_points_Y[i] - valid_template_sample_points_Y[i])
    return euclid_dist

def D_p_q(gesture_list_x, gesture_list_y, template_list_x, template_list_y, radius):
    sum_dist = 0.0
    for i in range(100):
        sum_dist += max(min_dist_p_q(gesture_list_x[i], gesture_list_y[i], template_list_x, template_list_y)-radius, 0)
    return sum_dist

def min_dist_p_q(gesture_coord, template_list_x, template_list_y):
    min_dist = 0.0
    for i in range(100):
        euclidean_distance = math.hypot(template_list_x[i] - gesture_coord[0], template_list_y[i] - gesture_coord[1])
        if min_dist > euclidean_distance:
            min_dist = euclidean_distance
    return min_dist

def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    word_score_map = {}
    for i in len(valid_words):
        word_score_map[integration_scores[i]] = valid_words[i]
    sortkey=sorted(word_score_map.keys())
    sortkey=sortkey[:n+1]
    words = []
    for j in sortkey:
        words.append(word_score_map[j])

    return words[0] + ', ' + words[1] + ', ' + words[2]

def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'