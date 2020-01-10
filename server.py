from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
import math


app = Flask(__name__)

# Centroid of 26 keys
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
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    threshold = 15
    for i in range(10000):
        length_gesture_point = len(gesture_points_X)
        interval = math.floor(length_gesture_point / 5)
        dist_start_to_start = math.hypot(gesture_points_X[0] - template_sample_points_X[i][0],
                                         gesture_points_Y[0] - template_sample_points_Y[i][0])
        dist_intermediate_one = math.hypot(gesture_points_X[interval] - template_sample_points_X[i][20],
                                           gesture_points_Y[interval] - template_sample_points_Y[i][20])
        dist_intermediate_two = math.hypot(gesture_points_X[3*interval] - template_sample_points_X[i][80],
                                             gesture_points_Y[3*interval] - template_sample_points_Y[i][80])
        dist_end_to_end = math.hypot(gesture_points_X[len(gesture_points_X) - 1] - template_sample_points_X[i][99],
                                     gesture_points_Y[len(gesture_points_X) - 1] - template_sample_points_Y[i][99])
        if len(words[i]) <= 4:
            if dist_start_to_start <= threshold+5 and dist_end_to_end <= threshold+3:
                valid_words.append(words[i])
                valid_template_sample_points_X.append(template_sample_points_X[i])
                valid_template_sample_points_Y.append(template_sample_points_Y[i])
        else:
            if dist_start_to_start <= threshold+5 and dist_end_to_end <= threshold+12 and dist_intermediate_one <= threshold+280 and dist_intermediate_two <= threshold+280:
                valid_words.append(words[i])
                valid_template_sample_points_X.append(template_sample_points_X[i])
                valid_template_sample_points_Y.append(template_sample_points_Y[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    shape_scores = []
    L = 1

    # shifting coordinates in position with centroid
    gesture_centroid = (sum(gesture_sample_points_X) / 100, sum(gesture_sample_points_Y) / 100)
    gesture_sample_points_X = [x - gesture_centroid[0] for x in gesture_sample_points_X]
    gesture_sample_points_Y = [y - gesture_centroid[1] for y in gesture_sample_points_Y]

    for i in range(len(valid_template_sample_points_X)):
        template_centroid = (sum(valid_template_sample_points_X[i]) / 100, sum(valid_template_sample_points_Y[i]) / 100)
        valid_template_sample_points_X[i] = [x - template_centroid[0] for x in valid_template_sample_points_X[i]]
        valid_template_sample_points_Y[i] = [y - template_centroid[1] for y in valid_template_sample_points_Y[i]]

    # scaling coordinates
    gesture_scaling_factor = L / max(max(gesture_sample_points_X), max(gesture_sample_points_Y))
    gesture_sample_points_X = [x * gesture_scaling_factor for x in gesture_sample_points_X]
    gesture_sample_points_Y = [y * gesture_scaling_factor for y in gesture_sample_points_Y]

    for i in range(len(valid_template_sample_points_X)):
        gesture_scaling_factor = L / max(max(valid_template_sample_points_X[i]), max(valid_template_sample_points_Y[i]))
        gesture_sample_points_X = [x * gesture_scaling_factor for x in valid_template_sample_points_X[i]]
        gesture_sample_points_Y = [y * gesture_scaling_factor for y in valid_template_sample_points_Y[i]]

    for i in range(len(valid_template_sample_points_X)):
        euclidean_distance_sum = 0.0
        for j in range(100):
            euclidean_distance_sum += math.hypot(gesture_sample_points_X[j] - valid_template_sample_points_X[i][j],
                                                 gesture_sample_points_Y[j] - valid_template_sample_points_Y[i][j])
        shape_scores.append(euclidean_distance_sum / 100.0)

    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    location_scores = []
    radius = 20
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
            sum_xl += alpha[j] * delta(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[i], valid_template_sample_points_Y[i], radius, i)
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
        sum_dist += max(min_dist_p_q((gesture_list_x[i], gesture_list_y[i]), template_list_x, template_list_y)-radius, 0)
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
    shape_coef = 0.5
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    best_word = 'the'
    n = 10
    word_score_map = {}
    for i in range(len(valid_words)):
        if integration_scores[i] in word_score_map:
            word_score_map[integration_scores[i]].append(valid_words[i])
        else:
            wordlist = []
            wordlist.append(valid_words[i])
            word_score_map[integration_scores[i]] = wordlist
        print(valid_words[i] + ': ' + str(integration_scores[i]))
    sortkey=sorted(word_score_map.keys())
    sortkey=sortkey[:n]
    words = []
    result = ''

    for i in sortkey:
        wordlist = word_score_map[i]
        for j in range(len(wordlist)):
            words.append(wordlist[j])

    for i in range(len(words)):
        if i <= 5:
            result += words[i] + ', '
        else:
            break
    return result


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    print('Pruning gesture templates...')
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)
    print('Number of words after pruning: ' + str(len(valid_words)))
    for i in range(len(valid_words)):
        print(valid_words[i])

    print('Calculating shape scores...')
    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)
    print('Shape scores calculated')

    print('Calculating location scores...')
    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)
    print('Location scores calculated')

    print('Calculating integration scores...')
    integration_scores = get_integration_scores(shape_scores, location_scores)
    print('Integration scores calculated')

    print('Getting best words...')
    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()
    print('Best words selected')

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
