'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''
from pprint import pprint

from flask import Flask, request
from flask import render_template
import time
import json
import math
import numpy as np
import copy
from scipy.interpolate import interp1d
import statistics

app = Flask(__name__)

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
    #ediff1d gives difference between consecutive elements of the array
    #we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    #basically when words like mm or ii have no path / little path, use centroid
    if(distance[-1]==0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        #get the proportion of line segments
        distance = distance / distance[-1]
        #scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        #generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        #use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()
    return sample_points_X, sample_points_Y

# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

#Code to check the sampling by plotting the points
# x = [165.22222232818604, 161.02536852095676, 157.21763510336797, 153.53123471762134, 150.0831291489417, 146.63502358026207, 142.8272862297719, 139.28832219969922, 136.35362332336436, 132.75983930504924, 128.99327942547478, 125.93261176700312, 122.70743716773039, 119.25933159905075, 115.70234856378922, 111.94425929177567, 108.76301622226089, 105.58224278031332, 102.38696285021173, 100.22222232818604, 99.22222232818604, 103.3869574113483, 107.84100397044958, 112.56696372364883, 117.23766726626901, 121.84027668712754, 126.32235659465053, 130.70810285471921, 135.06964973861434, 139.43119662250942, 143.6865970116664, 147.93892846613755, 152.29054017967755, 156.09833577984224, 160.16949464507346, 164.28777585554124, 168.34041552324564, 172.2422740359868, 176.2996488620542, 180.27479186583346, 184.40647576444408, 188.53834481681648, 192.50640026210672, 196.04830280516452, 200.65722221759796, 204.83866441405286, 209.1836905438649, 213.08884458034822, 217.2607014306179, 221.64136947498346, 225.67157872970077, 229.57385729317326, 233.0793547628179, 236.79923833017648, 241.08617671888194, 242.67840777361195, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 243.22222232818604, 242.39379142098875, 241.22222232818604, 239.09193012838145, 234.73038324448635, 230.19431749356147, 225.3179598338405, 220.51501926298087, 215.68602544497563, 210.83949288252788, 206.00970165054392, 201.17187025713756, 196.3197130329499, 191.45880430051054, 186.58244664078964, 181.70608898106866, 176.82973132134777, 171.95337366162687, 167.07701600190592, 162.200658342185, 157.324300682464, 152.44794302274312, 147.57158536302214, 142.69522770330124, 137.81887004358035, 132.9425123838594, 128.06615472413847, 123.18979706441752, 118.31343940469661, 113.4370817449757, 108.56072408525475, 104.09857998790693, 99.22222232818604]
# y = [115.72222137451172, 113.25765216969221, 110.21839868716334, 107.03123376394703, 103.5831281952674, 100.13502262658777, 97.12559730890229, 93.78832124602489, 89.91932286727919, 86.79745532823813, 84.2643355690892, 80.57741609392026, 77.20743621405607, 73.75933064537644, 70.46228449231332, 67.44425833810135, 63.80380916266142, 60.4422622787663, 57.72222137451172, 53.66088577668936, 49.198741679341516, 49.49871142995323, 51.245977702964424, 52.425643793150556, 53.82685485593662, 55.41567324262951, 57.33656463156793, 59.46516163777831, 61.64593507972587, 63.82670852167342, 66.20084618459994, 68.58057444348748, 70.77687565570494, 73.82311213583668, 76.49058476464417, 79.0992570594152, 81.81086627080641, 84.73558917971224, 87.44050573042382, 90.2616485277473, 92.81434809264074, 95.37659458067631, 98.21091989874077, 101.54830185149021, 102.98322130815887, 105.4920866260318, 107.70295548235116, 110.62218806363336, 113.14530883597084, 115.14136852130915, 117.72222137451172, 120.07385633949895, 123.00792002645953, 125.29923737650215, 126.72222137451172, 122.44129414738217, 117.61878474033982, 112.7424270806189, 107.866069420898, 102.98971176117705, 98.11335410145615, 93.2369964417352, 88.3606387820143, 83.48428112229332, 78.60792346257242, 73.7315658028515, 68.85520814313054, 63.97885048340963, 59.23692865291982, 54.802626386509296, 50.657075274609426, 48.47630183266187, 46.72222137451172, 46.72222137451172, 45.98078076147069, 45.32931060971056, 44.790806991660816, 44.12043270627713, 43.51718616740687, 43.031970444988104, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 42.72222137451172, 43.72222137451172, 43.72222137451172]
# plt.scatter(x,y)
# plt.show()
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
    threshold = 12
    # TODO: Do pruning (12 points)
    userinput_start_x = gesture_points_X[0]
    userinput_start_y = gesture_points_Y[0]
    userinput_end_x = gesture_points_X[-1]
    userinput_end_y = gesture_points_Y[-1]
    index = 0
    #print(len(template_sample_points_Y))
    for t1,t2 in zip(template_sample_points_X,template_sample_points_Y):
        template_start_x = t1[0]
        template_end_x = t1[-1]
        template_start_y = t2[0]
        template_end_y = t2[-1]
        #finding difference between start and end points of user input and template
        start_distance = math.floor(math.hypot(template_start_x-userinput_start_x,template_start_y-userinput_start_y))
        end_distance = math.floor(math.hypot(template_end_x - userinput_end_x, template_end_y - userinput_end_y))
        if(start_distance<=threshold and end_distance<=threshold):
            valid_template_sample_points_X.append(t1)
            valid_template_sample_points_Y.append(t2)
            valid_words.append(words[index])
        index +=1
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
    boundingbox_user_height = max(gesture_sample_points_Y) - min(gesture_sample_points_Y)
    boundingbox_user_width = max(gesture_sample_points_X) - min(gesture_sample_points_X)
    try:
        user_trace_scale = L / max(boundingbox_user_height,boundingbox_user_width)
    except ZeroDivisionError:
        user_trace_scale = 0
    #Normalize in scale
    for i in range(len(gesture_sample_points_X)):
        gesture_sample_points_X[i] = gesture_sample_points_X[i] * user_trace_scale
        gesture_sample_points_Y[i] = gesture_sample_points_Y[i] * user_trace_scale
    user_x_centroid = statistics.mean(gesture_sample_points_X)
    user_y_centroid = statistics.mean(gesture_sample_points_Y)

    #Normalize in location with centroid
    for i in range(len(gesture_sample_points_X)):
        gesture_sample_points_X[i] = gesture_sample_points_X[i] - user_x_centroid
        gesture_sample_points_Y[i] = gesture_sample_points_Y[i] - user_y_centroid

    for tx, ty in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        summation = 0
        boundingbox_template_height = max(ty) - min(ty)
        boundingbox_template_width = max(tx) - min(tx)
        try:
            template_scale = L / max(boundingbox_template_height, boundingbox_template_width)
        except ZeroDivisionError:
            template_scale = 0
        # Normalize template in scale
        for i in range(len(tx)):
            tx[i] = tx[i] * template_scale
            ty[i] = ty[i] * template_scale

        #Normalize template in location
        temp_centroid_x = statistics.mean(tx)
        temp_centroid_y = statistics.mean(ty)
        for i in range(len(tx)):
            tx[i] = tx[i] - temp_centroid_x
            ty[i] = ty[i] - temp_centroid_y

        for i in range(100):
            summation += (math.hypot(tx[i] - gesture_sample_points_X[i],ty[i] - gesture_sample_points_Y[i]))
        shape_scores.append(round((summation / 100),2))
    return shape_scores

#Compute d(pi,q) value
# P x,y are current points computed with
# q x,y are points enumerated from 1 to N
def d(Px,Py,qx,qy):
    return math.floor(math.hypot(qx - Px, qy - Py))

#compute D(u,t) or D(t,u) value
# cx,cy are current points computed with
# ex,ey are points enumerated from 1 to N
# Roles of u(x,y) and t(x,y) are interchanged based on parameter sent from getDValue
def D(cx,cy,ex,ey,radius):
    return max(d(cx,cy,ex,ey)-radius,0)

#Get D(u,t) and D(t,u)
# userx set of 100 x coordinates of user trace
# usery set of 100 y coords of user trace
# templatex set of 100 x coords of current word template trace
# templatey set of 100 y coords of current word template trace
# radius of tunnel
def getDValue(user_x,user_y,template_x,template_y,radius):
    Dut = D(user_x,user_y,template_x,template_y,radius)
    #Dtu = D(template_x, template_y,user_x, user_y,radius)
    return Dut

#Get Dell value in the xL formula(3) in paper
# userx set of 100 x coordinates of user trace
# usery set of 100 y coords of user trace
# templatex set of 100 x coords of current word template trace
# templatey set of 100 y coords of current word template trace
# radius of tunnel
def getDellValue(user_x,user_y,template_x,template_y,radius):
    Dvalue = getDValue(user_x,user_y,template_x,template_y,radius)
    return Dvalue

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
    #alpha scores generated by another code
    alpha_scores = [0.018649999999999976, 0.018296999999999976, 0.017943999999999977, 0.017590999999999978,
                    0.01723799999999998, 0.01688499999999998, 0.01653199999999998, 0.01617899999999998,
                    0.015825999999999982, 0.015472999999999983, 0.015119999999999984, 0.014766999999999985,
                    0.014413999999999986, 0.014060999999999987, 0.013707999999999988, 0.013354999999999988,
                    0.01300199999999999, 0.01264899999999999, 0.012295999999999991, 0.011942999999999992,
                    0.011589999999999993, 0.011236999999999994, 0.010883999999999994, 0.010530999999999995,
                    0.010177999999999996, 0.009824999999999997, 0.009471999999999998, 0.009118999999999999, 0.008766,
                    0.008413, 0.008060000000000001, 0.007707, 0.007354, 0.007001, 0.006648, 0.006295, 0.005942,
                    0.005589, 0.005236, 0.004883, 0.00453, 0.004177, 0.003824, 0.003471, 0.003118, 0.002765, 0.002412,
                    0.002059, 0.001706, 0.001353, 0.001353, 0.001706, 0.002059, 0.002412, 0.002765, 0.003118, 0.003471,
                    0.003824, 0.004177, 0.00453, 0.004883, 0.005236, 0.005589, 0.005942, 0.006295, 0.006648, 0.007001,
                    0.007354, 0.007707, 0.008060000000000001, 0.008413, 0.008766, 0.009118999999999999,
                    0.009471999999999998, 0.009824999999999997, 0.010177999999999996, 0.010530999999999995,
                    0.010883999999999994, 0.011236999999999994, 0.011589999999999993, 0.011942999999999992,
                    0.012295999999999991, 0.01264899999999999, 0.01300199999999999, 0.013354999999999988,
                    0.013707999999999988, 0.014060999999999987, 0.014413999999999986, 0.014766999999999985,
                    0.015119999999999984, 0.015472999999999983, 0.015825999999999982, 0.01617899999999998,
                    0.01653199999999998, 0.01688499999999998, 0.01723799999999998, 0.017590999999999978,
                    0.017943999999999977, 0.018296999999999976, 0.018649999999999976]
    # TODO: Calculate location scores (12 points)
    for tx, ty in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        summation = 0
        for i in range(100):
            dellValue = getDellValue(gesture_sample_points_X[i],gesture_sample_points_Y[i],tx[i],ty[i],radius)
            if (dellValue == 0):
                summation += 0
            else:
                summation += (alpha_scores[i] * (math.hypot(tx[i] - gesture_sample_points_X[i], ty[i] - gesture_sample_points_Y[i])))
        location_scores.append(round(summation,2))
    return location_scores

def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(round(shape_coef * shape_scores[i] + location_coef * location_scores[i],2))
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
    n = 4
    suggestion = ""
    # TODO: Get the best word (12 points)
    sortedIndex = np.argsort(np.array(integration_scores))
    best_word = valid_words[sortedIndex[0]]
    #Just returning alternate suggestions just to show prediction proximity
    if(len(sortedIndex)>=n):
        suggestion = valid_words[sortedIndex[1]] + " , " + valid_words[sortedIndex[2]] + " , " + valid_words[sortedIndex[3]]
    if not best_word:
        return "No such word in dictionary"
    return best_word, suggestion


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

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)
    temp_ux = copy.deepcopy(gesture_sample_points_X)
    temp_uy = copy.deepcopy(gesture_sample_points_Y)
    temp_tempx = copy.deepcopy(valid_template_sample_points_X)
    temp_tempy = copy.deepcopy(valid_template_sample_points_Y)
    #print()
    #print("----------------List of Valid Words-----------")
    #print(valid_words)
    #print("----------------------------------------------")
    #print()
    if not valid_words:
        end_time = time.time()
        return '{"best_word":"' + 'No such word in dictionary!! Please try again...' + '", "elapsed_time":"' + str(
            round((end_time - start_time) * 1000, 5)) + 'ms"}'

    shape_scores = get_shape_scores(temp_ux, temp_uy, temp_tempx, temp_tempy)
    # print()
    # print("----------------List of Shape Scores-----------")
    #print(shape_scores)
    # print("----------------------------------------------")
    # print()
    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)
    # print()
    # print("----------------List of Location Scores-----------")
    #print(location_scores)
    # print("----------------------------------------------")
    # print()
    integration_scores = get_integration_scores(shape_scores, location_scores)
    # print()
    # print("----------------List of Integration Scores-----------")
    # print(integration_scores)
    # print("----------------------------------------------")
    # print()
    best_word,suggestion = get_best_word(valid_words, integration_scores)
    end_time = time.time()
    return '{"best_word":"' + best_word + ' (Time elapsed: ' + '", "elapsed_time":"' + str(
        round((end_time - start_time) * 1000, 2)) + 'ms)   Other suggestions: ' + suggestion + '"}'

if __name__ == "__main__":
    app.run()