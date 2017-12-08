# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and Jake Barnwell (jb16)
from parse import *
from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')

################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    current_node = id_tree

    while not current_node.is_leaf():
        current_node = current_node.apply_classifier(point)
    return current_node.get_node_classification()

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    split_data = {}

    for datum in data:
        add_or_increment_dict(split_data, classifier.classify(datum), datum)
    return split_data

def add_or_increment_dict(dictionary,key,datum):
    """Given a dictionary, a key and a datum:
    if key is not in dictionary ==> maps key to list containing datum
    else ==> appends datum to key's existing data list."""
    if key in dictionary:
        dictionary[key].append(datum)
    else:
        dictionary[key] = [datum]
        
#### CALCULATING DISORDER

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""

    n_b = float(len(data))
    split_data = split_on_classifier(data,target_classifier)
  
    disorder = 0
    for key in split_data:
        n_bc = len(split_data[key])
        disorder += (-n_bc/n_b)* log2(n_bc/n_b)
    return disorder



def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    n_t = len(data)
    
    split_data = split_on_classifier(data,test_classifier)
    avg_disorder = 0
    for key in split_data:
        n_b = float(len(split_data[key]))
        avg_disorder += (n_b/n_t) * branch_disorder(split_data[key], target_classifier)

    return avg_disorder

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:
for classifier in tree_classifiers:
    print classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type"))


#### CONSTRUCTING AN ID TREE

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    #avg_disorder = [(possible_classifier,average_test_disorder(data,possible_classifier,target_classifier) for possible_classifier in possible_classifiers]
    min_disorder_classifier =  min(possible_classifiers, key = lambda possible_classifier:\
                                   average_test_disorder(data,possible_classifier,target_classifier))
    if len(split_on_classifier(data,min_disorder_classifier)) >1:
        return min_disorder_classifier
    raise NoGoodClassifiersError
    
    
## To find the best classifier from 2014 Q2, Part A, uncomment:
print find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type"))


def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node is None:
        id_tree_node  = IdentificationTreeNode(target_classifier)

    target_classified_data = split_on_classifier(data,target_classifier)

    if len(target_classified_data) == 1:
        #print target_classifier
        id_tree_node.set_node_classification(data[0][target_classifier.name])

    else:
        if len(possible_classifiers) >0:
            try:
                min_classifier = find_best_classifier(data, possible_classifiers, target_classifier)
                possible_classifiers.remove(min_classifier)
                classified_data = split_on_classifier(data,min_classifier)

                id_tree_node.set_classifier_and_expand(min_classifier,classified_data)
                
                for branch_name in id_tree_node.get_branches():
                    construct_greedy_id_tree(classified_data[branch_name],\
                                             possible_classifiers[:],\
                                             target_classifier,\
                                             id_tree_node.get_branches()[branch_name])   

            except NoGoodClassifiersError:
                pass
            
    return id_tree_node
    

                

#Optional: Construct and ID tree with real medical data
#tree_medical = construct_greedy_id_tree(heart_training_data, heart_classifiers,heart_target_classifier_binary)


test_patient = {\
    'Age': 20, #int
    'Sex': 'F', #M or F
    'Chest pain type': 'asymptomatic', #typical angina, atypical angina, non-anginal pain, or asymptomatic
    'Resting blood pressure': 100, #int
    'Cholesterol level': 120, #int
    'Is fasting blood sugar < 120 mg/dl': 'Yes', #Yes or No
    'Resting EKG type': 'normal', #normal, wave abnormality, or ventricular hypertrophy
    'Maximum heart rate': 150, #int
    'Does exercise cause chest pain?': 'No', #Yes or No
    'ST depression induced by exercise': 0, #int
    'Slope type': 'flat', #up, flat, or down
    '# of vessels colored': 0, #float or '?'
    'Thal type': 'normal', #normal, fixed defect, reversible defect, or unknown
}
# uncomment the line to see the tree
#tree_medical.print_with_data([test_patient])
    



## To construct an ID tree for 2014 Q2, Part A:

#print construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
#tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
#print id_tree_classify_point(tree_test_point, tree_tree)

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
#print construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification"))
#print construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class"))

#### MULTIPLE CHOICE

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"


#for datum in binary_data:
#    print(datum, id_tree_classify_point(datum,binary_tree_3))

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = "No"
ANSWER_9 = "No"


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### MULTIPLE CHOICE: DRAWING BOUNDARIES

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### WARM-UP: DISTANCE METRICS

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    dot_product = 0
    for i in range(len(u)):
        dot_product += u[i]*v[i]
    return dot_product
        
def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return math.sqrt(dot_product(v,v))


def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    return norm([u1-v1 for (u1,v1) in zip(point1,point2)])

    
def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    return sum([abs(u1-v1) for (u1,v1) in zip(point1,point2)])


def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    return len([1 for (u1,v1) in zip(point1,point2) if u1!=v1])

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    return 1 - (dot_product(point1.coords,point2.coords)/(norm(point1.coords)*norm(point2.coords)))

#### CLASSIFYING POINTS

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    distance_point_to_each_datum = [(datum,distance_metric(point,datum)) for datum in data]
    distance_point_to_each_datum.sort(key = lambda (datum,distance): datum.coords)
    distance_point_to_each_datum.sort(key = lambda (datum,distance): distance)
    return map(lambda (datum,distance): datum, distance_point_to_each_datum)[:k]

    
def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    k_neighbors = get_k_closest_points(point,data,k,distance_metric)
    k_neighbors_class = [datum.classification for datum in k_neighbors]
    unique_classes = set(k_neighbors_class)
    return max([(unique_class,k_neighbors_class.count(unique_class)) for unique_class in unique_classes],\
               key = lambda (clss, count): count)[0]



## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
#print knn_classify_point(knn_tree_test_point, knn_tree_data, 5, euclidean_distance)


#### CHOOSING k

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    good_classifications = 0
    for datum in data:
        train_data = data[:]
        train_data.remove(datum)
        test_data = datum
        test_class = knn_classify_point(test_data, train_data,k,distance_metric)
        if test_class == datum.classification:
            good_classifications +=1

    return float(good_classifications)/float(len(data))
        
def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    highest_params= (- INF,None,None)
    distance_metrics = [euclidean_distance,manhattan_distance,hamming_distance,cosine_distance]

    for distance_metric in distance_metrics:
        for k in range(1,len(data)):
            current_params = (cross_validate(data,k,distance_metric),k,distance_metric)
            if current_params[0] > highest_params[0]:
                highest_params = current_params
    return (highest_params[1],highest_params[2])
        

## To find the best k and distance metric for 2014 Q2, part B, uncomment:
print find_best_k_and_metric(knn_tree_data)


#### MORE MULTIPLE CHOICE

kNN_ANSWER_1 = "Overfitting"
kNN_ANSWER_2 = "Underfitting"
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3

#### SURVEY ###################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
