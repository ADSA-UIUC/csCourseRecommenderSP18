'''
Created on Mar 1, 2018

@author: Yeda
'''
# "#" ---> comments
# "'''" ---> broken or unused code that will be used in the future
# "!NEEDS WORK!" ---> what needs to be done
'''
from flask_cors import CORS,cross_origin
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
'''
import pandas as pd
import csv
import networkx as nx
from operator import itemgetter
networkprereqs = nx.DiGraph()
networksimilarity = nx.Graph()
#Flask used for front end
'''
app = Flask(__name__)
CORS(app)
'''
def preprocess_data():
    # creates a pandas dataframe from grade distributions of classes from fall 2014 file
    class_data = []
    dist = {}

    with open("Fall_2014.csv", 'r') as f:
        f.readlines(1)  #skip header
        for row in f.readlines():
            row = row.strip("\n").split(",")
            class_data.append(row)
            
    for i in range(len(class_data)):
        if class_data[i][2] not in dist:
            init_gpa = float(class_data[i][-1].strip("\""))
            results = list(map(int, class_data[i][9:-1]))
            dist["CS " + str(class_data[i][2])] = results, init_gpa, 1
        else:
            results = list(map(int, class_data[i][9:-1]))
            temp = []
            accumulated_results, tot_gpa, tot_occurrences = dist[class_data[i][2]]

            for num in range(len(results)):
                temp.append(accumulated_results[num] + results[num])

            new_tot_gpa = (tot_gpa * tot_occurrences + float(class_data[i][-1].strip("\""))) / (tot_occurrences + 1)
            dist["CS " + str(class_data[i][2])] = temp, float("%.2f" % new_tot_gpa), (tot_occurrences + 1)

    dist_df = pd.DataFrame.from_dict(dist, orient='index')
    dist_df2 = pd.DataFrame(dist_df[0].values.tolist(), columns=['A+','A','A-','B+','B','B-','C+','C','C-','D+',
                                                                 'D','D-','E','F', ])
    dist_df2 = dist_df2.drop(['E'], axis=1)
    stdv = (dist_df2.std(axis=1))

    dist_df2["Class"] = dist.keys()
    #calculates average gpa and standard deviation in case we want z scores
    dist_df2['Average GPA'] = dist_df[1].values.tolist()
    dist_df2['stdv'] = stdv

    return dist_df2


def add_weight(x1, x2, w):
    #if there isn't an edge yet it creates an edge with the weight w
    #if there is already an edge it adds weight w to the existing edge
    if networksimilarity.number_of_edges(x1, x2)>=1:
        networksimilarity[x1][x2]['weight']+=w  
    if (networksimilarity.number_of_edges(x2, x1) == 0):
        networksimilarity.add_edge(x2,x1,weight=w)
            
def create_network():
    data=dict()
        
    with open('CSData.csv', 'r') as dataread:
        reader= csv.reader(dataread)
        for row in reader:
            data[row[0].split(":")[0]]=row
    #builds 2 network objects, 1 directed graph called networkprereqs, and 1 undirected graph called networksimilarity
    for i in data.keys():
        networksimilarity.add_node(i)
        if data[i][6] != "['N/A']": #only uses classes that have prereqs
            networkprereqs.add_node(i)
            for p in data[i][6][2:-2].split("', '"):
                if p.split(" ")[0]=='CS': #only uses prereqs that are CS
                    networkprereqs.add_edge(p,i) #networkprereqs is directional, pointing away from the prereq (ex.: CS 125 --> CS 225)
    #adds weights to the edges of networksimilarity based on common topics, weight is set to .75/topic
    for k1 in data.keys():
        topick1 = data[k1][7][2:-2].split("', '")
        for k2 in list(data.keys())[list(data.keys()).index(k1)+1:]: 
            #k2 represents all the classes above k1 to avoid double counting
            #ex.: if k1 = 'CS 125', k2 goes through the list ['CS 126', 'CS 173', 'CS 196', 'CS 199', 'CS 210', ..., 'CS 585']
            topick2 = data[k2][7][2:-2].split("', '")
            topic = list(set(topick1).intersection(topick2)) #creates a list of common topics between k1 and k2
            #!NEEDS WORK! : find a better algorithm for the weight to add (instead of the arbitrary .5)
            if topic != [] and topic != ['N/A'] and topic != ['Required']:
                weight = .5*len(topic)
                add_weight(k1, k2, weight)
            elif topic == ['Required']:
                weight = .2
                add_weight(k1, k2, weight)
    #!NEEDS WORK! : improve the way omega is calculated so that it actually makes sense
    #currently values of omega range from 1.65 for CS 101 to 63.51 for CS 100 which will greatly mess with the recommendations
    '''
    df = preprocess_data()
    for k in data.keys():
        try:
            x_bar = float(df.loc[(df['Class'] == k)]['Average GPA'])
            omega = float(df.loc[(df['Class'] == k)]['stdv'])
            print(k,x_bar,omega)
        except:
            pass
    '''
    #generates a plot of networkprereqs using matplotlib and networkX
    #!NEEDS WORK! : figure out how to actually manipulate the way the nodes are arranged to generate a prettier graph to put on display
    '''
    networktree = nx.gn_graph(networkprereqs.number_of_nodes())
    pos=nx.spring_layout(networktree)
    nx.draw_networkx(networktree,pos=pos,with_labels=False, node_size=100)
    plt.show()
    '''
def inputs():
    #returns a dictionary with classes as keys and grades as values           
    a =int(input('enter number of classes'))
    classes = {}
    for i in range(a):
        c=str(input("enter class number:"))
        gpa = str(input('enter letter grade:')).upper()
        classes[c]=gpa
    return classes
    #Ex. {'101': 'A', '125': 'A', '173': 'B', '225': 'A', '126': 'A', '233': 'A'} 

def recommendations(classes):
    #Takes in a dict with keys: class taken, values: gpa
    #Returns the top ten recommended classes as a list of tuples sorted based on descending weights
    p={} #p is a dict of the classes with any recommendation weight > 0, 
    print("\n"+"Recommendation:")
    for j in classes.keys():
        #Adds weight based on prereqs
        #!NEEDS WORK! : find a better algorithm for the weight to add
        try:
            l = len(list(networkprereqs.successors(j))) #so that a class like CS 225 which is a prereq to 17 different classes doesn't affect the recommendation weight 17 times more than say CS 411 which has 1 "successor" 
            for key in networkprereqs.successors(j):
                if key not in classes.keys(): #so that we don't recommend classes already taken (classes.keys() are the user inputed classes) 
                    #if "key" isn't recommended yet, it creates the key "key" within p and assigns it some weight
                    #if "key" is already recommended, it some weight to the value of key "key" within p
                    if key not in p:
                        p[key]=3*(l+1)/(4*(l+.5))
                    else:
                        p[key]+=3*(l+1)/(4*(l+.5))
                    print("    prereq", j, key, 3*(l+1)/(4*(l+.5)), p[key])
        except: #exception is for classes that don't aren't prereqs to anything so they won't be in networkprereqs 
            pass
        #Adds weight based on similarity, takes the weights of the edges from the networksimilarity graph
        for key in networksimilarity.neighbors(j):
            if key not in classes.keys():
                if key not in p:
                    p[key]=networksimilarity[key][j]['weight']
                else:
                    p[key]+=networksimilarity[key][j]['weight']
                print("    topic ", j, key, networksimilarity[key][j]['weight'], p[key])
    print("\n"+'All recommended classes:')
    print("    "+str(p))
    recommendations=[] #a list of tuples of (recommended class, recommendation weight)
    for key,value in p.items():
        recommendations.append((key,round(value,2)))
    recommendations.sort(key=itemgetter(1), reverse = True) #sorts list(recommendations) by the second term of each tuple aka the weights
    print("\n"+'Top 10 recommended classes:')
    print("    "+str(recommendations[0:10]))
    return (recommendations[0:10])
    #Ex. [('CS 126', 2.0), ('CS 210', 1.6), ('CS 233', 1.6), ('CS 241', 1.2), ('CS 357', 1.0), ('CS 361', 1.0), ('CS 374', 0.4), ('CS 421', 0.4)]

#@app.route('/getRec', methods=['GET'])
def getRecommendationRequest():
    
    #classList is a dictionary of the user inputed 'class':'grade' pairs
    #use request.args.get("responses") if displayed on the front end, use inputs() if just working with the back end
    '''
    classList = request.args.get("responses")
    '''
    classList1 = inputs()
    classList = {}
    for key in classList1.keys():
        classList['CS ' + key] = classList1[key]
    print("\n"+"classList:")
    print(classList)
    df = preprocess_data()
    #converts user class grades to gpa so that we can compare to the class averages
    grades2gpa = {"A+": 4.00, "A": 4.00, "A-": 3.67, "B+": 3.33, "B": 3.00,
                  "B-": 2.67, "C+": 2.33, "C": 2.00, "C-": 1.67, "D+": 1.33,
                  "D": 1.00, "D-": 0.67, "F": 0.00
                  }
    for key, grade in classList.items():
        if grade in grades2gpa:
            try:
                new_grade = grades2gpa[grade]
            except: #assumes B grade if grade provided can't be understood
                new_grade = 3.00
            classList[key] = int(new_grade)
    #calculates the difference between the user's grade and the grade average for that class
    var = {} #a dict with keys: 'classes taken', values: 'grade variance from the mean'  
    for k, v in classList.items():
        try:
            x_bar = float(df.loc[(df['Class'] == k)]['Average GPA'])
        except:
            x_bar = 3.0
        var[k] = round((v - x_bar)/10, 3)
    recs={} #turns the list of tuples into a dict with keys: 'recommended class', values: 'recommendation weight'
    for i in recommendations(classList):
        recs[list(i)[0]]=list(i)[1]
    #reweights the recommendation weight for each recommended class in the following way:
    #if a class already taken (key1) is a direct prereq of the recommended class, the recommended weight increases by the variance (value1) (or decreases if value1<0)
    #if a class already taken (key1) is a prereq of a prereq of the recommended class, the recommended weight increases by the variance/2 (value1/2) (or decreases if value1<0)
    #!NEEDS WORK! : find a better reweighting algorithm
    print("\n"+"Grade re-adjustment:")
    for key,value in recs.items():
        for key1, value1 in var.items():
            try:
                if key1 in networkprereqs.predecessors(key):
                    value += value1
                    print("    first ", key, key1, value1, value)
                for j in networkprereqs.predecessors(key):
                    if key1 in networkprereqs.predecessors(j):
                        value += (value1)/2
                        print("    second", key, key1, (value1)/2, value)
            except:
                pass
        recs[key]=round(value,2)
    m=max(list(recs.values()))
    #normalizes the recommendation weights by dividing everything by the maximum weight 'm' 
    for i in recs.keys():
        recs[i]=recs[i]/m
    final = []
    discard = ['CS 100','CS 101', 'CS 125', 'CS 126', 'CS 173', 'CS 210']
    #removes the courses we don't want to recommend
    #!NEEDS WORK! : have a built in system that removes all those courses by removing all the prereqs, prereqs of prereqs, etc of the user inputed courses
    for key,value in recs.items():
        if key not in discard:
            final.append((key,round(value,2)))
    final.sort(key=itemgetter(1), reverse = True)
    responses = str(final) #returns a string to be displayed on the front end
    return responses
create_network()
result = getRecommendationRequest()
print("\n"+"End result:")
print("    "+result)


#Front end stuff, currently broken
'''
@app.route('/')
def index():
    return "wow m8"

if __name__ == "__main__":
    print("hahahaha")
    create_network()
    print(networkprereqs.number_of_nodes())
    port = int(os.environ.get('PORT',8008))
    app.run(host='0.0.0.0', debug=True)
'''
#!NEEDS WORK! : Use pickle or JSON to store a copy of the two network so it doesn't have to generate each time
#!NEEDS WORK! : Test various scenarios and evaluate the accuracy of the algorithms
#Current Algorithms:
# A = class taken
# B = class recommended
# W = weight added to B
# Prereqs: If A is a prereq to B, W = 3*(l+1)/(4*(l+.5)) where l = the number of classes with A as a prereq
#          It ranges from l=1 --> W=1 to l=17 (CS 225) --> W=0.77
#          (Arbitrary algorithm, tested a bunch of scenarios and this algorithm seemed to work well)
# Topic: If A and B share topics, W = 0.5*l where l = the number of topics they share (line 100)
# Required Courses: If A and B are both core course, W = 0.2
# Grade: If A is a prereq of B, W = (a-b)/10
#        If A is a prereq of a prereq of B, W = (a-b)/20
#        where a = the grade input for A, and b = the mean grade of B

