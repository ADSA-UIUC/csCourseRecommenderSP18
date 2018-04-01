'''
Created on Mar 1, 2018

@author: Yeda
'''
from flask_cors import CORS,cross_origin
from flask import Flask, render_template, request
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import networkx as nx
from operator import itemgetter
networkprereqs = nx.DiGraph()
networksimilarity = nx.Graph()
app = Flask(__name__)
CORS(app)

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
    print("wheeeeee")
    for i in data.keys():
        networksimilarity.add_node(i)
        if data[i][6] != "['N/A']":
            networkprereqs.add_node(i)
            for p in data[i][6][2:-2].split("', '"):
                if p.split(" ")[0]=='CS':
                    networkprereqs.add_edge(p,i)
    print("wewwwww")
    for k1 in data.keys():
        topick1 = data[k1][7][2:-2].split("', '")
        for k2 in list(data.keys())[list(data.keys()).index(k1)+1:]: 
            topick2 = data[k2][7][2:-2].split("', '")
            topic = list(set(topick1).intersection(topick2))
            if topic != [] and topic != ['N/A']:
                add_weight(k1, k2, .75)
    print("whoooooo")
    
    networktree = nx.gn_graph(networkprereqs.number_of_nodes())
    pos=nx.spring_layout(networktree)
    nx.draw_networkx(networktree,pos=pos,with_labels=False, node_size=100)
    plt.show()
    
def input():            
    a =int(input('enter number of classes'))
    classes = {}
    for i in range(a):
        c='CS ' + str(input("enter class number:"))
        gpa = str(input('enter letter grade:')).upper()
        classes[c]=gpa
    return classes

def recommendations(classes):
    p={}
    q=[]              
    for j in classes.keys():
        try:
            for key in networkprereqs.successors(j):
                l = len(networkprereqs.successors(j))
                if key not in classes.keys():
                    if key not in p:
                        p[key]=(l+2)/l
                    else:
                        p[key]+=(l+2)/l
        except:
            pass
        for key in networksimilarity.neighbors(j):
            if key not in classes.keys():
                if key not in p:
                    p[key]=.4
                else:
                    p[key]+=.4
    
    for key,value in p.items():
        q.append((key,round(value,1)))
    q=dict(q)
    recommendations=[]
    for key,value in q.items():
        recommendations.append((key,value))
    recommendations.sort(key=itemgetter(1), reverse = True)
    return (recommendations[0:10])

@app.route('/getRec', methods=['GET'])
def getRecommendationRequest():
    print("we are here")
    classList = request.args.get("responses")
    print(classList)
    breakList = str.split(classList, ",")
    grades = {}
    i = 0
    print("list next")
    print(breakList)
    print("list should be done")
    while i<len(breakList):
        classTaken = breakList[i]
        grade = breakList[i+1]
        print(classTaken)
        print(grade)
        i+=2
        grades["CS "+classTaken] = grade[0].upper()
      
    df = preprocess_data()
    #converts user class grades to gpa so that we can compare to the class averages
    grades2gpa = {"A+": 4.00, "A": 4.00, "A-": 3.67, "B+": 3.33, "B": 3.00,
                  "B-": 2.67, "C+": 2.33, "C": 2.00, "C-": 1.67, "D+": 1.33,
                  "D": 1.00, "D-": 0.67, "F": 0.00
                  }
    for key, grade in grades.items():
        if grade in grades2gpa:
            new_grade = grades2gpa[grade]
            grades[key] = grade, int(new_grade)
    #calculates the difference between the user's grade and the grade average for that class
    var = {}
    print(grades.items())
    
    for k, v in grades.items():
        try:
            x_bar = float(df.loc[(df['Class'] == k)]['Average GPA'])
        except:
            x_bar = 3.0
        var[k] = (int(v[1]) - x_bar)/10
    recs={}
    for i in recommendations(grades):
        recs[list(i)[0]]=list(i)[1]
    reweighted = []
    for key,value in recs.items():
        for key1, value1 in var.items():
            try:
                if key1 in networkprereqs.predecessors(key):
                    value += value1
                    reweighted.append(key)
                for j in networkprereqs.predecessors(key):
                    if key in networkprereqs.predecessors(j):
                        value += (value1)/2
                        reweighted.append(key)
            except:
                pass
        recs[key]=round(value,2)
    for key in recs.keys():
        if key not in reweighted:
            recs[key]+= -.5
    recommendation=[]
    for key,value in recs.items():
        recommendation.append((key,value))
    m= max(recommendation, key=itemgetter(1))[0]
    recommendation = dict(recommendation)
    m=recommendation[m]
    for i in recommendation.keys():
        recommendation[i]=recommendation[i]/m
    final = []
    discard = ['CS 100','CS 101', 'CS 125', 'CS 126', 'CS 173', 'CS 210']
    for key,value in recommendation.items():
        if key not in discard:
            final.append((key,round(value,2)))
    final.sort(key=itemgetter(1), reverse = True)
    responses = str(final)
    return responses


@app.route('/')
def index():
    return "wow m8"

if __name__ == "__main__":
    print("hahahaha")
    create_network()
    print(networkprereqs.number_of_nodes())
    port = int(os.environ.get('PORT',8008))
    app.run(host='0.0.0.0', debug=True)

