# uiuc-tools

Course Recommender Program -- version 1.0 04/01/2018

Intro:

This is a course recommender tool for CS courses at UIUC (will expand to all courses in future versions). It takes in user inputted data about courses already taken and returns a list of courses to take next based on similarities between classes (prereqs, topics) and the estimated GPA for each class. The back end is written in python 3.6. The front end is an HTML webpage (currently localhosted).

The python code uses networkx to organize the data into two network objects with courses as nodes and weighted edges between them. The first network (networkprereqs) is a directional graph that connects a classes prereqs to itself. The second network (networksimilarities) is a non-directional graph that connects classes together based on similarities in topics. The recommendation algorithm generates a sorted list of recommended courses using the weights of the edges of the two networks along with the statistical analysis of the GPA data. Flask is used to direct this list to the front end webpage. 

Files:

Fall_2014.csv -- Used for the statistical analysis of the GPA distribution.                                                               
CSData.csv -- All the data collected on all CS courses at UIUC (except CS 498, CS 591, and CS 598).                                       
Network.py -- Back end code for the network generation and recommendation algorithm.                                                       
classGet.html -- Front end code for the webpage.
