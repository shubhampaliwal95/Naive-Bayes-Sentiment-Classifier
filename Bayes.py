import numpy as np
import re
print('Reading dataset')
#method =input()
num_Pos=0
num_Neg=0
#0 is negative
#1 is positive
def prediction(Data,bag,positive_Words,negative_Words,unique_pos,unique_neg):
	mat=np.zeros((2,2))							#matrix of 2*2
	#print(unique_neg)
	#print(unique_pos)
	#print(positive_Words)
	#print(negative_Words)
	tot_Unique=len(bag.keys())
	#print(tot_Unique)
	for line in Data:
		n=0
		line=line.replace("\n","")
		words=re.split('[.?, ]',line)
		#print(words[1])
		if(words[1]=="neg"):
			actual=0
		else:
			actual=1
		#print(actual)
		positive_Prob=1.0
		negative_Prob=1.0
		global num_Pos
		global num_Neg
											#Calculating probability of document being in class positive or negative
		for word in words:
			n+=1
			if(n>4):
				try:
					#print(bag[word][0])
					negative_Prob*=((bag[word][0]+1)/(unique_neg))			
				except:
					negative_Prob*=1.0
				try:
					#print(bag[word][])
					positive_Prob*=((bag[word][1]+1)/(unique_pos))
				except:
					positive_Prob*=1.0
		#print(positive_Prob)
		#print(negative_Prob)
		
		#print(num_Pos)
		#print(num_Neg)
		positive_Prob*=num_Pos/(num_Pos+num_Neg)		#Muliplying by class probabilities
		negative_Prob*=num_Neg/(num_Pos+num_Neg)
		#Filling Confusion matrix
		if(positive_Prob>negative_Prob and actual==1):		#True positive		
			mat[1][1]+=1
		elif(positive_Prob>negative_Prob and actual==0):	#False positive
			mat[0][1]+=1
		elif(negative_Prob>positive_Prob and actual==0):	#True negative
			mat[0][0]+=1
		elif(negative_Prob>positive_Prob and actual==1):	#False negative
			mat[1][0]+=1
		
	print('Confusion matrix:')
	print(mat)
	Accuracy = (mat[1][1]+mat[0][0])/(mat[1][1]+mat[0][0]+mat[0][1]+mat[1][0]) 
	print('Accuracy is: '+str(Accuracy))
	Precision = mat[1][1]/(mat[1][1]+mat[0][1])
	Recall = mat[1][1]/(mat[1][1]+mat[1][0])
	print('Precision is: '+str(Precision))
	print('Recall is: '+str(Recall))
	f1_score = (2*Precision*Recall)/(Precision+Recall)
	print('F1 score is: '+str(f1_score))



def bayes(Data):											#Calculating frequency of words
	training=[]
	test=[]
	bag_Of_Words={}
	global num_Pos
	global num_Neg
	count=0
	positive_Words=0.0
	negative_Words=0.0
	unique_pos=0.0
	unique_neg=0.0
	reader=open(Data)
	for line in reader:
		count+=1
		if(count%5==0):										#Splitting dataset into 80 20
			test.append(line)
			continue
		else:
			training.append(line)
		n=0
		#training the model
		line=line.replace("\n","")
		words=re.split('[.?, ]',line)						#splitting the line into tokens
		done_words={}
		if(words[1]=="neg"):								#Counting positive and negative sentiments
			num_Neg+=1
		else:
			num_Pos+=1
		for word in words:									
			n+=1
			if(word!='' and n>4 and len(word)>1):			#ignoring empty tokens and tokens of length 1
				try:
					_=done_words[word]
					continue
				except:
					try:
						if(words[1]=="neg"):

							bag_Of_Words[word][0]+=1
						else:
							bag_Of_Words[word][1]+=1
					except:
						if(words[1]=="neg"):
							bag_Of_Words[word]=[1,0]
						else:
							bag_Of_Words[word]=[0,1]
					done_words[word]=True
	for word in bag_Of_Words:
		positive_Words+=bag_Of_Words[word][1]
		negative_Words+=bag_Of_Words[word][0]
		if(bag_Of_Words[word][0]>0):
			unique_neg+=1
		if(bag_Of_Words[word][1]>0):
			unique_pos+=1

	#print(len(test))
	#print(len(training))
	prediction(test,bag_Of_Words,positive_Words,negative_Words,unique_pos,unique_neg)	#Calling prediction on test dataset
bayes('naive_bayes_data.txt')
