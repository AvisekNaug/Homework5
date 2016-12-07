import operator
import math
examples = []
with open('TrainingDataSet.txt') as inputFile:
	for line in inputFile:
		age,spectacle,astigmatism,tear,lense = line.split(',')
		temp = {'AGE':age,'SPECTACLE':spectacle,'ASTIGMATISM':astigmatism,'TEAR':tear,'LENSE':lense[0]}
		examples.append(temp)
global feature
feature = {'AGE':['Y','P','B'],'SPECTACLE':['M','H'],'ASTIGMATISM':['N','Y'],'TEAR':['R','N']}
attributes = set(feature.keys())
parentExamples = examples

def pluralityValue(givenExamples): # retruns the label category value with highest frequency
	counter = {'N':0,'S':0,'H':0}
	for i4 in givenExamples:
		if i4['LENSE']=='N':
			counter['N'] = counter['N'] + 1
		elif i4['LENSE']=='S':
				counter['S'] = counter['S'] + 1
		else:
			counter['H'] = counter['H'] + 1
	maxCategoryvalue = max(counter, key=lambda i: counter[i])
	return maxCategoryvalue

def InformationContent(examples):
	tn = 0
	ts = 0
	th = 0
	for i in examples:
		if i['LENSE'] == 'N':
			tn = tn + 1
		elif i['LENSE'] == 'S':
			ts = ts + 1
		else:
			th = th + 1
	en = 0
	if tn == 0:
		en = 0.01
	es = 0
	if ts == 0:
		es = 0.01
	eh = 0
	if th == 0:
		eh = 0.01
	sum = 1.0*len(examples)
	if sum!=0:
		IC = -(tn*math.log((tn+en)/sum,2)+ ts*math.log((ts+es)/sum,2)+ th*math.log((th+eh)/sum,2))/sum
	else:
		IC = 0
	return IC

def Importance(attributes,examples):
	IC = InformationContent(examples)
	importanceDictionary = {}
	examplesInaClass = {}
	#Empty initialization
	for i5 in attributes:
		examplesInaClass[i5] = {}
		for i6 in feature[i5]:
			examplesInaClass[i5][i6] = {}
			examplesInaClass[i5][i6]['N'] = 0
			examplesInaClass[i5][i6]['S'] = 0
			examplesInaClass[i5][i6]['H'] = 0
	#Starting to find out distribution of each possible attribute
	for i5 in attributes: #i5 is AGE,SPECTACLE,ASTIGMATISM,TEAR
		importanceDictionary[i5] = IC
		for i6 in feature[i5]: #i6 is for AGE:Y,B,P or SPECTACLE:M,H etc
			for i7 in examples:
				if i7[i5]==i6:
					if i7['LENSE']=='N':
						examplesInaClass[i5][i6]['N'] = examplesInaClass[i5][i6]['N'] + 1
					elif i7['LENSE']=='S':
						examplesInaClass[i5][i6]['S'] = examplesInaClass[i5][i6]['S'] + 1
					else:
						examplesInaClass[i5][i6]['H'] = examplesInaClass[i5][i6]['H'] + 1
			tn = examplesInaClass[i5][i6]['N']
			en = 0
			if tn==0:
				en = 0.01
			ts = examplesInaClass[i5][i6]['S']
			es = 0
			if ts==0:
				es = 0.01
			th = examplesInaClass[i5][i6]['H']
			eh = 0
			if th==0:
				eh = 0.01
			sum = tn + ts + th
			sum = sum*1.0
			#Calculate Entropy for each attribute value i6
			if sum!=0:
				examplesInaClass[i5][i6]['Entropy'] = -(tn*math.log((tn+en)/sum,2)+ ts*math.log((ts+es)/sum,2)+ th*math.log((th+eh)/sum,2))/sum
			else:
				examplesInaClass[i5][i6]['Entropy'] = 0
			importanceDictionary[i5] = importanceDictionary[i5] - sum*examplesInaClass[i5][i6]['Entropy']/len(examples)
	#print importanceDictionary
	return importanceDictionary

def DecisionLearningTree(examples,attributes,parentExamples):
	#print attributes
	if not examples:
		return pluralityValue(parentExamples)
	sameClassification = True
	for i1 in xrange((len(examples)-1)):
		if examples[i1]['LENSE']!= examples[i1+1]['LENSE'] :
			sameClassification = False
			break
	if sameClassification:
		return examples[0]['LENSE']
	if not attributes:
		return pluralityValue(examples)
	else:
		attributeImportance = Importance(attributes,examples) #return a dictionary of {attribute:informationGain}
		A = max(attributeImportance, key=lambda i: attributeImportance[i])
		j = set([A])
		tree = {}
		tree[A] = {}
		for i2 in feature[A]:
			exs = []
			for i3 in examples:
				if i3[A]==i2:
					exs.append(i3)
			reduced_attribute = attributes-j
			#print 'reduced set is' , reduced_attribute
			subtree = DecisionLearningTree(exs,reduced_attribute,examples)
			tree[A][i2] = subtree
		return tree
lasttree = DecisionLearningTree(examples,attributes,parentExamples)

print lasttree
