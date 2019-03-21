# Name: Jade Garisch
# Email: jadeng@brandeis.edu
# Last updated: March 13, 2019
# Program:  Naive Bayes Classifier and Evaluation


import os as os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.feature_dict = {0: 'great', 1: 'amazing', 2: 'terrible', 3: 'enjoyed', 4: 'beautiful', 5: 'awful',
                             6: 'loved', 7: 'hated', 8: 'best', 9: 'worst', 10: 'boring', 11: 'award', 12: 'lame ',
                             13: 'mess ', 14: 'outstanding', 15: 'waste', 16: 'ridiculous', 17: 'stupid', 18: 'bad',
                             19: 'perfect', 20: 'life', 21: 'memorable', 22: 'dull', 23: 'poor', 24: 'plot', 25: 'taste',
                             26: 'excellent', 27: 'romantic', 28: 'funny', 29: 'cinematography', 30: 'family',
                             31: 'mediocre', 32: 'problem', 33: 'top', 34: 'happy', 35: 'angry', 36: 'furious',
                             37: 'stunning', 38: 'exquisite', 39: 'uninteresting', 40: 'admire', 41: 'bliss',
                             42: 'charm', 43: 'delight', 44: 'enchanting', 45: 'endearing', 46: 'fearless',
                             47: 'fantastic', 48: 'gorgeous', 49: 'glorious', 50: 'heart', 51: 'hot',
                             52: 'impressive', 53: 'ingenuous', 54: 'inspire', 55: 'intimate', 56: 'joy',
                             57: 'lively', 58: 'magic', 59: 'marvelous', 60: 'masterpiece', 61: 'mesmorize',
                             62: 'noteworthy', 63: 'nurture', 64: 'orginality', 65: 'outshine', 66: 'passionate',
                             67: 'profound', 68: 'quality', 69: 'reward', 70: 'reaffirm', 71: 'rich', 72: 'satisfy',
                             73: 'sensation', 74: 'spellbinding', 75: 'suspense', 76: 'truth', 77: 'upbeat',
                             78: 'vivid', 79: 'warm', 80: 'wow', 81: 'yay', 82: 'zest', 83: 'wholeheartedly',
                             84: 'a+', 85: 'appall', 86: 'atrocious', 87: 'banal', 88: 'baffle', 89: 'bother',
                             90: 'cheap', 91: 'cliche', 92: 'crap', 93: 'bullshit', 94: 'creep', 95: 'depress',
                             96: 'disgrace', 97: 'dislike', 98: 'excruciating', 99: 'fuck',
                             100: 'far-fetched', 101: 'tired', 102: 'garish', 103: 'god-awful', 104: 'horrible',
                             105: 'horendous', 106: 'hurt', 107: 'inability', 108: 'ineffective', 109: 'garbage',
                             110: 'junk', 111: 'loathe', 112: 'lifeless', 113: 'meager', 114: 'melodramatic',
                             115: 'negative', 116: 'outrage', 117: 'pitiful', 118: 'profane', 119: 'rant',
                             120: 'regret', 121: 'ridiculous', 122: 'scum', 123: 'slow', 124: 'sucks',
                             125: 'thoughtless', 126: 'trash', 127: 'trivial', 128: 'unethical',
                             129: 'vile', 130: 'vulgar', 131: 'weak', 132: 'yawn', 133: 'uninspiring', 134: 'ugly',
                             135: 'annoying', 136: 'irritating', 137: 'cringe', 138: 'sweet', 139: 'favorite',
                             140: 'moving', 141: 'learn', 142: 'tells', 143: 'worth', 144: 'talent', 145: 'change',
                             146: 'pleased'}
        self.prior = np.zeros(2)
        self.likelihood = np.zeros((2, 147)) #uses classes and features
        self.doc_count = 0 #the number of items in document (should be negative + positive count)
        self.cnt_pos = 0
        self.cnt_neg = 0
        self.pos_doc_names = []
        self.neg_doc_names = []
        self.pos_dict = {}
        self.neg_dict = {}
        self.lem = WordNetLemmatizer()
        for num in range(0,len(self.feature_dict)):
            self.pos_dict.update({num: 0})
            self.neg_dict.update({num: 0})
    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    
    returns (D, C) returns log(c) and log(W|C)

    
    '''
    def train(self, train_set):
        print("training model...") #a message to human user to keep track of proccess
        pos = [] #list of positive words
        neg = [] #list of negative words

        for root, dirs, files in os.walk(train_set): #we will now iterate over files
            for file in files:
                self.doc_count += 1
                with open(os.path.join(root, file)) as f:
                    myfile = f.read()
                    if root[-3:] == 'pos':
                        self.pos_doc_names.append(file) #create a list of pos files
                        self.cnt_pos = self.cnt_pos + 1 #add to pos file count
                        pos = pos + self.lem.lemmatize(myfile.lower()).split() # add to pos vocab
                    if root[-3:] == 'neg':
                        self.neg_doc_names.append(file) #create a list of neg files
                        self.cnt_neg = self.cnt_neg + 1 #add to neg file count
                        neg = neg + self.lem.lemmatize(myfile.lower()).split()  # add to neg vocab
                    f.close()

        # Gathering a priori probabilities for each class and creating vocab
        self.prior[1] = np.log(self.cnt_pos / self.doc_count)  # positive
        self.prior[0] = np.log(self.cnt_neg / self.doc_count)  # negative
        vocab = set(neg + pos)
        #lemmatize words in feature dict for better results
        features = [self.lem.lemmatize(self.feature_dict.get(item)) for item in self.feature_dict]
        #add to pos_dict and neg_dict count, update self.likelihood for each class and word
        for num in range(len(self.feature_dict)):
            for word in pos:
                if word == features[num]:
                    self.pos_dict[num] += 1
            for word in neg:
                if word == features[num]:
                    self.neg_dict[num] += 1
            self.likelihood[1][num] = np.log( (self.pos_dict[num] + 1)  / (len(pos) + len(vocab)) )
            self.likelihood[0][num] = np.log((self.neg_dict[num] + 1) / (len(neg) + len(vocab)) )

        return self.prior, self.likelihood, vocab

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    
    '''
    def test(self, dev_set):
        print("creating dictionary of results...")
        results = {}

        for root, dirs, files in os.walk(dev_set):
            results.update({file: {'predicted': 'neg', 'correct': 'neg'} for file in files})
            for file in files:
                if root[-3:] == 'pos':
                    results[file]['correct'] ='pos'
                if root[-3:] == 'neg':
                    results[file]['correct'] ='neg'

        # for root, dirs, files in os.walk(dev_set):
                if root[-3:] == 'pos' or root[-3:] == 'neg':
                    with open(os.path.join(root, file)) as f:
                        myfile = f.read()
                        myfile = self.lem.lemmatize(myfile.lower()).split()
                        vector = np.zeros(len(self.feature_dict)) #created the vector
                        doc_likelihood = {0: 0, 1: 0}
                        for i in range(len(self.feature_dict)):
                            for word in myfile:
                                if self.lem.lemmatize(self.feature_dict[i]) == word:
                                    vector[i] += 1

                        for feature in range(len(self.feature_dict)):
                            doc_likelihood[0] = doc_likelihood[0] + self.likelihood[0][feature] * vector[feature]
                            doc_likelihood[1] = doc_likelihood[1] + self.likelihood[1][feature] * vector[feature]
                        doc_likelihood[0] = doc_likelihood[0] + self.prior[0]
                        doc_likelihood[1] = doc_likelihood[1] + self.prior[1]

                        if doc_likelihood[0] > doc_likelihood[1]:
                            results[file]['predicted'] = 'neg'
                        else:
                            results[file]['predicted'] = 'pos'
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        print("EVALUATION: ")
        pos_tp = 0
        pos_fp = 0
        pos_total = 0
        neg_tp = 0
        neg_fp = 0
        neg_total = 0

        for item in results:
            if results[item]['correct'] == 'pos':
                pos_total += 1
            else:
                neg_total += 1
            if results[item]['predicted'] == 'pos':
                if results[item]['correct'] == 'pos':
                    pos_tp += 1
                else:
                    pos_fp += 1
            else:
                if results[item]['correct'] == 'neg':
                    neg_tp += 1
                else:
                    neg_fp += 1

        #precision, recall and f-measure
        pos_precision = pos_tp / (pos_tp + pos_fp)
        print("precision for positive reviews: " + str(pos_precision))
        pos_recall = pos_tp / pos_total
        print("recall for positive reviews: " + str(pos_recall))
        pos_fmeasure = ((2 * pos_precision * pos_recall) / (pos_precision + pos_recall))
        print("f-measure for positive reviews: " + str(pos_fmeasure))

        neg_precision = neg_tp / (neg_tp + neg_fp)
        print("precision for negative reviews: " + str(neg_precision))
        neg_recall = neg_tp / neg_total
        print("recall for negative reviews: " + str(neg_recall))
        neg_fmeasure = ((2 * neg_precision * neg_recall) / (neg_precision + neg_recall))
        print("f-measure for negative reviews: " + str(neg_fmeasure))

        # # accuracy
        right = 0
        for item in results:
            if results[item]['predicted'] == results[item]['correct']:
                right +=1
        print('overall accuracy: ' + str(right/(neg_total+pos_total)))
        pass

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)
