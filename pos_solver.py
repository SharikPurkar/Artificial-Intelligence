###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
#Tejashree Khot:tpkhot
#
# (Based on skeleton code by D. Crandall)
#
#
####
"""
The following code implements 5 POS tagging algorithms.

Training:
In training we just precompute all probabilities for:
P(s_i+1|s_i), P(w_i|s_i),P(w),P(s),P(s at start of sentence)

Naive Bayes POS Tagger:
Naive Bayes model assumes that tag depends only on current word.
If the word to be tagged has appeared in the training data, then we pick tag with highest P(s|w)
P(s|w)=P(w|s)*P(s)/P(w)

If the word appears in the training data, but P(w|s)=0
We consider P(w|s)=0.000000001


Also, there is a small tweak to algorithm.
If the word never appears in training data, the word is assigned the most common tag instead of a random tag.


Sampling POS Tagger:

For Sampling, Gibbs sampling method is used and the full model is assumed.
First we generate an initial sample.
To generate initial sample, we have randomly selected tags from P(s|w) distribution for each word separately.

Then iteratively we generate a new sample from older sample.
New sample is initially just copy of older sample.
Then we iteratively update s1 to sn in new sample as follows:
{
P(s_1|w_1)=P(s_1 at start)*P(w_1|S_1)*P(s_2|s_1)
then we sample s1 from above distribution

for i-> s2 to sn-1:
P(s_i|w_i)=P(s_i|s_i-1)*P(w_i|s_i)*P(s_i+1|s_i)
s_i is sampled from above distribution

P(s_n|w_n)=P(s_n|s_n-1)*P(w_n|s_n)
s_n is sampled from above distribution
}
Tweak in algorithm:
However if the word never appears in training set the following probability distributions are used for sampling:

P(s_1|w_1)=P(s_1 at start)*P(s_2|s_1)
then we sample s1 from above distribution

for i-> s2 to sn-1:
P(s_i|w_i)=P(s_i|s_i-1)*P(s_i+1|s_i)
s_i is sampled from above distribution

P(s_n|w_n)=P(w_n|s_n)
s_n is sampled from above distribution

Also, in these calculations no Probability is considered 0.
In case a particular phenomenon doesn't occur in training, it's probability is considered to be 0.000000001.

We thus iteratively generate 5 samples and 5th sample is used as an output.
This algorithm converges to actual distribution very quickly.
Moreover, due to the randomness of this algorithm, it gives inconsistent results. These results can be shockingly good or bad.


Approximate max-marginal inference POS Tagger:

For Approximate inference, we use Gibbs sampling exactly as above to generate 1000 samples for each sentence.
Using these samples we calculate approximate P(s|w).
Then we select the most likely tag as per these approximate calculations.
We are also displaying confidence of each tag.
Approximated over 1000 samples, it gives way better results than the sampler.
It theoretically could give best results due to it's randomness.
However, it's scores are marginally less than that of Viterbi Algorithm.
Moreover, it is the slowest of all the POS Taggers

Exact maximum a posteriori inference POS tagger:

This uses the Viterbi Algorithm. Viterbi algorithm does exact inference.
We get best results using Viterbi Algorithm because it calculates the probability of entire sequence instead of just each word.

We select sequence such that has maximum possibility.

To calculate probability of sequences, we dynamically implement the following:
P(s_i|w_i)=max (P(s_i-1|w_i-1)*P(s_i|s_i-1)*P(w_i|s_i)) for all possible values of s_i-1

It was mainly because of Viterbi coding, that the assumption if a phenonmenon does occur, instead of P=0 we keep P=0.000000001
This generalization was then made throughout the program.

Viterbi algorithm gives the best result among the four algorithms and  also much faster than Approximate Max-Marginal Inference.

Best Algorithm:
The best algorithm we have implemented is just a little modification of Viterbi algorithm.
P(s_i|w_i)=max (P(s_i-1|w_i-1)*P(s_i|s_i-1)*P(s_i)) for all possible values of s_i-1

We replaced P(w_i|s_i)=0.000000001 by P(s_i).
However, this tweak improves the accuracy by only a little over a percent.
Also, it is much faster than Approximate Max-Marginal Inference.


Results of evaluation of bc.test file:
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
          1. Naive:       93.96%               47.50%
        2. Sampler:       92.16%               38.50%
   3. Max marginal:       95.08%               54.20%
            4. MAP:       95.04%               54.45%
           5. Best:       95.28%               55.60%


As seen above the following is ranking of tagger performance wise:
1. Best
2. MAP
3. Max Marginal
4. Naive Bayes
5. Sampler
"""
####



# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import copy
import random
import math

class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag,start
        prob=1
        for i in range (0,len(sentence)):
            if i==0:
                if (sentence[i],label[i]) in probwgs:
                    prob=probstart[label[i]]*probwgs[(sentence[i],label[i])]
                else:
                    prob=probstart[label[i]]*0.000000001
            else:
                if (sentence[i],label[i]) in probwgs:
                    prob=prob*probsgs[label[i],label[i-1]]*probwgs[(sentence[i],label[i])]
                else:
                    prob=prob*probsgs[label[i],label[i-1]]*0.000000001

        return math.log(prob)

    # Do the training!
    #
    def train(self, data):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag,start
        tag={}
        totaltag=0.0
        for i in data:
            for j in i[1]:
                totaltag+=1
                if j not in tag:
                    tag[j]=1
                else:
                    tag[j]+=1
        totalprobtag=0
        for key in tag:
            tag[key]=tag[key]/totaltag
            totalprobtag+=tag[key]
        tags=tag.keys()
        word={}
        totalword=0.0
        for i in data:
            for j in i[0]:
                totalword+=1
                if j not in word:
                    word[j]=1
                else:
                    word[j]+=1
        totalprobword=0
        for key in word:
            word[key]=word[key]/totalword
            totalprobword+=word[key]
        words=word.keys()
        probwgs={}
        for i in words:
            for j in tags:
                probwgs[(i,j)] = 0.000000001;
        for i in data:
            for j in i[0]:
                k=i[0].index(j)
                probwgs[(j,i[1][k])] += (1/(tag[i[1][k]]*totaltag))
        probsgs={}
        probstart={}
        probprevtag={}
        probprevprevtag={}
        for i in tags:
            probprevtag[i]=0.000000001
            probprevprevtag[i]=0.000000001
            probstart[i]=0.000000001
            for j in tags:
                probsgs[(i,j)]=0.000000001
        for i in data:
            for j in range (0,len(i[1])):
                if j==0:
                    if i[1][j] in probstart:
                        probstart[i[1][j]]+=1.0/len(data)
                    else:
                        probstart[i[1][j]]=1.0/len(data)
                elif j==1:
                    probsgs[(i[1][j],i[1][j-1])]+=1.0
                    probprevtag[i[1][j-1]]+=1.0
                else:
                    probsgs[(i[1][j],i[1][j-1])]+=1.0
                    probprevtag[i[1][j-1]]+=1.0
                    probprevprevtag[i[1][j-2]]+=1.0
        for j in probsgs:
            probsgs[j]=probsgs[j]/probprevtag[j[1]]
        mostprobabletag=[key for key,val in tag.iteritems() if val == max(tag.values())]

    # Functions for each algorithm.
    def naive(self, sentence):
        global tag, word, probwgs,tags,words,mostprobabletag
        probsentence=[]
        for i in sentence:
            if i not in word:
                probsentence.append(mostprobabletag[0])
            else:
                probtgw=[]
                for j in tags:
                    probtgw.append(probwgs[(i,j)]*tag[j]/word[i])
                maxprob=max(probtgw)
                maxprobindex=probtgw.index(maxprob)
                probsentence.append(tags[maxprobindex])

        return [ [probsentence], [] ]

    def mcmc(self, sentence, sample_count):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag,start
        initial=[]
        Samples=[]
        for i in sentence:
            if i not in word:
                initial.append(mostprobabletag[0])
            else:
                probtgw=[]
                total=0
                cumulative=0
                for j in tags:
                    probtgw.append(probwgs[(i,j)]*tag[j]/word[i])
                    total+=probwgs[(i,j)]*tag[j]/word[i]
                probtgwcumm=[]
                for j in probtgw:
                    x=j+cumulative
                    probtgwcumm.append(x)
                    cumulative=x
                r=random.uniform(0,probtgwcumm[-1])
                for j in range (0,12):
                    if probtgwcumm[j]>r:
                        break
                initial=initial+[tags[j]]
        Samples.append(initial)
        for n in range (1,sample_count):
            newsample=copy.deepcopy(Samples[-1])
            sam=[]
            for i in range(0,len(sentence)):
                tagsp=[]
                total=0
                for j in tags:
                    if len(sentence)>1:
                        if i==0:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(newsample[i+1],j)]*probstart[j]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(newsample[i+1],j)]*probstart[j]
                                tagsp.append(x)
                                total+=x
                        elif i!=len(sentence)-1:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(j,newsample[i-1])]*probsgs[(newsample[i+1],j)]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(j,newsample[i-1])]*probsgs[(newsample[i+1],j)]
                                tagsp.append(x)
                                total+=x
                        else:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(j,newsample[i-1])]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(j,newsample[i-1])]
                                tagsp.append(x)
                                total+=x
                    else:
                        if  (sentence[i],j) in probwgs:
                            x=probwgs[sentence[i],j]*probstart[j]
                            tagsp.append(x)
                            total+=x
                        else:
                            x=probstart[j]
                            tagsp.append(x)
                            total+=x
                probtgwcumm=[]
                cumulative=0
                for j in tagsp:
                    x=j+cumulative
                    probtgwcumm.append(x)
                    cumulative=x
                r=random.uniform(0,probtgwcumm[-1])
                for j in range (0,12):
                    if probtgwcumm[j]>r:
                        break
                sam=sam+[tags[j]]
            Samples.append(sam)
        return [ Samples, [] ]

    def best(self, sentence):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag,start
        V={}
        path={}
        for i in range(0,len(sentence)):
            w=sentence[i]
            if i==0:
                for t in probstart:
                    if (w,t) in probwgs:
                        V[(i,t)]=probstart[t]*probwgs[(w,t)]
                    else:
                        V[(i,t)]=probstart[t]*0.000000001
                    path[(i,t)]=[t]
            else:
                for t in tags:
                    l=[]
                    for tp in tags:
                        if (w,t) in probwgs:
                            p=V[(i-1,tp)]*probwgs[(w,t)]*probsgs[(t,tp)]
                        else:

                            p=V[(i-1,tp)]*tag[t]*probsgs[(t,tp)]
                        l.append(p)

                    V[(i,t)]=max(l)
                    previ=l.index(max(l))
                    prev=tags[previ]
                    path[(i,t)]=path[(i-1,prev)]+[t]
        l=[]
        i=len(sentence)-1
        for t in tags:
                p=V[(i,t)]
                l.append(p)
        maxprob=max(l)
        ind=l.index(maxprob)
        finaltag=tags[ind]
        problist=path[(len(sentence)-1,finaltag)]
        return [ [problist], [] ]

    def max_marginal(self, sentence):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag,start
        initial=[]
        Samples=[]
        counts=[[0 for i in range(0,len(tags))] for j in range(0,len(sentence))]
        for i in sentence:
            if i not in word:
                initial.append(mostprobabletag[0])
            else:
                probtgw=[]
                total=0
                cumulative=0
                for j in tags:
                    probtgw.append(probwgs[(i,j)]*tag[j]/word[i])
                    total+=probwgs[(i,j)]*tag[j]/word[i]
                probtgwcumm=[]
                for j in probtgw:
                    x=j+cumulative
                    probtgwcumm.append(x)
                    cumulative=x
                r=random.uniform(0,probtgwcumm[-1])
                for j in range (0,12):
                    if probtgwcumm[j]>r:
                        break
                initial=initial+[tags[j]]
        for w in range(0,len(sentence)):
            for i in range (0,12):
                if initial[w]==tags[i]:
                    counts[w][i]+=1
        Samples.append(initial)
        for n in range (1,1000):
            newsample=copy.deepcopy(Samples[-1])
            sam=[]
            for i in range(0,len(sentence)):
                tagsp=[]
                total=0
                for j in tags:
                    if len(sentence)>1:
                        if i==0:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(newsample[i+1],j)]*probstart[j]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(newsample[i+1],j)]*probstart[j]
                                tagsp.append(x)
                                total+=x
                        elif i!=len(sentence)-1:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(j,newsample[i-1])]*probsgs[(newsample[i+1],j)]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(j,newsample[i-1])]*probsgs[(newsample[i+1],j)]
                                tagsp.append(x)
                                total+=x
                        else:
                            if  (sentence[i],j) in probwgs:
                                x=probwgs[sentence[i],j]*probsgs[(j,newsample[i-1])]
                                tagsp.append(x)
                                total+=x
                            else:
                                x=probsgs[(j,newsample[i-1])]
                                tagsp.append(x)
                                total+=x
                    else:
                        if  (sentence[i],j) in probwgs:
                            x=probwgs[sentence[i],j]*probstart[j]
                            tagsp.append(x)
                            total+=x
                        else:
                            x=probstart[j]
                            tagsp.append(x)
                            total+=x
                probtgwcumm=[]
                cumulative=0
                for j in tagsp:
                    x=j+cumulative
                    probtgwcumm.append(x)
                    cumulative=x
                r=random.uniform(0,probtgwcumm[-1])
                for j in range (0,12):
                    if probtgwcumm[j]>r:
                        break
                sam=sam+[tags[j]]
                counts[i][j]+=1
            Samples.append(sam)
        finaltags=[]
        confi=[]
        for i in counts:
            m=max(i)
            mind=i.index(m)
            finaltags+=[tags[mind]]
            confi+=[m/1000.0]
        return [ [finaltags], [confi] ]

    def viterbi(self, sentence):
        global tag, word, probwgs,tags,words, probsgs,probstart,mostprobabletag
        V={}
        path={}
        for i in range(0,len(sentence)):
            w=sentence[i]
            if i==0:
                for t in probstart:
                    if (w,t) in probwgs:
                        V[(i,t)]=probstart[t]*probwgs[(w,t)]
                    else:
                        V[(i,t)]=probstart[t]*0.000000001
                    path[(i,t)]=[t]
            else:
                for t in tags:
                    l=[]
                    for tp in tags:
                        if (w,t) in probwgs:
                            p=V[(i-1,tp)]*probwgs[(w,t)]*probsgs[(t,tp)]
                        else:
                            p=V[(i-1,tp)]*0.000000001*probsgs[(t,tp)]
                        l.append(p)

                    V[(i,t)]=max(l)
                    previ=l.index(max(l))
                    prev=tags[previ]
                    path[(i,t)]=path[(i-1,prev)]+[t]
        l=[]
        i=len(sentence)-1
        for t in tags:
                p=V[(i,t)]
                l.append(p)
        maxprob=max(l)
        ind=l.index(maxprob)
        finaltag=tags[ind]
        problist=path[(len(sentence)-1,finaltag)]
        return [ [problist], [] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

