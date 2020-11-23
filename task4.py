import task3
import os
import numpy as np
import math
from collections import defaultdict

def main():
    # TAKEN FROM TASK 3 #

    layers = int(input("Enter the amount of layers you want: "))
    hashes = int(input("Enter the amount of hashes per layer you want: "))

    bins = task3.LSH(layers, hashes)

    gesture = input("Enter a gesture file name: ")
    t = int(input("Enter how many similar gestures you want: "))

    output, count = task3.findGestures(bins, gesture, t)

    print()
    print()
    print(output)
    print()
    print()

    # TAKEN FROM TASK 3 #

    wIndex=np.loadtxt("wcVectors.txt",dtype=str,skiprows=0,max_rows=1) # Load 1st line of word count vectors (index)
    weights=np.empty(wIndex.shape)                                     # Create empty array for probabalistic weights
    vectors=np.empty([len(output),wIndex.shape[0]])                    # Create empty array that will store the word count vectors for each file returned in output

    relevant=[]                                                        # Holds user input
    files=os.listdir("C:/Users/njepa/Desktop/data/W")                  # Get list of files in order, use your own directory. Ex: "C:/Users/njepa/Desktop/data/W"

    # Append user input to relevant and load the corresponding word count vectors for the files returned in output
    for i in range(len(output)):
        relevant.append(input("Is " + output[i] + " relevant? (Y or N): "))
        vectors[i,:]=np.loadtxt("wcVectors.txt",dtype=str,skiprows=1+(files.index(output[i])),max_rows=1)
    

    r=relevant.count("y")+relevant.count("Y")   # R = total # of relevant results
    n=len(output)                               # N = total # of results (unclear in the paper)

    # Calculate probabalistic weights for each word
    for i in range(weights.shape[0]):
        terms=np.array(vectors[:,i])
        ri=0
        ni=0
        for i in range(terms.shape[0]):
            if relevant[i].lower()=="y" and terms[i] > 0: 
                ri+=1                                           # ri = # of relevant results with word count > 0
                ni+=1                                           # ni = # of results with word count > 0
            elif relevant[i].lower()=="n" and terms[i] > 0:
                ni+=1                                           # ni = # of results with word count > 0
        pi = (ri+0.5)/(r+1)
        ui = ((ni-ri+0.5)/(n-r+1))
        weights[i]=math.log( (pi*(1-ui)) / (ui*(1-pi)) ,10)     # Store probabalistic weight

    # Load all the files in the data directory and calculate their relevance score based on the small subset that was used from task 3.
    scores={}

    ######################################################################### This is to run the ranking on ALL files, does not give consistent results

    # for i in range(len(files)):
    #     vector=np.loadtxt("wcVectors.txt",dtype=int,skiprows=1+i,max_rows=1)
    #     score=0
    #     for j in range(vector.shape[0]):
    #         if vector[j] > 0:
    #             score+=1*weights[j]
    #     scores[files[i]] = score

    #########################################################################

    for i in range(len(output)):
        vector=np.loadtxt("wcVectors.txt",dtype=int,skiprows=1+(files.index(output[i])),max_rows=1)
        score=0
        for j in range(vector.shape[0]):
            if vector[j] > 0:
                score+=1*weights[j]
        scores[output[i]] = score

    results=defaultdict(list)
    for k, v in scores.items():
        results[v].append(k)
    
    # Print reordered results, rarely ever improves ordering
    for k, v in sorted(results.items(), reverse=True):
        print(k, ', '.join(v))
    
    


if __name__ == "__main__":
    main()