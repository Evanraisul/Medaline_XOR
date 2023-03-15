import numpy as np
import matplotlib.pyplot as plt

def step(n):
    if n>=0:
        return 1
    return -1

# weight initialization
w=np.round((np.random.rand(6)*10)-5,2)
w1=w[0:2]

w2=w[2:4]
b1=w[4]
b2=w[5]

print(w1,w2,b1,b2)

learningRate=.1
v1=.5
v2=.5
b3=.5

input=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
targatOutput=np.array([-1,1,1,-1])
print("Input: \n",input,"\nTarget Output: ",targatOutput)

plt.plot([1,.1],[.1,1])
plt.plot([0,0,1,1],[0,1,0,1],'o')
plt.show()

maxiteration=700
t=0
error=np.array([0,0,0,1])
e=np.sum(error)
value=[[w1,w2,b1,b2]]

def fun(w5, lR, tO, hO,inp):
    x=round((w5[0]+lR*(tO-hO)*inp[0]),2)
    y=round(w5[1]+lR*(tO-hO)*inp[1],2)
    z=round(b1+lR*(tO-hO),2)
    return x,y,z

while(t<maxiteration and e!=0):
    for i in range(len(input)):
        #print("y1: ",np.dot(w1,input[i])+b1,"y2: ",np.dot(w2,input[i])+b2)
        hiddenOutput1=step(np.dot(w1,input[i])+b1)
        hiddenOutput2=step(np.dot(w2,input[i])+b2)
        actualOutput=step(np.dot(np.array([v1,v2]),np.array([hiddenOutput1,hiddenOutput2]))+b3)
        # print(hiddenOutput1, hiddenOutput2, actualOutput)
        if actualOutput!=targatOutput[i]:
            error[i]=1
            if targatOutput[i]==-1:
                if hiddenOutput1>=0:
                    w1[0],w1[1],b1=fun(w1,learningRate,targatOutput[i],hiddenOutput1,input[i])
                if hiddenOutput2>=0:
                    w2[0],w2[1],b2=fun(w2,learningRate,targatOutput[i],hiddenOutput2,input[i])
            elif targatOutput[i]==1:
                temp=min(hiddenOutput1,hiddenOutput2,key=abs)
                if temp == hiddenOutput1:
                    w1[0],w1[1],b1=fun(w1,learningRate,targatOutput[i],hiddenOutput1,input[i])
                if temp == hiddenOutput2:
                   w2[0],w2[1],b2=fun(w2,learningRate,targatOutput[i],hiddenOutput2,input[i])
        else:
            error[i]=0
    value.append([w1,w2,b1,b2])
    e=np.sum(error)
    t+=1
    
print(value)
print("Iteration: ",t)
print(error)