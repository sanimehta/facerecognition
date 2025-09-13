import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(0,11) 
y = x + 1 

plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 

#plt.plot(x, y, marker='o', color='b', linestyle='') 
plt.scatter(x,y,marker='o',color='b')#doesnt consider linestyle even if written.

plt.show()