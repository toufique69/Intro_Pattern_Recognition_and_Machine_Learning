# Linear Solver
def my_linfit(x,y):
    
    x_b = np.mean(x)
    y_b = np.mean(y)
    
    dividend = 0
    divisor = 0
    
    for i in range(len(x)):
        dividend = dividend + ((y[i]-y_b)*(x[i]-x_b))
        divisor = divisor + (x[i]-x_b)**2
    
    a = dividend / divisor
    b = y_b -(a*x_b)
    
    return a,b
  
from pynput import mouse
import matplotlib.pyplot as plt
import numpy as np
x = []
y = []


def on_click(a,b,button, pressed):
    if pressed:
        if str(button) == "Button.left":
            x.append(a)
            y.append(b)
        elif str(button) == "Button.right":
            return False
    
#mouse click listener       
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

a,b = my_linfit(x, y)
plt.title('Fitting 2D linear model to N > 2 training points')
plt.plot(x,y,'kx')
xp = np.arange(0,max(x),0.1)
plt.plot(xp,a*xp+b,'r-')
plt.axis([0,1600,0,1600])
print (f"My_fit : a={a}_and_b={b}")
plt.show( )