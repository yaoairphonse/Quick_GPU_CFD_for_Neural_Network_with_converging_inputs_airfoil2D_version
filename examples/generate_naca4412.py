"""Generate the standard NACA 4412 coordinates, with a finite trailing edge."""
from pathlib import Path
import numpy as np

def generate(panels=100):
    x=.5*(1-np.cos(np.linspace(0,np.pi,panels+1)))
    m,p,t=.04,.4,.12
    yc=np.where(x<p,m/p**2*(2*p*x-x*x),m/(1-p)**2*((1-2*p)+2*p*x-x*x))
    dy=np.where(x<p,2*m/p**2*(p-x),2*m/(1-p)**2*(p-x))
    angle=np.arctan(dy)
    yt=5*t*(.2969*np.sqrt(x)-.1260*x-.3516*x*x+.2843*x**3-.1015*x**4)
    upper=np.column_stack((x-yt*np.sin(angle),yc+yt*np.cos(angle)))
    lower=np.column_stack((x+yt*np.sin(angle),yc-yt*np.cos(angle)))
    points=np.vstack((upper[::-1],lower[1:]))
    patches=np.r_[np.ones(panels),np.full(panels,2),3]
    return np.column_stack((points,patches))

if __name__=='__main__':
    target=Path(__file__).with_name('NACA4412_polygon.csv')
    np.savetxt(target,generate(),delimiter=',',header='x,y,patch',comments='',fmt=['%.17g','%.17g','%d'])
    print(target)
