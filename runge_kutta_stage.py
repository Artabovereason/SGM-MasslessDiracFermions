import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

def potential(position):
    return 1/(1+position[0]**2+position[1]**2)

"""
def force(position):
    return [2*position[0]/((1+position[0]**2+position[1]**2)**2),
            2*position[1]/((1+position[0]**2+position[1]**2)**2)]
"""

space_x = np.linspace(-10,10,100)
space_y = np.linspace(-10,10,100)

space_z = np.zeros((100,100))
for i in range(len(space_x)):
    for j in range(len(space_y)):
        space_z[i][j] = potential([space_x[i],space_y[j]])
#ax = plt.axes(projection='3d')
#space_x, space_y = np.meshgrid(space_x, space_y)
#ax.plot_surface(space_x, space_y, space_z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')



class electron:
    def __init__(self,starting_position,starting_velocity,number_of_iterations,sign):
        self.starting_position    = starting_position
        self.starting_velocity    = starting_velocity
        self.number_of_iterations = number_of_iterations
        self.list_of_p            = []
        self.list_of_r            = []
        self.sign                 = sign

    def force(self,position):
        return [self.sign*2*position[0]/((1+position[0]**2+position[1]**2)**2),
                self.sign*2*position[1]/((1+position[0]**2+position[1]**2)**2)]


    def numerical_scheme(self,method):
        self.method  = method
        '''
        There is a stability criteron : ∆t/4∆x^2 < 1
        '''
        self.step_space = 0.1
        self.step_time  = 0.01

        '''
        Euler's Method
        '''
        if self.method == "Euler":
            #print('Euler\'s method')
            for N in range(int(self.number_of_iterations)):
                if N == 0:
                    self.list_of_p.append(self.starting_velocity)
                    self.list_of_r.append(self.starting_position)
                else:
                    self.list_of_p.append([self.list_of_p[N-1][0]+self.step_time*(force(self.list_of_r[N-1])[0]),
                                           self.list_of_p[N-1][1]+self.step_time*(force(self.list_of_r[N-1])[1])])
                    self.list_of_r.append([self.list_of_r[N-1][0]+self.step_time*self.list_of_p[N-1][0]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2),
                                           self.list_of_r[N-1][1]+self.step_time*self.list_of_p[N-1][1]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2) ])

        elif self.method == "RK4":
            #print('Runge-Kutta 4')

            for N in range(int(self.number_of_iterations)):

                kn1 = []
                ln1 = []
                kn2 = []
                ln2 = []
                kn3 = []
                ln3 = []
                kn4 = []
                ln4 = []
                if N == 0:
                    self.list_of_p.append(self.starting_velocity)
                    self.list_of_r.append(self.starting_position)
                else:
                    kn1.append([self.force([self.list_of_r[N-1][0],self.list_of_r[N-1][1]])[0],
                                self.force([self.list_of_r[N-1][0],self.list_of_r[N-1][1]])[1]])
                    ln1.append([self.list_of_p[N-1][0]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2),
                                self.list_of_p[N-1][1]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2)])
                    kn2.append([self.force([self.list_of_r[N-1][0]+self.step_time*ln1[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln1[0][1]/2.0])[0],
                                self.force([self.list_of_r[N-1][0]+self.step_time*ln1[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln1[0][1]/2.0])[1]])
                    ln2.append([(self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)**2)),
                                (self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)**2))])
                    kn3.append([self.force([self.list_of_r[N-1][0]+self.step_time*ln2[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln2[0][1]/2.0])[0],
                                self.force([self.list_of_r[N-1][0]+self.step_time*ln2[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln2[0][1]/2.0])[1]])
                    ln3.append([(self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)**2)),
                                (self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)**2))])
                    kn4.append([self.force([self.list_of_r[N-1][0]+self.step_time*ln3[0][0],self.list_of_r[N-1][1]+self.step_time*ln3[0][1]])[0],
                                self.force([self.list_of_r[N-1][0]+self.step_time*ln3[0][0],self.list_of_r[N-1][1]+self.step_time*ln3[0][1]])[1]])
                    ln4.append([(self.list_of_p[N-1][0]+self.step_time*kn3[0][0])/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn3[0][0])**2+(self.list_of_p[N-1][1]+self.step_time*kn3[0][1])**2)),
                                (self.list_of_p[N-1][1]+self.step_time*kn3[0][1])/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn3[0][0])**2+(self.list_of_p[N-1][1]+self.step_time*kn3[0][1])**2))])
                    self.list_of_p.append([self.list_of_p[N-1][0]+(self.step_time/6.0)*(kn1[0][0]+2*kn2[0][0]+2*kn3[0][0]+kn4[0][0]),
                                           self.list_of_p[N-1][1]+(self.step_time/6.0)*(kn1[0][1]+2*kn2[0][1]+2*kn3[0][1]+kn4[0][1])])
                    self.list_of_r.append([self.list_of_r[N-1][0]+(self.step_time/6.0)*(ln1[0][0]+2*ln2[0][0]+2*ln3[0][0]+ln4[0][0]),
                                           self.list_of_r[N-1][1]+(self.step_time/6.0)*(ln1[0][1]+2*ln2[0][1]+2*ln3[0][1]+ln4[0][1])])
                if potential(self.list_of_r[N])>0.2:
                    self.sign = -1# self.sign
                elif potential(self.list_of_r[N])<0.2:
                    self.sign = +1
                else:
                    pass


        else :
            print('Not implemented yet')


for j in np.linspace(-np.pi/4,np.pi/4,10):
    if j==0:
        pass
    else:
        electron1 = electron([-6,0],[np.cos(j),np.sin(j)],2000,+1)
        electron1.numerical_scheme('RK4')
        plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='blue')

'''
for j in np.linspace(-np.pi/2,np.pi/2,20):
    if j==0:
        pass
    else:
        electron1 = electron([-6,0],[np.cos(j),np.sin(j)],2000,-1)
        electron1.numerical_scheme('RK4')
        plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='green')


'''



plt.xlim([-6, 10])
plt.contour(space_x,space_y,space_z, origin='lower', cmap=cm.Reds, levels=30)
plt.colorbar()
plt.xlabel('$x$ axis')
plt.ylabel('$y$ axis')

plt.show()
