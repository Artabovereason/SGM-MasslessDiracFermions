import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d



'''
# For the Barrier potential (either smooth or abrupt) one has to use :
def potential(position):
    if position[0] < 0 :
        return 0
    elif position[0] >0 and position[0] <2:
        return position[0]*0.4
    else:
        return 0.8

# and put this force in the electron class

def force(self,position):
    if position[0]<0 :
        value_value = 0
    elif position[0]>0 and position[0]<2:
        value_value = 0.1
    else :
        value_value = 0
    return [-value_value,
            0]

# with this condition afterwards

if potential(self.list_of_r[N]) > np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2):
    if   self.list_of_p[N][1] < 0 and self.list_of_r[N][1]>0:
        pass
    elif self.list_of_p[N][1] > 0 and self.list_of_r[N][1]<0:
        pass
    else :
        self.list_of_p[N][1] = -self.list_of_p[N][1]


elif potential(self.list_of_r[N]) < np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2)  :
    self.sign = +1

else:
    pass

'''


intensity_parameter = 0.8 #Intensity of the potential
def potential(position):
    return intensity_parameter/(1+position[0]**2+position[1]**2)

space_x = np.linspace(-10,10,100)
space_y = np.linspace(-10,10,100)
space_z = np.zeros((100,100))
for i in range(len(space_x)):
    for j in range(len(space_y)):
        space_z[i][j] = potential([space_x[i],space_y[j]])


class electron:
    def __init__(self,starting_position,starting_velocity,number_of_iterations,sign):
        self.starting_position    = starting_position
        self.starting_velocity    = starting_velocity
        self.number_of_iterations = number_of_iterations
        self.list_of_p            = []
        self.list_of_r            = []
        self.sign                 = sign

    def force(self,position):
        return [intensity_parameter*2*position[0]/((1+position[0]**2+position[1]**2)**2),
                intensity_parameter*2*position[1]/((1+position[0]**2+position[1]**2)**2)]

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
            flip   = 0
            self.source = 0
            self.drain  = 0
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
                    self.list_of_p.append([self.list_of_p[N-1][0]+          (self.step_time/6.0)*(kn1[0][0]+2*kn2[0][0]+2*kn3[0][0]+kn4[0][0]),
                                           self.list_of_p[N-1][1]+self.sign*(self.step_time/6.0)*(kn1[0][1]+2*kn2[0][1]+2*kn3[0][1]+kn4[0][1])])

                    self.list_of_r.append([self.list_of_r[N-1][0]+(self.step_time/6.0)*(ln1[0][0]+2*ln2[0][0]+2*ln3[0][0]+ln4[0][0]),
                                           self.list_of_r[N-1][1]+(self.step_time/6.0)*(ln1[0][1]+2*ln2[0][1]+2*ln3[0][1]+ln4[0][1])])
                '''
                This is where the "flip" of the band will happen, and thus, we will change the self.sign from +1 to -1 and then -1 to +1.
                '''
                if   potential(self.list_of_r[N]) > np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2) and self.sign ==+1 and flip <1:
                    flip +=1
                    self.sign = -1
                elif potential(self.list_of_r[N]) < np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2) and self.sign ==-1 and flip <2:
                    flip +=1
                    self.sign = +1
                else:
                    pass

                '''
                In order to compute the classical transmission, we need to enclose our system in a box,
                here the box is delimited inbetween x from [-6 to +10] and y from [-10 to +10].
                In order to make the specular reflection (mirror-like) we just have to flip the sign of the impulsion p at the interface.
                '''

                if np.abs(self.list_of_r[N][1]) > y_top_limit:                                           #Reflection on top and bottom of the box
                    self.list_of_p[N][1] = -self.list_of_p[N][1]
                elif np.abs(self.list_of_r[N][1]) > captation_sz and self.list_of_r[N][0] > x_rgt_limit: #Reflection on the right of the box and outside of the drain captation
                    self.list_of_p[N][0] = -self.list_of_p[N][0]
                elif np.abs(self.list_of_r[N][1]) > captation_sz and self.list_of_r[N][0] < x_lft_limit: #Reflection on the left of the box and outisde of the source captation
                    self.list_of_p[N][0] = -self.list_of_p[N][0]
                else:
                    pass



                if   self.list_of_r[N][0] > x_rgt_limit and np.abs(self.list_of_r[N][1]) < captation_sz: #Drain captation
                    self.drain = 1
                elif self.list_of_r[N][0] < x_lft_limit and np.abs(self.list_of_r[N][1]) < captation_sz: #Source captation
                    self.source = 1
                else :
                    pass

                if self.drain == 1 or self.source ==1 : #This is so when the electron is either in the drain or the source, it stop the computation
                    break


        else :
            print('Not implemented yet')

if __name__ == '__main__':
    x_lft_limit         = -6       #left limit of the box (x axis)
    x_rgt_limit         = +10      #right limit of the box (x axis)
    y_top_limit         = +10      #top limit of the box (y axis)
    y_bot_limit         = -10      #bottom limit of the box (y axis)
    captation_sz        = 2.5      #Captation size
    start_countdown     = 0        #Show % of completion on the terminal
    number_drain        = 0
    number_source       = 0
    sum_transmission    = 0
    number_electrons    = 300      #Number of electrons in the simulation
    start_angle         = -np.pi/2 #First angle of diffusion
    end_angle           = +np.pi/2 #Last angle of diffusion
    for j in np.linspace(start_angle,end_angle,number_electrons):
        if j==0:
            pass
        else:
            start_countdown +=1
            print(' '+str(100*start_countdown/number_electrons)+'%', end="\r") #the end="\r" allows to only print one % and delete afterwards

            electron1 = electron([-6,0],[np.cos(j),np.sin(j)],200000,+1)
            electron1.numerical_scheme('RK4')

            if electron1.drain == 1:
                number_drain     += 1
                sum_transmission += np.cos(j) * (end_angle-start_angle)/number_electrons
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='red'  ,alpha=0.1)
            elif electron1.source == 1:
                number_source    += 1
                sum_transmission += 0
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='green',alpha=0.1)
            else:
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='blue' ,alpha=0.1)
    print(' ')
    print('In the end, there is about '+str(100*number_drain/(number_drain+number_source))+'% of electrons back in the drain')
    print('The transmission is about '+str(sum_transmission))
    print(' ')
    plt.contour(space_x,space_y,space_z, origin='lower', cmap=cm.Reds, levels=30)
    plt.colorbar()

    plt.fill_between([x_rgt_limit-0.2,x_rgt_limit], [+captation_sz,+captation_sz], color='red'                 )
    plt.fill_between([x_rgt_limit-0.2,x_rgt_limit], [-captation_sz,-captation_sz], color='red'  ,label='Drain' )
    plt.fill_between([x_lft_limit,x_lft_limit+0.2], [+captation_sz,+captation_sz], color='green',label='Source')
    plt.fill_between([x_lft_limit,x_lft_limit+0.2], [-captation_sz,-captation_sz], color='green'               )
    plt.xlim([x_lft_limit, x_rgt_limit])
    plt.ylim([y_bot_limit, y_top_limit])
    plt.xlabel('$x$ axis')
    plt.ylabel('$y$ axis')
    plt.legend(loc='upper left')
    plt.savefig(str(np.random.rand())+'.png',dpi=600)
    plt.show()
