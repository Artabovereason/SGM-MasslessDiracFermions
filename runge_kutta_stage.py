import numpy             as np
import matplotlib.pyplot as plt
import timeit
from matplotlib   import cm
from mpl_toolkits import mplot3d

start_time = timeit.default_timer()

intensity_parameter    = 1             #Intensity of the potential : u_t
fermi_energy_parameter = 1000                #meV
length_parameter       = 2.5               #micrometer
width_parameter        = length_parameter  #micrometer
A_parameter            = 1#fermi_energy_parameter*length_parameter*width_parameter
tip_size_parameter     = 1               #micrometer  2*10**(-2)
tip_position           = [0,1]

def potential(position,tip_position):
    return intensity_parameter*A_parameter/((tip_size_parameter**2)+(position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)
'''

def potential(position,tip_position):
    return intensity_parameter/(1+(position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)


def force(position,tip_position):
    return [intensity_parameter*2*(position[0]-tip_position[0])/((1+(position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)**2),
            intensity_parameter*2*(position[1]-tip_position[1])/((1+(position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)**2)]

'''
def force(position,tip_position):
    return [intensity_parameter*A_parameter*2*(position[0]-tip_position[0])/((tip_size_parameter**2+(((position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)**2))**2),
            intensity_parameter*A_parameter*2*(position[1]-tip_position[1])/((tip_size_parameter**2+(((position[0]-tip_position[0])**2+(position[1]-tip_position[1])**2)**2))**2)]


class electron:
    def __init__(self,starting_position,starting_velocity,number_of_iterations,sign,geometry,tip_position):
        self.starting_position    = starting_position
        self.starting_velocity    = starting_velocity
        self.number_of_iterations = number_of_iterations
        self.list_of_p            = []
        self.list_of_r            = []
        self.sign                 = sign
        self.geometry             = geometry
        self.tip_position         = tip_position

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
                    self.list_of_p.append([self.list_of_p[N-1][0]+self.step_time*(force(self.list_of_r[N-1],self.tip_position)[0]),
                                           self.list_of_p[N-1][1]+self.step_time*(force(self.list_of_r[N-1],self.tip_position)[1])])
                    self.list_of_r.append([self.list_of_r[N-1][0]+self.step_time*self.list_of_p[N-1][0]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2),
                                           self.list_of_r[N-1][1]+self.step_time*self.list_of_p[N-1][1]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2) ])

        elif self.method == "RK4":
            #print('Runge-Kutta 4')
            flip        = 0
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
                    kn1.append([force([self.list_of_r[N-1][0],self.list_of_r[N-1][1]],self.tip_position)[0],
                                force([self.list_of_r[N-1][0],self.list_of_r[N-1][1]],self.tip_position)[1]])
                    ln1.append([v0*self.list_of_p[N-1][0]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2),
                                v0*self.list_of_p[N-1][1]/np.sqrt(self.list_of_p[N-1][0]**2+self.list_of_p[N-1][1]**2)])
                    kn2.append([force([self.list_of_r[N-1][0]+self.step_time*ln1[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln1[0][1]/2.0],self.tip_position)[0],
                                force([self.list_of_r[N-1][0]+self.step_time*ln1[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln1[0][1]/2.0],self.tip_position)[1]])
                    ln2.append([v0*(self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)**2)),
                                v0*(self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn1[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn1[0][1]/2.0)**2))])
                    kn3.append([force([self.list_of_r[N-1][0]+self.step_time*ln2[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln2[0][1]/2.0],self.tip_position)[0],
                                force([self.list_of_r[N-1][0]+self.step_time*ln2[0][0]/2.0, self.list_of_r[N-1][1]+self.step_time*ln2[0][1]/2.0],self.tip_position)[1]])
                    ln3.append([v0*(self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)**2)),
                                v0*(self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn2[0][0]/2.0)**2+(self.list_of_p[N-1][1]+self.step_time*kn2[0][1]/2.0)**2))])
                    kn4.append([force([self.list_of_r[N-1][0]+self.step_time*ln3[0][0],self.list_of_r[N-1][1]+self.step_time*ln3[0][1]],self.tip_position)[0],
                                force([self.list_of_r[N-1][0]+self.step_time*ln3[0][0],self.list_of_r[N-1][1]+self.step_time*ln3[0][1]],self.tip_position)[1]])
                    ln4.append([v0*(self.list_of_p[N-1][0]+self.step_time*kn3[0][0])/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn3[0][0])**2+(self.list_of_p[N-1][1]+self.step_time*kn3[0][1])**2)),
                                v0*(self.list_of_p[N-1][1]+self.step_time*kn3[0][1])/(np.sqrt((self.list_of_p[N-1][0]+self.step_time*kn3[0][0])**2+(self.list_of_p[N-1][1]+self.step_time*kn3[0][1])**2))])
                    self.list_of_p.append([self.list_of_p[N-1][0]+          (self.step_time/6.0)*(kn1[0][0]+2*kn2[0][0]+2*kn3[0][0]+kn4[0][0]),
                                           self.list_of_p[N-1][1]+self.sign*(self.step_time/6.0)*(kn1[0][1]+2*kn2[0][1]+2*kn3[0][1]+kn4[0][1])])

                    self.list_of_r.append([self.list_of_r[N-1][0]+(self.step_time/6.0)*(ln1[0][0]+2*ln2[0][0]+2*ln3[0][0]+ln4[0][0]),
                                           self.list_of_r[N-1][1]+(self.step_time/6.0)*(ln1[0][1]+2*ln2[0][1]+2*ln3[0][1]+ln4[0][1])])
                '''
                This is where the "flip" of the band will happen, and thus, we will change the self.sign from +1 to -1 and then -1 to +1.
                '''
                if   potential(self.list_of_r[N],self.tip_position) > v0*np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2) and self.sign ==+1 and flip <1:
                    flip +=1
                    self.sign = -1

                elif potential(self.list_of_r[N],self.tip_position) < v0*np.sqrt(self.list_of_p[N][0]**2+self.list_of_p[N][1]**2) and self.sign ==-1 and flip <2:
                    flip +=1
                    self.sign = +1
                else:
                    pass

                if self.geometry == 'cubic':
                    '''
                    In order to compute the classical transmission, we need to enclose our system in a box.
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

                elif self.geometry == 'spheric':
                    '''
                    In order to compute the classical transmission, we need to enclose our system in a box.
                    In order to make the specular reflection (mirror-like) we just have to use Snell's law of reflection.
                    '''
                    if np.sqrt( ((self.list_of_r[N][0]-self.tip_position[0])**2)+(self.list_of_r[N][1]-self.tip_position[1])**2 ) >= y_top_limit and N !=0 :
                        n_x                  = self.list_of_r[N][0]/np.sqrt(self.list_of_r[N][0]**2+self.list_of_r[N][1]**2)
                        n_y                  = self.list_of_r[N][1]/np.sqrt(self.list_of_r[N][0]**2+self.list_of_r[N][1]**2)
                        self.list_of_p[N][0] = self.list_of_p[N][0]-2*(self.list_of_p[N][0]*n_x +self.list_of_p[N][1]*n_y)*n_x
                        self.list_of_p[N][1] = self.list_of_p[N][1]-2*(self.list_of_p[N][0]*n_x +self.list_of_p[N][1]*n_y)*n_y
                    else:
                        pass

                    if   np.sqrt( ((self.list_of_r[N][0]-self.tip_position[0])**2)+(self.list_of_r[N][1]-self.tip_position[1])**2 ) > y_top_limit and np.abs(self.list_of_r[N][1]) < captation_sz and self.list_of_r[N][0]>0:
                        self.drain  = 1
                    elif np.sqrt( ((self.list_of_r[N][0]-self.tip_position[0])**2)+(self.list_of_r[N][1]-self.tip_position[1])**2 ) > y_top_limit and np.abs(self.list_of_r[N][1]) < captation_sz and self.list_of_r[N][0]<0:
                        self.source = 1
                    else:
                        pass

                    if np.sqrt( ((self.list_of_r[N][0]-self.tip_position[0])**2)+(self.list_of_r[N][1]-self.tip_position[1])**2 )  > 1.1*y_top_limit: #limit very long time occuring
                        break

                    if self.drain == 1 or self.source ==1 : #This is so when the electron is either in the drain or the source, it stop the computation thus reducing the time of the computation
                        break
                else:
                    print('Not implemented yet')
        else :
            print('Not implemented yet')

if __name__ == '__main__':
    x_lft_limit         = -length_parameter      #Left limit of the box (x axis)
    x_rgt_limit         = +length_parameter      #Right limit of the box (x axis)
    y_top_limit         = +length_parameter      #Top limit of the box (y axis)
    y_bot_limit         = -length_parameter      #Bottom limit of the box (y axis)
    y_top_span          = +length_parameter/5.0  #
    y_bot_span          = -length_parameter/5.0  #
    captation_sz        = y_top_span             #Captation size
    start_countdown     = 0                      #Show % of completion on the terminal
    number_drain        = 0
    number_source       = 0
    sum_transmission    = 0
    number_electrons    = 10                     #Number of electrons in the simulation
    number_span         = 1
    start_angle         = -np.pi/2               #First angle of diffusion
    end_angle           = +np.pi/2               #Last angle of diffusion
    geometry            = 'spheric'              #it is either 'cubic' or 'spheric' at the moment, planning to do ellpisoid and rectangular as generalization of the two first mentionned here
    tip_position        = tip_position
    v0          = 1#10**6
    for w in np.linspace(y_bot_span,y_top_span,number_span):
        w=0
        sum_transmission_i = 0                                                               #intermediate scalar to sum after each theta_j and then sum_transmission is the sum for all y_w values
        for j in np.linspace(start_angle,end_angle,number_electrons):
            start_countdown +=1
            print(' '+str(100*start_countdown/(number_electrons*number_span))+'%', end="\r") #the end="\r" allows to only print one % and delete afterwards
            electron1 = electron([x_lft_limit,w],[v0*np.cos(j),v0*np.sin(j)],50000,+1,geometry,tip_position)
            electron1.numerical_scheme('RK4')
            if electron1.drain == 1:
                number_drain       += 1
                sum_transmission_i += np.cos(j) * (end_angle-start_angle)/number_electrons #integral into sums
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='red'  ,alpha=0.1)
            elif electron1.source == 1:
                number_source      += 1
                sum_transmission_i += 0                                                    #integral into sums
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='green',alpha=0.1)
            else:
                plt.plot([electron1.list_of_r[i][0] for i in range(len(electron1.list_of_r))],[electron1.list_of_r[i][1] for i in range(len(electron1.list_of_r))],color='blue' ,alpha=0.1)
        sum_transmission += sum_transmission_i*(y_top_span-y_bot_span)/number_span         #integral into sums

    if number_drain+number_source != 0 :
        print(' ')
        print(' ')
        print('In the end, there is about '+str(100*number_drain/(number_drain+number_source))+'% of electrons back in the drain')
        print('The transmission is about '+str(sum_transmission))
    else:
        pass

    stop_time = timeit.default_timer()
    print(' ')
    print('time of simulation (in s): ', stop_time - start_time)
    print(' ')
    space_x = np.linspace(x_lft_limit,x_rgt_limit,100)
    space_y = np.linspace(y_bot_limit,y_top_limit,100)
    space_z = np.zeros((len(space_x),len(space_y)))
    for i in range(len(space_x)):
        for j in range(len(space_y)):
            space_z[j][i] = potential([space_y[i],space_x[j]],tip_position)
    plt.contour(space_x,space_y,space_z, origin='lower', cmap=cm.Blues, levels=10,alpha=0.3)
    plt.colorbar()
    if geometry == 'spheric' :
        angle = np.linspace(0, 2*np.pi, 100)
        a     = y_top_limit*np.cos(angle)
        b     = y_top_limit*np.sin(angle)

        angle_captation = np.linspace(-np.arcsin(captation_sz/y_top_limit),+np.arcsin(captation_sz/y_top_limit),20)
        a_captation     = y_top_limit*np.cos(angle_captation)
        b_captation     = y_top_limit*np.sin(angle_captation)
        plt.plot(a           ,b,color='yellow'         ,alpha=1, linewidth=2)
        plt.plot(+a_captation,b_captation,color='red'  ,alpha=1, linewidth=2)
        plt.plot(-a_captation,b_captation,color='green',alpha=1, linewidth=2)
    if geometry == 'cubic' :
        plt.plot([x_lft_limit,x_rgt_limit],[y_bot_limit,y_bot_limit],color='yellow',alpha=1,linewidth=2)
        plt.plot([x_lft_limit,x_rgt_limit],[y_top_limit,y_top_limit],color='yellow',alpha=1,linewidth=2)
        plt.plot([x_lft_limit,x_lft_limit],[y_bot_limit,y_top_limit],color='yellow',alpha=1,linewidth=2)
        plt.plot([x_rgt_limit,x_rgt_limit],[y_bot_limit,y_top_limit],color='yellow',alpha=1,linewidth=2)
        plt.fill_between([x_rgt_limit+0.2,x_rgt_limit], [+captation_sz,+captation_sz], color='red'                 )
        plt.fill_between([x_rgt_limit+0.2,x_rgt_limit], [-captation_sz,-captation_sz], color='red'  ,label='Drain' )
        plt.fill_between([x_lft_limit,x_lft_limit-0.2], [+captation_sz,+captation_sz], color='green',label='Source')
        plt.fill_between([x_lft_limit,x_lft_limit-0.2], [-captation_sz,-captation_sz], color='green'               )
    plt.xlim([x_lft_limit-1, x_rgt_limit+1])
    plt.ylim([y_bot_limit-1, y_top_limit+1])
    plt.xlabel(r'$x[\mu m]$')
    plt.ylabel(r'$y[\mu m]$')
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    plt.savefig(str(np.random.rand())+'.png',dpi=600)
    plt.show()
