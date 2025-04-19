import numpy as np
import sympy as sp


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __radd__(self, v):
        return Vec2(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vec2(self.x - v.x, self.y - v.y)

    def __rsub__(self, v):
        return Vec2(v.x- self.x , v.y - self.y)

    def __mul__(self, n):
        return Vec2(self.x * n, self.y * n)

    def __rmul__(self, n):
        return Vec2(self.x * n, self.y * n)

    def dot(self, v):
        return self.x*v.x + self.y*v.y

    def get_length(self):
        return np.sqrt(self.dot(self) )


class Particle():
    # n = number of particles
    n = 0
    def __init__(self,initial_pos,initial_vel, mass):

        # i = particle index
        self.i = Particle.n
        Particle.n += 1

        self.m = mass
        self.G = 1  # change this to 6.67408 × 1e-11 if you want real world measuring units.
        
      
        self.pos = Vec2(sp.symbols("x_"+str(self.i)),sp.symbols("y_"+str(self.i)))
        self.vel = Vec2(sp.symbols("vx_"+str(self.i)),sp.symbols("vy_"+str(self.i)))
        self.acc = Vec2(0,0)
        
      
        self.lamb_vel = Vec2(None,None)
        self.lamd_acc = Vec2(None,None)
        
        
        self.initial_pos = initial_pos
        self.initial_vel = initial_vel
        
        
        self.vf_vel = Vec2(0,0)
        self.vf_acc = Vec2(0,0)
        
  
        self.sol_pos = Vec2(None,None)
        self.sol_vel = Vec2(None,None)
        

    def calculate_acc(self,particles):
        for j in range(len(particles)):
            if self.i !=j:
                self.acc += (particles[j].pos - self.pos)*particles[j].m*self.G*(1/(((self.pos.x-particles[j].pos.x)**2 + (self.pos.y-particles[j].pos.y)**2)**(3/2)))

   

    def lambdify_vel(self,particles):
        self.lamb_vel.x = sp.lambdify(self.vel.x, self.vel.x)
        self.lamb_vel.y = sp.lambdify(self.vel.y, self.vel.y)
   

    def lambdify_acc(self,particles):
        
        var = []
        for j in range(len(particles)):           
            var.append(particles[j].pos.x)
            var.append(particles[j].pos.y)
               
        self.lamd_acc.x = sp.lambdify([var], self.acc.x)
        self.lamd_acc.y = sp.lambdify([var], self.acc.y)





par = []



par.append(Particle(initial_pos = Vec2(5,2), initial_vel = Vec2(0.5,0.2) , mass = 1.))
par.append(Particle(initial_pos = Vec2(3,3), initial_vel = Vec2(0.1,0.5) , mass = 1.))
par.append(Particle(initial_pos = Vec2(0.6,2.5), initial_vel = Vec2(0.5,0.5) , mass = 1.))
par.append(Particle(initial_pos=Vec2(8, 8), initial_vel=Vec2(0.3, 0.7), mass=1.5))
par.append(Particle(initial_pos=Vec2(2, 9), initial_vel=Vec2(0.3, 0.7), mass=1.))

t_end = 70.0
steps = 800





n = len(par)



for i in range(n):
    par[i].calculate_acc(par)

for i in range(n):
    par[i].lambdify_vel(par)
    par[i].lambdify_acc(par)




import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def vectorfield(var, t):
    
    
    pos = var[0:2*n] 
    vel = var[2*n:4*n] 
    f = []
    
    for i in range(0,n):        
        par[i].vf_vel.x = par[i].lamb_vel.x(vel[2*i])
        par[i].vf_vel.y = par[i].lamb_vel.y(vel[2*i + 1])
        f.append(par[i].vf_vel.x)
        f.append(par[i].vf_vel.y)
        
    for i in range(0,n):        
        par[i].vf_acc.x = par[i].lamd_acc.x(pos)
        par[i].vf_acc.y = par[i].lamd_acc.y(pos)
        f.append(par[i].vf_acc.x)
        f.append(par[i].vf_acc.y)

    return f




from scipy.integrate import odeint



var = []
for i in range(len(par)):
    var.append(par[i].initial_pos.x)
    var.append(par[i].initial_pos.y)
    
for i in range(len(par)):
    var.append(par[i].initial_vel.x)
    var.append(par[i].initial_vel.y)




# ODE solver parameters


t = np.linspace(0,t_end,steps+1)

sol = odeint(vectorfield, var, t)
sol = np.transpose(sol)

# order the solution for clarity

for i in range(n):
    par[i].sol_pos.x = sol[2*i]
    par[i].sol_pos.y = sol[2*i+1]
    
for i in range(n):
    par[i].sol_vel.x = sol[2*n + 2*i]
    par[i].sol_vel.y = sol[2*n + 2*i+1]
    


Energy = 0 
for i in range(0,n):
    for j in range(i+1,n):
        Energy += (-1/(((par[i].sol_pos.x-par[j].sol_pos.x)**2 + (par[i].sol_pos.y-par[j].sol_pos.y)**2)**(1/2)))


for i in range(0,n):
    Energy += 0.5*(par[i].sol_vel.x*par[i].sol_vel.x + par[i].sol_vel.y*par[i].sol_vel.y)

plt.style.use('dark_background')
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1,1,1)

plt.subplots_adjust(bottom=0.2, left=0.15)

ax.axis('equal')
ax.axis([-1, 30, -1, 30])
ax.set_title('Energy =' + str(Energy[0]))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Add timer text
timer_text = ax.text(0.02, 0.95, 'Time: 0.00', transform=ax.transAxes, 
                     color='white', fontsize=12)

circle = [None]*n
line = [None]*n
for i in range(n):
    circle[i] = plt.Circle((par[i].sol_pos.x[0], par[i].sol_pos.y[0]), 0.08, ec="w", lw=2.5, zorder=20)
    ax.add_patch(circle[i])
    line[i] = ax.plot(par[i].sol_pos.x[:0], par[i].sol_pos.y[:0])[0]


play_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
play_button = plt.Button(play_ax, '▶ Play', color='#5c05ff')

class Animator:
    def __init__(self):
        self.anim_running = True
        self.current_frame = 0
        self.animation = None
        
    def play_pause(self, event):
        if self.anim_running:
            self.animation.event_source.stop()
            play_button.label.set_text('▶ Play')
        else:
            self.animation.event_source.start()
            play_button.label.set_text('⏸ Pause')
        self.anim_running = not self.anim_running
    
    def update(self, frame):
        self.current_frame = frame
        timer_text.set_text(f'Time: {t[frame]:.2f}')
        ax.set_title(f'Energy = {Energy[frame]:.2f}')
        for j in range(n):
            circle[j].center = par[j].sol_pos.x[frame], par[j].sol_pos.y[frame]
            line[j].set_xdata(par[j].sol_pos.x[:frame+1])
            line[j].set_ydata(par[j].sol_pos.y[:frame+1])
        return circle + line + [timer_text]

animator = Animator()
play_button.on_clicked(animator.play_pause)

# Create animation
from matplotlib.animation import FuncAnimation
animator.animation = FuncAnimation(fig, animator.update, frames=len(t),
                                 interval=50, blit=False, repeat=True)

plt.show()
