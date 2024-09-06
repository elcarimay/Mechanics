def Implicit(dx, x_max, dt, t_max, legend_plot = 1):
    x, t = np.arange(0, x_max+dx, dx).round(3), np.arange(0,t_max+dt, dt).round(3)
    bc = [0,0]; lx, lt = len(x), len(t)
    initial = np.sin(np.pi*x)
    T = np.zeros((len(t),len(x)))
    T[:,0], T[:,-1], T[0,:] = bc[0], bc[1], initial
    alpha = dt/dx**2
    mat = np.diag([1+2*alpha]*(lx-2),0) + np.diag([-alpha]*(lx-3),-1) + np.diag([-alpha]*(lx-3),1)
    for c in range(1,len(T)):
        b = T[c-1,1:-1]
        b[0], b[-1] = b[0] + alpha*T[c,1], b[-1] + alpha*T[c,-1]
        T[c,1:-1] = np.linalg.solve(mat, b)
    T.round(3)
    R, B, G = np.linspace(1,0,len(T)), np.linspace(0,1,len(T)), 0
    ext_sol = Exact(t,x)
    plt.figure(figsize=(6,4))
    for i in range(len(t)):
        plt.plot(x,ext_sol[i], color = [R[i], G, B[i]])
    for i in range(len(T)):
        plt.plot(x,T[i], 's',color = [R[i], G, B[i]])
    plt.title('$Points=%d,dt=%1.4f,dx=%1.3f$,Line=Exact.Sol.,Ret.:Cal.Value'%(len(x),dt,dx))
    plt.xlabel('distance [m]')
    plt.ylabel('Temperature [degree C]')
    if(legend_plot):
        plt.legend([f't = {value} s' for value in t])
    plt.grid()

from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def solve_tdm(x, coeffs, bc_0, bc_n):
    h = (x[2] - x[1])

    #[Step-1] TDM(i = 1, ..., n-1)
    a, b, c, d = map(np.asfarray, coeffs)
    diags = np.r_[a[1:-1], 0,0], b, np.r_[0, 0, c[1:-1]]
    A = spdiags(diags, [-1, 0, 1], x.size, x.size, format="lil")

    # [Step-2] BC (i=0): p0*u0 + q0*u0' = r0
    p0, q0, r0 = bc_0
    A[0, :3] = 2*h*p0 - 3*q0, 4*q0, -q0
    d[0] = 2*r0*h

    # [Step-3] BC (i=n) : pn*un + qn*un' = rn
    pn, qn, rn = bc_n
    A[-1, -3:] = qn, -4*qn, 2*pn*h + 3*qn
    d[-1] = 2*rn*h

    return spsolve(A.tocsr(), d)
L = 0.02 # m
k = 10 # W/m K
pc = 1e7 # J/m^3 K
nx = 5
dx = 0.004
x = np.r_[0,np.linspace(dx/2,L-dx/2,nx),L]
t = np.r_[40,80,120]
dt = 2

def Exact_8_1(x, t, repeat = 100):
    alpha = k/pc
    sum_value = 0
    for i in range(1,repeat):
        lambda_n = (2*i-1)*np.pi/2/L
        sum_value += (-1)**(i+1)/(2*i-1)*np.exp(-alpha*lambda_n**2*t)*np.cos(lambda_n*x)
    return 4/np.pi*sum_value*200

T = np.zeros((len(t),len(x)))
TB = 0

def aW(n):
    if n == 1: return 0
    else: return k/dx
def aE(n):
    if n == 5: return 0
    else: return k/dx
def Su(n,Tp0):
    if n == 5: return 2*k/dx*(TB-Tp0)
    else: return 0

initial = 200
T = x.copy()
nt = 61
for i in range(nt):
    T = np.c_[T,np.r_[0, np.linspace(dx/2, L-dx/2, nx), L]]
T = T.T
T[0,:] = initial
dx = x[2]-x[1]
aP = pc*dx/dt
for r in range(1,nt):
    for c in range(1,nx+1):
        T[r,c] = (aW(c)*T[r-1,c-1]+aE(c)*T[r-1,c+1]+(aP-aW(c)-aE(c))*T[r-1,c]+Su(c,T[r-1,c]))/aP
        T[r,0] = T[r,1]
        T[r,-1] = 0

plt.figure(figsize=(6,4))
plt.plot(x,T[20],'s', label="Sim_40")
plt.plot(x,T[40],'^', label="Sim_80")
plt.plot(x,T[60],'o', label="Sim_120")
for i in t:
    plt.plot(x, Exact_8_1(x,i), label="Exact_%d s"%i)
plt.title('$Points=%d,dt=%1.4f,dx=%1.3f$'%(nx,dt,dx),fontsize = 15, fontweight = 'bold', fontfamily='Times New Roman')
plt.xlabel('Distance [m]',fontsize = 15, fontweight = 'bold', fontfamily='Times New Roman')
plt.ylabel('Temperature [$\degree C$]',fontsize = 15, fontweight = 'bold', fontfamily='Times New Roman')
plt.xlim([min(x), max(x)])
plt.ylim([0, 200])
plt.grid()
plt.legend()
plt.show()

nx = 5
dx = 0.004
x = np.r_[0,np.linspace(dx/2,L-dx/2,nx),L]
t = np.r_[40,80,120]
dt = 2

T = np.zeros((len(t),len(x)))
TB = 0

def aW(n):
    if n == 1: return 0
    else: return k/dx
def aE(n):
    if n == 5: return 0
    else: return k/dx
def Sp(n):
    if n == 5: return -2*k/dx
    else: return 0
def Su(n):
    if n == 5: return 2*k/dx*(TB)
    else: return 0

aP = pc*dx/dt
a, c, d = map(aW, range(1,nx+1)), map(aE, range(1,nx+1)), map(aP, range(1,nx+1))
b = np.zeros(nx)
for i in range(1, nx):
    b[i] = aW(i)+aE(i)+aP-Sp(i)

b
