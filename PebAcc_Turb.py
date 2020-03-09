import funcs as fn
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

def d_V_12(t_1=1.,t_2=1.,t_L=1,Re=1e8):
    """Function for reproducing Eqn. (16) in Ormel and Cuzzi (2007). """
    t_eta = Re**(-0.5)*t_L
    if (t_1 <= t_eta) and (t_2 <= t_eta):
        return np.sqrt(t_L / t_eta * (t_1 - t_2)**2)
    elif (t_1 > t_eta) and (t_1 < t_L):
        y_a = 1.6
        eps = t_1 / t_2
        return np.sqrt(2 * y_a - (1 + eps) + 2/(1 + eps) * (1/(1 + y_a) + eps**3/(y_a + eps))) * np.sqrt(t_1)
    elif (t_1 >= t_L):
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))
    else:
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))

def ts_sto_ram(r,params=[]):
    vth,mfp,rho_g,v_pg_tot,rho_s,tmp = params

    re = 4*r*v_pg_tot/vth/mfp
    cd = fn.drag_c(re)
    fd = 1./2.*cd*np.pi*r**2*rho_g*v_pg_tot**2

    t_s = 4./3.*np.pi*r**3*rho_s*v_pg_tot/fd

    return t_s

def st_solver(st=1,alph = 1e-100,a_au=1,m_suns=1,m_earths=1,verbose=0,smooth=1,gas_dep=1.,sig_g_in = 0, temp_in = 0):

    if verbose:
        print("St = %.7g" %st)
        print("a = %.3g AU" %a_au)
        print("m_star = %.3g m_sun" %m_suns)
        print("m_core = %.3g m_earths" %m_earths)

    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths

    rho_obj = 2.0

    if sig_g_in:
        sig = sig_g_in
    else:
        sig = fn.surf_dens(a_core)

    if temp_in:
        t = temp_in
    else:
        t =  fn.temp(a_core)

    #Return derived parameters
    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(t)
    h = cs/om
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    vkep = fn.vkep(m_star,a_core)
    eta = fn.eta(cs,vkep)


    h_r = fn.hill_rad(m_core,a_core,m_star)
    b_r = fn.bondi_rad(m_core,cs)

    if verbose:
        print("sig = %s" %sig)
        print("T = %s" %t)
        print("om = %s" %om)
        print("cs = %s" %cs)
        print("H = %s" %h)
        print("rho_g = %s" %rho_g)
        print("mfp = %s" %mfp)
        print("vth = %s" %vth)

    r_eps = st*rho_g/rho_obj*vth/om

    if verbose:
        print("r_eps = %.5g" %r_eps)

    if r_eps < 9.*mfp/4.:
        if verbose:
            print("Epstein Regime")
        return r_eps
    elif verbose:
        print("Not in Epstein")


    #Directly calculate v_rel from given St
    t_s = st/om
    if verbose:
        print("t_s = %.7g" %t_s)

    re_f = fn.re_f(alph,cs,h,vth,mfp)
    v_turb = np.sqrt(alph)*cs
    v_gas_tot = np.sqrt(v_turb**2 + (eta*vkep)**2)

    v_pg_L = eta*vkep*st*np.sqrt(4.+st**2.)/(1+st**2.)
    t_eddy = om**(-1)/np.sqrt(1+v_pg_L**2/v_turb**2)

    stl = t_s/t_eddy
    v_pg_turb = fn.v_pg(v_turb,stl,re_f)

    v_pg_tot = np.sqrt(v_pg_turb**2 + v_pg_L**2)

    if verbose:
        print("Re_t = %.7g" %re_f)
        print("t_eddy = %.7g" %(t_eddy*om))
        print("st_L = %.7g" %stl)
        print("v_turb = %.7g" %v_turb)
        print("v_gas_tot = %.7g" %v_gas_tot)
        print("v_pg_L = %.7g" %v_pg_L)
        print("v_pg_turb = %.7g" %v_pg_turb)
        print("v_pg_tot = %.7g" %v_pg_tot)

    def ts_zero(r,params):
        t_s = params[-1]
        return ts_sto_ram(r,params)-t_s

    params = [vth,mfp,rho_g,v_pg_tot,rho_obj,t_s]
    r_sto_ram = fsolve(ts_zero,9.*mfp/4.,args=params)[0]
    return r_sto_ram

class Disk:
    """ This class contains all of the properties of the protoplanetary disk. """
    def __init__(self, alpha=1e-100, a_core_au=1, m_star_solarmasses=1, gas_dep=1., sol_gas_ratio=0.01,
                 sig_p_in=0, sig_g_in=0, temp_in=0):

        self.m_star = m_star_solarmasses * fn.m_sun # Defines mass of star
        self.a_core = a_core_au * fn.au # Converts semi-major axis into cgs units

        ### Set disk parameters ###
        if sig_g_in: self.sig_gas = sig_g_in
        else: self.sig_gas = fn.surf_dens(self.a_core)/gas_dep # Calculates gas surface density
        if sig_p_in: self.sig_solid = sig_p_in # Calculates solid surface density
        else: self.sig_solid = self.sig_gas * sol_gas_ratio
        
        # Calculates temperature of disk
        if temp_in: self.T = temp_in
        else: self.T = fn.temp(self.a_core)

        self.alpha = alpha
        self.om = fn.omega(self.m_star, self.a_core) # Orbital frequency
        self.cs = fn.sound_speed(self.T) # Sound speed
        self.H = self.cs/self.om # Gas Scale height
        self.rho_gas = fn.gas_density(self.sig_gas, self.H) #Volumetric Density of gas
        self.mfp = fn.mean_free_path(fn.mu, self.rho_gas, fn.cross) # Mean free path
        self.v_th = fn.therm_vel(self.cs) # Thermal velocity
        self.v_kep = fn.vkep(self.m_star, self.a_core) # Keplerian velocity of core
        self.eta = fn.eta(self.cs, self.v_kep) # η, a measure of the local gas pressure support
        self.v_rel_i = self.v_kep # Sets initial velocity of core relative to gas

        # Turbulent gas parameters, see e.g. Ormel and Cuzzi (2007)
        self.re_f = fn.re_f(alpha, self.cs, self.H, self.v_th, self.mfp) # Reynolds number, MAYBE?
        self.v_gas_turb = (alpha)**(1/2)*self.cs # Turbulent velocity of gas
        self.v_gas_tot = np.sqrt(self.v_gas_turb**2 + (self.eta * self.v_kep)**2) # Total velocity of gas

class Core(Disk):
    """ This class contains all of the properties of an accreting core. """
    def __init__(self, a_core_au=1, m_core_earthmasses=1, alpha=1e-100, m_star_solarmasses=1, gas_dep=1.,
                 sol_gas_ratio=0.01, rho_core=2., sig_p_in=0, sig_g_in=0, temp_in=0, r_shear_off=0, extend_rh=1, alpha_z=0, h_mod=1):
        super().__init__(alpha, a_core_au, m_star_solarmasses, gas_dep, sol_gas_ratio, sig_p_in, sig_g_in, temp_in)
        self.r_shear_off = r_shear_off
        self.extend_rh = extend_rh
        self.alpha_z = alpha_z
        self.h_mod = h_mod

        self.m_core = m_core_earthmasses * fn.m_earth # Converts core mass into cgs units
        self.r_core = ((3 * self.m_core)/(4 * np.pi * rho_core))**(1/3) # Calculates radius of core
        
        ### Set core parameters ###
        self.r_hill = fn.hill_rad(self.m_core, self.a_core, self.m_star) # Hill radius
        self.r_bondi = fn.bondi_rad(self.m_core, self.cs) # Bondi radius
        self.r_bondi = max(self.r_bondi, self.r_core) # Prevents Bondi radius from shrinking below core radius
        self.rho_core = rho_core


    def t_stop(self, s, rho_obj=2.):
        """ Calculates stopping time, given particle size s. """
        if s > 9./4 * self.mfp: # Solve iteratively if we're in the fluid regime
            # Calculate terminal velocity and stopping time by iterating over force law. We've ignored angle here.
            delta = 1 # Used to check if we've converged
            v_i = self.v_rel_i
            while np.abs(delta) > .001:
                re = fn.rey(s, v_i, self.v_th, self.mfp) # Reynolds number
                m_obj = fn.obj_mass(s, rho_obj) # Mass of accreting objects
                dc = fn.drag_c(re) # Drag coefficient
                f_d = fn.stokes_ram(s, dc, self.rho_gas, v_i) # Drag force (ram pressure regime)
                t_s = m_obj*v_i/f_d # Stopping time
                v_r_lam = fn.v_new_r(self.eta, self.v_kep, t_s, self.om) # Radial component of laminar velocity (particle rel. to gas)
                v_phi_lam = fn.v_new_phi(self.eta, self.v_kep, t_s, self.om) # Phi component of laminar velocity
                v_lam = np.sqrt(v_r_lam**2 + v_phi_lam**2) # Total laminar velocity
                t_eddy = (self.om * (1 + (v_lam/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
                stl = t_s/t_eddy # Stokes number
                if stl > 10:
                    v_new_turb = self.v_gas_turb * np.sqrt(1 - (1 + stl)**-1)
                else:
                    v_new_turb = fn.v_pg(self.v_gas_turb, stl, self.re_f)
                v_new = np.sqrt(v_lam**2 + v_new_turb**2) # Calculates total velocity, adding laminar and turbulent
                delta = (v_new - v_i)/v_i 
                v_i = v_new
            t_s = t_s
        else: # Applies Epstein drag law if in diffuse regime
            t_s = fn.ts_eps(rho_obj, self.rho_gas, s, self.v_th)

        # Calculate laminar velocities from converged stopping time.
        v_r_lam = fn.v_new_r(self.eta, self.v_kep, t_s, self.om) # Radial component
        v_phi_lam = fn.v_new_phi(self.eta, self.v_kep, t_s, self.om) # Phi component
        v_lam = np.sqrt(v_r_lam**2 + v_phi_lam**2) # Total velocity

        t_eddy = (self.om * (1 + (v_lam/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
        stl = t_s/t_eddy
        tau_s = t_s * self.om # Dimensionless stopping time

        if stl > 10: # Avoids using OC07 expressions for large stopping time
            v_obj_gas_turb = self.v_gas_turb * np.sqrt(1 - (1 + stl)**-1)
        else:
            v_obj_gas_turb = fn.v_pg(self.v_gas_turb, stl, self.re_f)
        v_obj_gas = np.sqrt(v_lam**2 + v_obj_gas_turb**2)
        return np.array([t_s, v_r_lam, v_phi_lam, v_lam, t_eddy, stl, tau_s, v_obj_gas_turb, v_obj_gas])


    def drag_force(self, s, vrel_i, rho_g, mfp, vth):
        re = fn.rey(s, vrel_i, vth, mfp)
        dc = fn.drag_c(re)
        if s > 9.*mfp/4.:
            fd = fn.stokes_ram(s,dc,rho_g,vrel_i)
        else:
            fd = fn.epstein_drag(s,rho_g,vrel_i,vth)
        return fd


    def r_wish(self, s, rho_obj=2.):
        """ Calculates wind-shearing radius, given particle size s. """
        v_cap = self.v_gas_tot # Set relevant velocity for orbit capture
        self.f_drag_obj = self.drag_force(s, v_cap, self.rho_gas, self.mfp, self.v_th)
        # Drag force on core
        self.f_drag_core = self.drag_force(self.r_core, self.v_core_gas, self.rho_gas, self.mfp, self.v_th)

        self.m_obj = (4./3.*np.pi*s**3)*rho_obj # Calculates mass of accreted object
        self.delta_a = np.abs(self.f_drag_obj/self.m_obj - self.f_drag_core/self.m_core) # Differential acceleration between core and object

        self.r_ws = fn.wish_radius(self.delta_a, self.m_core, self.m_obj) # Wind shearing radius
        return self.r_ws


    def r_shear(self, s, rho_obj=2.):
        """ Calculates shearing radius, given particle size s. """
        self.m_obj = (4./3.*np.pi*s**3)*rho_obj # Calculates mass of accreted object

        if self.r_shear_off: r_shear = 1e100 # Checks for flag
        elif s > 9./4 * self.mfp:
            def r_shear_solver(r_shear): # Function to pass to f_solve to determine r_shear
                v_rel = r_shear * self.om
                re = fn.rey(s, v_rel, self.v_th, self.mfp)
                dc = fn.drag_c(re)
                f_drag = fn.drag_force(s, v_rel, dc, self.rho_gas, self.mfp, self.v_th)
                return r_shear - np.sqrt(fn.G * self.m_core * self.m_obj/f_drag)

            # Solution for r_shear in the ram regime
            self.r_shear_ram = (fn.G * self.m_core * self.m_obj/
                                (.5 * .47 * self.rho_gas * np.pi * s**2. * self.om**2))**(1./4)
            self.r_shear_an = (3. * self.tau_s)**(1./3) * self.r_hill
            # Guess is the minimum of the analtyic solution in a linear regime and the solution in Ram
            r_shear_guess = min(self.r_shear_ram, self.r_shear_an)
            self.r_sh = fsolve(r_shear_solver, r_shear_guess)[0]
        else:
            self.r_sh = 3.**(1./3) * self.r_hill * self.tau_s**(1./3)
        return self.r_sh


    def r_accretion(self):
        """ Calculates accretion radius. """
        min_1 = np.minimum(self.r_ws, self.r_sh)
        self.r_stab = np.minimum(min_1, self.r_hill)

        self.r_atm = np.minimum(self.r_bondi, self.r_hill) # Calculates atmospheric radius as minimum of shearing and hill radii. Mickey currently has this coded as maximum, but should be minimum
        self.r_acc = np.maximum(self.r_atm, self.r_stab) # Calculates Accretion radius as maximum of bondi radius and stability radius
        self.r_acc = np.maximum(self.r_acc, self.r_core) # Make sure that the accretion radius doesn't get below the physical radius
        return self.r_acc


    def set_velocities(self, disp=0):
        """ Calculates a bunch of velocities, given particle size s. """
        self.v_cross = fn.vkep(self.m_core, self.r_acc) # Orbit velocity about core

        self.v_obj_phi = -self.eta * self.v_kep * (1/(1 + self.tau_s**2)) # Object velocity relative to Keplerian #np.abs(self.v_phi_lam + self.v_gas_lam)
        self.v_core_phi = -self.eta * self.v_kep * (1/(1 + self.tau_s_core**2)) # Core velocity relative to Keplerian #np.abs(self.v_phi_lam_core + self.v_gas_lam)

        self.v_lam_iner = np.sqrt((self.v_obj_phi - self.v_core_phi)**2 + self.v_r_lam**2) # Laminar, relative to inertial frame

        self.v_obj_core_lam = np.sqrt((self.v_obj_phi - self.v_core_phi)**2 + (self.v_r_lam - self.v_r_lam_core)**2) # Velocity of object relative to core
        self.v_obj_core_turb  = self.v_gas_turb * d_V_12(t_1=self.stl_core,t_2=self.stl,t_L=1,Re=self.re_f) # Use expresioons from Ormel Cuzzi 2007 to get relative turbulent particle velocity from stopping time of core #np.sqrt(self.v_gas_turb**2 - self.v_core_gas_turb**2)
        self.v_obj_core = np.sqrt(self.v_obj_core_lam**2 + self.v_obj_core_turb**2)
        
        self.v_shear = self.r_acc * self.om # Shear velocity

        self.v_inf = np.maximum(self.v_obj_core, self.v_shear) # Sets v_infinity
        self.v_enc = fn.G * self.m_core/self.r_acc/self.v_inf # Applies impulse approx to calculate encounter velocity

        if self.v_inf > self.v_cross: # Checks if impulse approximation is OK
            self.v_gas_enc = max(self.v_enc, self.v_obj_gas)
            self.v_grav = self.v_enc
        else:
            self.v_gas_enc = max(self.v_cross, self.v_obj_gas)
            self.v_grav = self.v_cross


    def encounter(self, s, rho_obj=2.):
        """ Calculate drag force, work, and kinetic energy during encounter, given particle size s. 
            Also calculates the accretion probability. """
        self.re_enc = fn.rey(s, self.v_gas_enc, self.v_th, self.mfp) # Reynolds number during encounter
        self.dc_enc = fn.drag_c(self.re_enc) # Drag coefficient during encounter
        self.f_drag_enc = fn.drag_force(s, self.v_gas_enc, self.dc_enc, self.rho_gas, self.mfp, self.v_th) # Drag force
        self.work_enc = 2 * self.f_drag_enc * self.r_acc # Work done by drag over course of encounter
        self.ke = .5 * self.m_obj * self.v_inf**2 # Kinetic energy of object during encounter

        # Modify growth time by the ratio of the kinetic energy to work done over one orbit
        if (self.r_acc == self.r_hill and self.v_inf == self.v_shear and self.extend_rh):
            self.prob = min(self.work_enc/self.ke, 1.)
        else:
            self.prob = 1.


    def scale_heights(self):
        """ Calculate the turbulent scale height, Kelvin-Helmholtz scale height, disk scale height,
            accretion height, and accretion area. """
        if not(self.alpha_z): # Checks if vertical turbulence is different from other directions
            self.alpha_z = self.alpha
        self.H_turb = min(np.sqrt(self.alpha_z/self.tau_s) * self.H, self.H) # Turbulent scale height
        # Kelvin-Helmholtz scale height
        self.H_KH = fn.H_KH(self.H, self.a_core) * self.h_mod * min(1., 1/np.sqrt(self.tau_s))
        self.H_disk = np.maximum(self.H_turb, self.H_KH) # Scale height of disk, also called h_p

        self.H_acc = np.minimum(self.H_disk, self.r_acc) # Accretion height
        self.area_acc = 4 * self.r_acc * self.H_acc # Area over which objects are accreted


    def t_accretion(self):
        """ Calculate the accretion/growth time for an object of size s. """
        self.t_acc = (fn.growth_time(self.m_core, self.H_disk, self.sig_solid, self.area_acc, self.v_inf)
                      * fn.sec_to_years/self.prob)

        # Check energy criterion for accretion
        if (((self.r_stab > self.r_bondi and self.work_enc < self.ke) or (self.r_stab < self.r_bondi and
              self.work_enc > self.ke)) and not
           (self.r_stab == self.r_hill and self.v_inf == self.v_shear and self.extend_rh and self.work_enc < self.ke)):
            self.t_acc = 0 # Really should be infinity, but set to 0 for easy plotting
        return self.t_acc
 

    def main(self, s, rho_obj=2., disp=0):
        """ Runs each method defined for this class, in order to calculate and set all of the attributes
            of the object. """
        ts_arr = self.t_stop(s, rho_obj=rho_obj)
        self.t_s, self.v_r_lam, self.v_phi_lam, self.v_lam, self.t_eddy, self.stl, self.tau_s, self.v_obj_gas_turb, self.v_obj_gas = ts_arr
        ts_c_arr = self.t_stop(self.r_core, self.rho_core)
        self.t_s_core, self.v_r_lam_core, self.v_phi_lam_core, self.v_lam_core, self.t_eddy_core, self.stl_core, self.tau_s_core, self.v_core_gas_turb, self.v_core_gas = ts_c_arr
        r_ws = self.r_wish(s, rho_obj=rho_obj)
        r_sh = self.r_shear(s, rho_obj=rho_obj)
        r_acc = self.r_accretion()
        self.set_velocities(disp=disp)
        self.encounter(s, rho_obj=rho_obj)
        self.scale_heights()
        t_acc = self.t_accretion()