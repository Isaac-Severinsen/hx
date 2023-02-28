
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import numpy as np
import copy
import json
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numba import jit
import time

import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from phasepy import component, preos
from phasepy.equilibrium import flash

import ht
import fluids

import json
from collections import namedtuple

class ImaginaryError(Exception):
    pass

class TimeError(Exception):
    pass

def jsondecoder(dictt):
    return namedtuple('X', dictt.keys())(*dictt.values())

class stream:
    """
    A class defining a fluid stream.
    This defines the thermodynamic properties of the stream.
    """

    def __init__(self,input_str, A, B, x_dict):

        comp_list = list(x_dict.keys())
        comp_frac_list = [x_dict[key] for key in comp_list]
        
        self.instantiate(comp_list, comp_frac_list)

        self.update_state_props(input_str, A, B)
        self.update_thermo_props()
        
    def instantiate(self, comp_list, comp_frac_list):
        self.comp_list = comp_list
        self.comp_frac_list = comp_frac_list
        self.RP = CoolProp.AbstractState('HEOS','&'.join(comp_list))
        # self.RP = CoolProp.AbstractState('PR','&'.join(comp_list))
        # self.RP = CoolProp.AbstractState('REFPROP','&'.join(comp_list))
        self.RP.set_mole_fractions(comp_frac_list)
        self.Tc  = self.RP.keyed_output(CoolProp.iT_critical)
        self.Pc  = self.RP.keyed_output(CoolProp.iP_critical)
        self.Mw  = self.RP.keyed_output(CoolProp.imolar_mass) * 1000 # convert kg/mol -> g/mol
        
    def update(self,input_str, A, B):
        self.update_state_props(input_str, A, B)
        self.update_thermo_props()
        
    def copy(self):
        return copy.copy(self)

    def update_state_props(self, input_str, A, B):

        if input_str == 'TP':
            self.T = A # [K]
            self.P = B # [bar]
            self.RP.update(CoolProp.PT_INPUTS, self.P, self.T)
            self.Q = self.RP.keyed_output(CoolProp.iQ)
            self.H   = self.RP.keyed_output(CoolProp.iHmass) # [J/kg]
            self.S   = self.RP.keyed_output(CoolProp.iSmolar)

        elif input_str =='TQ':
            self.T = A # [K]
            self.Q = B # [-]
            if self.T > self.Tc:
                print('Temperature exceeds critical temperature')
            self.RP.update(CoolProp.QT_INPUTS, self.Q, self.T)
            self.P   = self.RP.keyed_output(CoolProp.iP) # [bar]
            self.H   = self.RP.keyed_output(CoolProp.iHmass) # [J/kg]
            self.S   = self.RP.keyed_output(CoolProp.iSmolar)

        elif input_str =='PQ':
            self.P = A # [bar]
            self.Q = B # []
            if self.P > self.Pc:
                print('Pressure exceeds critical pressure')
            self.RP.update(CoolProp.PQ_INPUTS, self.P, self.Q)
            self.T   = self.RP.keyed_output(CoolProp.iT) # [K]
            self.H   = self.RP.keyed_output(CoolProp.iHmass) # [J/kg]
            self.S   = self.RP.keyed_output(CoolProp.iSmolar)
            
        elif input_str == 'HP':
            self.H = A # [J/kg]
            self.P = B # [bar]
            self.RP.update(CoolProp.HmassP_INPUTS, self.H, self.P)
            self.Q = self.RP.keyed_output(CoolProp.iQ)
            self.T = self.RP.keyed_output(CoolProp.iT) # [K]
            self.S = self.RP.keyed_output(CoolProp.iSmolar)
            
        elif input_str == 'PS':
            self.P = A # [bar]
            self.S = B # [J/kg]
            self.RP.update(CoolProp.PSmolar_INPUTS, self.P, self.S)
            self.Q = self.RP.keyed_output(CoolProp.iQ)
            self.T = self.RP.keyed_output(CoolProp.iT) # [K]
            self.H = self.RP.keyed_output(CoolProp.iHmass) # [J/kg]

        try:
            self.phase = self.RP.keyed_output(CoolProp.iPhase)
        except:
            bubble = self.copy()
            bubble.update('PQ',bubble.P,0)
            dew = self.copy()
            dew.update('PQ',dew.P,1)

            if self.H < bubble.H: # Liq
                self.phase = 0
            elif self.H < dew.H: # 2P
                self.phase = 6
            elif self.P < self.Pc and self.P < self.Tc: # gas
                self.phase = 5
            else:
                self.phase = 1
        
        if self.Q > 1:
            self.Q = 1
        elif self.Q < 0:
            self.Q = 0
        

    def update_thermo_props(self):
        self.rho = self.RP.keyed_output(CoolProp.iDmass)
#         self.H   = self.RP.keyed_output(CoolProp.iHmass)

        self.cp  = self.RP.keyed_output(CoolProp.iCpmass)
        # self.cv  = self.RP.keyed_output(CoolProp.iCvmass)

        self.mu  = self.RP.keyed_output(CoolProp.iviscosity)

        if self.mu < 0:
            print('negative viscosity')
        self.k = self.RP.keyed_output(CoolProp.iconductivity)


        if self.phase == 6:
            self.sigma = self.RP.keyed_output(CoolProp.isurface_tension)

        if self.cp < 0:
            # if verbose:
            #     print('negative specific heat')
            #     print('original cp:', self.cp)
            liquid = self.copy()
            liquid.update('PQ', liquid.P, 0)

            gas = self.copy()
            gas.update('PQ', liquid.P, 1)

            self.cp = liquid.cp * (1-self.Q) + gas.cp * self.Q
        
class air_stream:
    """
    A class defining a humid air instance.
    This defines the thermodynamic properties of the stream.
    As this is humid air we need to use the CoolProp high level interface.
    """
    
    def __init__(self,T, P, RH):
        
        self.T = T #
        self.P = P #
        self.RH = RH#/100

        self.update('TP', T,P)

    def update(self, input_str, A, B):

        if input_str == 'TP':
            self.T = float(A) #+ 273.15 # [°C]
            self.P = float(B) #* 1e5    # [bar]

            # self.T_wb = HAPropsSI('T_wb' ,'T',self.T,'P',self.P,'R',self.RH) # [K]     
            self.cp   = HAPropsSI('cp_ha','T',self.T,'P',self.P,'R',self.RH) # [J/kg.K]  (humid air)   
            # self.cv   = HAPropsSI('cv_ha','T',self.T,'P',self.P,'R',self.RH) # [J/kg.K]  (humid air)    
            # self.T_dp = HAPropsSI('T_dp' ,'T',self.T,'P',self.P,'R',self.RH) # [K]    
            self.H    = HAPropsSI('Hha'  ,'T',self.T,'P',self.P,'R',self.RH) # [J/kg]  (humid air)     
            self.k    = HAPropsSI('k'    ,'T',self.T,'P',self.P,'R',self.RH) # [W/mK]  
            self.mu   = HAPropsSI('mu'   ,'T',self.T,'P',self.P,'R',self.RH) # [Pa.s]    
            self.rho  = HAPropsSI('Vha'  ,'T',self.T,'P',self.P,'R',self.RH)**(-1) # [kg/m³] (humid air)  
            
        elif input_str == 'HP':
            self.H = float(A) #+ 273.15 # [°C]
            self.P = float(B) #* 1e5    # [bar]

            self.T    = HAPropsSI('T_db' ,'H',self.H,'P',self.P,'R',self.RH) # [K]       
            # self.T_wb = HAPropsSI('T_wb' ,'H',self.H,'P',self.P,'R',self.RH) # [K]        
            self.cp   = HAPropsSI('cp_ha','H',self.H,'P',self.P,'R',self.RH) # [J/kg.K]  (humid air)   
            # self.cv   = HAPropsSI('cv_ha','H',self.H,'P',self.P,'R',self.RH) # [J/kg.K]  (humid air)   
            # self.T_dp = HAPropsSI('T_dp' ,'H',self.H,'P',self.P,'R',self.RH) # [K]   
            self.k    = HAPropsSI('k'    ,'H',self.H,'P',self.P,'R',self.RH) # [W/mK]   
            self.mu   = HAPropsSI('mu'   ,'H',self.H,'P',self.P,'R',self.RH) # [Pa.s]  
            self.rho  = HAPropsSI('Vha'  ,'H',self.H,'P',self.P,'R',self.RH)**(-1) # [kg/m³] (humid air)   

        self.phase = 5

    def copy(self):
        return copy.copy(self)

def NCG_flash(NCG_content, T, P, eos, verbose = True): # check T, P units

    P = P /1e5 # convert pressure to bar

    Z = np.array([1-NCG_content, NCG_content])  # overall composition
    x0 = np.array([0.99, 0.01]) # liquid phase compositions guess
    y0 = np.array([0.5, 0.5]) # vapour phase compositions guess

    liq_phase_comp, gas_phase_comp, Q = flash(x0, y0, 'LV', Z, T, P, eos)

    water_vapour_content = gas_phase_comp[0]

    if verbose:
        print('condensate vapour fraction', round(Q,4))
        print('NCG composition:')
        print('CO2',round(1-water_vapour_content,2))
        print('H2O',round(water_vapour_content,2))

    return Q, water_vapour_content

def turbine(inlet, outlet_P, m_WF, eta_in, verbose = True):

    eta_gen = 0.95
    
    outlet_isen = inlet.copy()
    outlet_isen.update('PS',outlet_P, inlet.S)
    outlet_poly = outlet_isen.copy()

    iterr = True
    fce_old = 0
    eta = eta_in
    
    while iterr: 
        dQ = (outlet_isen.H - inlet.H) * eta
        outlet_poly.update('HP',inlet.H + dQ, outlet_P)

        V = m_WF / inlet.rho 
        dH = (inlet.H-outlet_poly.H)

        # 10.1016/j.energy.2012.10.039
        V_max = 1.061 # [m³/s] from PFD + HYSYS
        dH_max = 103858 # [kJ/kg] from PFD + HYSYS

        r_T = np.sqrt(dH/dH_max)
        r_h = (((1.398*r_T - 5.425) * r_T + 6.274) * r_T - 1.866) * r_T + 0.619
        r_VT = np.sqrt(V/V_max)
        r_v = (((-0.21*r_VT+1.117)*r_VT-2.533)*r_VT + 2.588) * r_VT + 0.0038
        eta = eta_in * r_h * r_v
     
        n_isen = np.log(outlet_poly.P/inlet.P) / np.log(outlet_isen.rho/inlet.rho)
        n_poly = np.log(outlet_poly.P/inlet.P) / np.log(outlet_poly.rho/inlet.rho)
        
        fce_new = (((outlet_poly.P / inlet.P) ** ((n_poly - 1) / n_poly) - 1) *
               ((n_poly / (n_poly - 1)) * (n_isen - 1) / n_isen) / ((outlet_poly.P / inlet.P) ** ((n_isen - 1) / n_isen) - 1))
        
        if abs(fce_new - fce_old) < 1e-6:
            iterr = False
            
        fce_old = fce_new
        
    if verbose:
        print('Off-design turbine efficiency loss', round(r_h*r_v,2))

    power = eta_gen * m_WF * (inlet.H - outlet_poly.H)
    return outlet_poly, power 

def pump_curve2(inlet, outlet_P, power):   
    outlet = inlet.copy() 
    dP = outlet_P - inlet.P

    N = 1
    Q_guess = ((-1.870901e-21*dP**3+3.764173e-15*dP**2-1.726355e-8*dP+1.187829e-1) * N)

    def min_fun(N):
        fun = lambda Q: (((8.972547e9*Q**4 - 2.846193e9*Q**3 + 3.073669e8*Q**2 -1.238131e7*Q+3.170715e5)* N**3) - power*0.5)
        Q = fsolve(fun, Q_guess)[0]
        dP_model = (-2.230091e9*Q**3+1.347074e8*Q**2-1.703525e7*Q+3.802477e6) * N

        return (dP_model - dP)**2

    res = minimize(min_fun, N, method = 'Powell', tol = 1e-6)

    N = res.x[0]
    # print('N',N)
    # N = 1

    Q = ((-1.870901e-21*dP**3+3.764173e-15*dP**2-1.726355e-8*dP+1.187829e-1) * N)
    m_WF = (Q) * 2 * inlet.rho # [kg/s]

    eff = (-5.7964e6 * (Q)**4 + 1.7319e6*Q**3 - 1.9829e5*Q**2+1.0242e4*Q-1.149e2)/100

    outlet_T = inlet.T + (power*(1-eff))/(m_WF * inlet.cp)
    outlet.update('TP', outlet_T, outlet_P)

    return m_WF, outlet

def get_tube_htc(tube, HX, m_tube, params, Te):

    excess = tube.copy()
    vap_p = tube.copy()

    if tube.phase == 6:
        # two-phase flow

        bubble = tube.copy()
        dew = tube.copy()
        bubble.update('PQ',tube.P,0.0)
        dew.update('PQ',tube.P,1.0)

        dHvap = dew.H - bubble.H

        bubble.update('TP',bubble.T-1,bubble.P)
        dew.update('TP',dew.T+1,dew.P)

        vap_p.update('TQ',tube.T,0)

        Q = max(0,min(0.999,tube.Q))

        if Te > 0:
            # boiling

            if tube.T + Te > tube.Tc:
                Te = (tube.Tc-1) - tube.T
            excess.update('TQ',tube.T + Te,0)
            dPsat = (excess.P - vap_p.P)

            h_tube1 = ht.boiling_flow.Chen_Bennett(m_tube, Q, HX.tube.tube_ID, bubble.rho, dew.rho, 
                            bubble.mu, dew.mu, bubble.k, bubble.cp, dHvap, bubble.sigma, dPsat, Te)
            h_tube2 = ht.boiling_flow.Chen_Edelstein(m_tube, Q, HX.tube.tube_ID, bubble.rho, dew.rho, 
                            bubble.mu, dew.mu, bubble.k, bubble.cp, dHvap, bubble.sigma, dPsat, Te)
            h_tube3 = ht.boiling_flow.Liu_Winterton(m_tube, Q, HX.tube.tube_ID, bubble.rho, dew.rho, 
                            bubble.mu, bubble.k, bubble.cp, tube.Mw, tube.P, tube.Pc, Te)

            h_tube = np.nanmean([h_tube1,h_tube2,h_tube3])
            # h_tube = h_tube1 * params['tube_condensation_htc_mult']

        else:
            # condensing
            h_tube1 = ht.condensation.Cavallini_Smith_Zecchin(m_tube, Q, HX.tube.tube_ID, bubble.rho, dew.rho, bubble.mu, dew.mu,bubble.k, bubble.cp)
            h_tube2 = ht.condensation.Boyko_Kruzhilin(m_tube, dew.rho, bubble.rho,bubble.k, bubble.mu, bubble.cp, HX.tube.tube_ID, Q)
            h_tube3 = ht.condensation.Akers_Deans_Crosser(m_tube, dew.rho, bubble.rho,bubble.k, bubble.mu, bubble.cp, HX.tube.tube_ID, Q)
            h_tube4 = ht.condensation.Shah(m_tube, Q, HX.tube.tube_ID, bubble.rho, bubble.mu, bubble.k, bubble.cp, tube.P, tube.Pc)
                        
            h_tube = np.nanmean([h_tube1,h_tube2,h_tube3,h_tube4]) * params['tube_condensation_htc_mult']
            

    elif tube.phase == 0 or tube.phase == 5:

        u = (m_tube / tube.rho)/(0.25*np.pi * HX.tube.tube_ID**2) # [m/s]
        Re = HX.tube.tube_ID*u*tube.rho/tube.mu
        Pr = (tube.cp * tube.mu) / (tube.k)
        fd = fluids.friction_factor(Re=Re, eD=HX.tube.eps/HX.tube.tube_ID)

        Nu = ht.conv_internal.Nu_conv_internal(Re, Pr, eD=HX.tube.eps/HX.tube.tube_ID, Di=HX.tube.tube_ID, x=1, fd=fd)
        
        h_tube = Nu * tube.k / HX.tube.tube_ID

        if tube.phase == 0:
            h_tube = h_tube * params['tube_liq_htc_mult']

        if tube.phase == 5:
            h_tube = h_tube * params['tube_gas_htc_mult']

    else: # supercritical but just ignore it for now.
        u = (m_tube / tube.rho)/(0.25*np.pi * HX.tube.tube_ID**2) # [m/s]
        Re = HX.tube.tube_ID*u*tube.rho/tube.mu
        Pr = (tube.cp * tube.mu) / (tube.k)
        fd = fluids.friction_factor(Re=Re, eD=HX.tube.eps/HX.tube.tube_ID)

        Nu = ht.conv_internal.Nu_conv_internal(Re, Pr, eD=HX.tube.eps/HX.tube.tube_ID, Di=HX.tube.tube_ID, x=1, fd=fd)
        
        h_tube = Nu * tube.k / HX.tube.tube_ID

    return h_tube

def get_ext_htc(ext, HX, m_bank, params, Te, verbose = False):

    wall = ext.copy()
    wall.update('TP', ext.T + Te, ext.P)
    Pr_wall = (wall.cp * wall.mu) / (wall.k)

    if HX.shell.type == 'air_cooler':

        ACC = fluids.geometry.AirCooledExchanger(tube_rows=len(HX.tube.rows), tube_passes=HX.tube.passes, tubes_per_row=int(HX.tube.N_tubes / len(HX.tube.rows)),
                tube_length=HX.tube.length,tube_diameter=HX.tube.tube_OD, fin_thickness=HX.tube.fin_thickness, 
                angle=HX.tube.pitch_angle, pitch_parallel=HX.tube.longitudinal_pitch, pitch_normal=HX.tube.transverse_pitch,
                fin_diameter = HX.tube.fin_OD, fin_density=HX.tube.fin_frequency, parallel_bays=HX.shell.bays,
                bundles_per_bay=HX.shell.bundles_per_bay, fans_per_bay = HX.shell.fans_per_bay, corbels = True, 
                tube_thickness=(HX.tube.tube_OD - HX.tube.tube_ID)*0.5,fan_diameter = HX.shell.diameter)

        h_ext1 = ht.air_cooler.h_Briggs_Young(m_bank, ACC.A_per_bundle, ACC.A_min_per_bundle, ACC.A_increase, ACC.A_fin_per_bundle, 
                                ACC.A_tube_showing_per_bundle, HX.tube.tube_OD, HX.tube.fin_OD, HX.tube.fin_thickness, 
                                ACC.bare_length, ext.rho, ext.cp, ext.mu, ext.k, HX.tube.fin_k)

        h_ext2 = ht.air_cooler.h_ESDU_high_fin(m_bank, ACC.A_per_bundle, ACC.A_min_per_bundle, ACC.A_increase, ACC.A_fin_per_bundle,
                                ACC.A_tube_showing_per_bundle, HX.tube.tube_OD, HX.tube.fin_OD, HX.tube.fin_thickness,  
                                ACC.bare_length, HX.tube.longitudinal_pitch, HX.tube.transverse_pitch, 
                                len(HX.tube.rows), ext.rho, ext.cp, ext.mu, ext.k, HX.tube.fin_k, Pr_wall)

        h_ext3 = ht.air_cooler.h_Ganguli_VDI(m_bank, ACC.A_per_bundle, ACC.A_min_per_bundle, ACC.A_increase, ACC.A_fin_per_bundle,
                                ACC.A_tube_showing_per_bundle, HX.tube.tube_OD, HX.tube.fin_OD, HX.tube.fin_thickness,  
                                ACC.bare_length, HX.tube.longitudinal_pitch, HX.tube.transverse_pitch, 
                                len(HX.tube.rows), ext.rho, ext.cp, ext.mu, ext.k, HX.tube.fin_k)

        h_ext = np.nanmean([h_ext1,h_ext2,h_ext3])* params['air_htc_mult']

    elif HX.shell.type == 'lowfin':

        # Check for phase changes
        wall_dew = ext.copy()
        wall_dew.update('PQ', ext.P, 0)

        if wall.Q != ext.Q: # condensation on the wall
            # print('condensation on the wall')
            wall.update('TP', wall_dew.T + 1, ext.P)
            Pr_wall = (wall.cp * wall.mu) / (wall.k)

        ACC = fluids.geometry.AirCooledExchanger(tube_rows=len(HX.tube.rows), tube_passes=HX.tube.passes, tubes_per_row=int(HX.tube.N_tubes / len(HX.tube.rows)),
                tube_length=HX.tube.length,tube_diameter=HX.tube.tube_OD, fin_thickness=HX.tube.fin_thickness, 
                angle=HX.tube.pitch_angle, pitch_parallel=HX.tube.longitudinal_pitch, pitch_normal=HX.tube.transverse_pitch,
                fin_diameter = HX.tube.fin_OD, fin_density=HX.tube.fin_frequency, parallel_bays=1,
                bundles_per_bay=1, fans_per_bay = 1, corbels = False, 
                tube_thickness=(HX.tube.tube_OD - HX.tube.tube_ID)*0.5,fan_diameter = 1)

        h_ext = ht.air_cooler.h_ESDU_low_fin(m_bank, ACC.A, ACC.A_min, HX.tube.Ao_Ai, ACC.A_fin, ACC.A_tube_showing, 
                HX.tube.tube_OD, HX.tube.fin_OD, HX.tube.fin_thickness, ACC.bare_length, HX.tube.longitudinal_pitch, HX.tube.transverse_pitch,
                len(HX.tube.rows), ext.rho, ext.cp, ext.mu, ext.k, HX.tube.fin_k, Pr_wall=Pr_wall)

        
        h_ext = h_ext * params['lowfin_shell_htc_mult']

    elif ext.phase == 6:
        if Te > 0: # nucleic boiling
            bubble = ext.copy()
            dew = ext.copy()
            bubble.update('PQ',ext.P,0.0)
            dew.update('PQ',ext.P,1.0)
            dHvap = dew.H - bubble.H
            Tsat = 0.5*(bubble.T + dew.T)

            bubble.update('TP',bubble.T-1,bubble.P)
            dew.update('TP',dew.T+1,dew.P)

            excess = ext.copy()
            vap_p = ext.copy()

            if ext.T + Te > ext.Tc:
                Te = (ext.Tc-1) - ext.T
            excess.update('TQ',ext.T + Te,0)
            vap_p.update('TQ',ext.T,0)

            dPsat = (excess.P - vap_p.P)

            h_ext = ht.boiling_nucleic.h_nucleic(Te=Te, Tsat=Tsat, P=ext.P, dPsat=dPsat, Cpl=bubble.cp, kl=bubble.k, 
                        mul=bubble.mu, rhol=bubble.rho, sigma=bubble.sigma, Hvap=dHvap, rhog=dew.rho, MW=ext.Mw, Pc=ext.Pc, 
                        CAS='109-66-0')

            # h_ext1 = ht.boiling_nucleic.Forster_Zuber(bubble.rho, dew.rho, bubble.mu, bubble.k, bubble.cp, 
            #                                         dHvap, bubble.sigma, dPsat, Te=Te)
            # h_ext2 = ht.boiling_nucleic.Cooper(ext.P, ext.Pc, ext.Mw, Te=Te)
            # h_ext3 = ht.boiling_nucleic.Gorenflo(ext.P, ext.Pc,CASRN = '109-66-0', Te=Te)
            # h_ext4 = ht.boiling_nucleic.HEDH_Taborek(ext.P, ext.Pc, Te=Te)
            # h_ext5 = ht.boiling_nucleic.McNelly(bubble.rho, dew.rho, bubble.k, bubble.cp,  dHvap, bubble.sigma, ext.P, Te=Te)
            # h_ext6 = ht.boiling_nucleic.Montinsky(ext.P, ext.Pc, Te=Te)
            # h_ext7 = ht.boiling_nucleic.Stephan_Abdelsalam(bubble.rho, dew.rho, bubble.mu, bubble.k, bubble.cp, dHvap,
            #              bubble.sigma, Tsat, Te=Te, correlation='hydrocarbon')

            h_ext = h_ext * params['nucleic_boiling_htc_mult']
        else: # condensation in a shell with tubes
            bubble = ext.copy()
            dew = ext.copy()
            bubble.update('PQ',ext.P,0.0)
            dew.update('PQ',ext.P,1.0)

            bubble.update('TP',bubble.T-1,bubble.P)
            dew.update('TP',dew.T+1,dew.P)

            # 21 degrees angle should be the horizontal tube correction
            h_ext = ht.condensation.Nusselt_laminar(bubble.T, ext.T+Te, dew.rho, bubble.rho, bubble.k, bubble.mu,
                                                     dew.H-bubble.H, 21)
            
            # if ~np.isreal(h_ext):
            #     raise ImaginaryError('Imaginary number detected')

            h_ext = h_ext * np.mean([(j+1)**(-(1/6)) for j in range(len(HX.tube.rows))])
            # Serth 2018 Process Heat transfer, page 441, eq 11.38 - accounts for dripping condensate.
            # a proper implimentation would calculate this for each tube but annoying to pass this information through.

            # K.O. Beatty and D.L. Katz, "Condensation of Vapors on Ouside of finned Tubes," Chem. Eng. Prog., 44, pp. 55-70, 1948. 
            # might be better - apparently surface tension cause draining issues from formed condensate
            h_ext = np.real(h_ext)
            
    

    elif ext.phase == 0 or ext.phase == 5:

        effective_area = HX.shell.shell_ID * HX.shell.baffle_spacing * ((HX.tube.transverse_pitch - HX.tube.tube_OD) / HX.tube.transverse_pitch)
        u = (m_bank / ext.rho) / (effective_area)
        effective_diameter = 4 * (HX.tube.transverse_pitch**2 - (np.pi*HX.tube.tube_OD**2*0.25)) / (np.pi*HX.tube.tube_OD)
        Re = HX.tube.tube_OD * u * ext.rho / ext.mu
        Pr = (ext.cp * ext.mu) / (ext.k)

        n = HX.shell.shell_ID  / HX.tube.transverse_pitch # very approximate

        wall = ext.copy()
        wall.update('TP', ext.T + Te, ext.P)
        Pr_wall = (wall.cp * wall.mu) / (wall.k)

        Nu0 = ht.conv_tube_bank.Nu_ESDU_73031(Re, Pr, len(HX.tube.rows), HX.tube.longitudinal_pitch, HX.tube.transverse_pitch, Pr_wall=Pr_wall)     
        # Nu1 = ht.conv_tube_bank.Nu_HEDH_tube_bank(Re, Pr, HX.tube.tube_OD, len(HX.tube.rows), HX.tube.longitudinal_pitch, HX.tube.transverse_pitch)   
        # Nu2 = ht.conv_tube_bank.Nu_Grimison_tube_bank(Re, Pr, HX.tube.tube_OD, len(HX.tube.rows), HX.tube.longitudinal_pitch, HX.tube.transverse_pitch) 
        Nu3 = ht.conv_tube_bank.Nu_Zukauskas_Bejan(Re, Pr, len(HX.tube.rows), HX.tube.longitudinal_pitch, HX.tube.transverse_pitch, Pr_wall=Pr_wall) 

        # print('Nu_ESDU_73031',Nu0)
        # print('Nu_HEDH_tube_bank',Nu1)
        # print('Nu_Grimison_tube_bank',Nu2)
        # print('Nu_Zukauskas_Bejan',Nu3)

        Nu = np.nanmean([Nu0,Nu3])
        # Nu = np.nanmean([Nu0,Nu1,Nu2,Nu3])

        h_ext = Nu * ext.k / HX.tube.tube_OD

        # print('h_ext',h_ext)

        if ext.phase == 0:
            h_ext = h_ext * params['shell_liq_htc_mult']
        if ext.phase == 5:
            h_ext = h_ext * params['shell_gas_htc_mult']

    else: # supercritical but ignore for now.
        effective_area = HX.shell.shell_ID * HX.shell.baffle_spacing * ((HX.tube.transverse_pitch - HX.tube.tube_OD) / HX.tube.transverse_pitch)
        u = (m_bank / ext.rho) / (effective_area)
        effective_diameter = 4 * (HX.tube.transverse_pitch**2 - (np.pi*HX.tube.tube_OD**2*0.25)) / (np.pi*HX.tube.tube_OD)
        Re = HX.tube.tube_OD * u * ext.rho / ext.mu
        Pr = (ext.cp * ext.mu) / (ext.k)

        n = HX.shell.shell_ID  / HX.tube.transverse_pitch # very approximate

        Nu = ht.conv_tube_bank.Nu_HEDH_tube_bank(Re, Pr, HX.tube.tube_OD, n, HX.tube.longitudinal_pitch, HX.tube.transverse_pitch)        
        h_ext = Nu * ext.k / HX.tube.tube_OD

    return h_ext

def get_both_htc(tube, ext, HX, m_tube, m_bank, params, U, verbose = False):

    Te_tube = (ext.T - tube.T)*0.5
    # Te_ext  = (tube.T - ext.T)*0.5
    Te_ext  = (ext.T - tube.T)*0.5


    # if verbose:
    #     print('Te', Te_ext)
    Te_tube_old = Te_tube
    Te_ext_old = Te_ext

    iterr = True
    ii = 0
    while iterr:

        h_tube  = get_tube_htc(tube, HX, m_tube, params, Te_tube)
        h_ext  = get_ext_htc(ext, HX, m_bank, params, Te_ext, verbose)

        if verbose:
            print('h_ext',h_ext)
            print('h_tube',h_tube)

        U_htc = U(h_ext, h_tube) # [W/m²K]

        Te_tube = (U_htc*(ext.T-tube.T)) / h_tube
        Te_ext  = (U_htc*(tube.T-ext.T)) / h_ext

        if tube.phase == 6  and Te_tube > 0: # tube is boiling
            iterr = True
        elif HX.shell.type == 'lowfin' or ext.phase == 6: # boiling and condensing
            iterr = True
        else:
            iterr = False

        if (Te_tube - Te_tube_old) < 1 and (Te_ext - Te_ext_old) < 1:
            iterr = False
        elif ii > 3:
            raise ImaginaryError
            # print('Te_tube',Te_tube)
            # print('Te_ext',Te_ext)
        else:
            Te_tube_old = Te_tube
            Te_ext_old = Te_ext
            ii += 1

    # if U > 2000:
    #     print('tube.phase',tube.phase)
    #     print('ext.phase',ext.phase)
    #     print('h_tube',h_tube)
    #     print('h_ext',h_ext)
    #     print('h_cond',h_cond)

    return h_ext, h_tube

def get_ext_dP(ext, HX, m_bank ,m_ext):
    if HX.shell.type == 'air_cooler':
        ACC = fluids.geometry.AirCooledExchanger(tube_rows=len(HX.tube.rows), tube_passes=HX.tube.passes, tubes_per_row=int(HX.tube.N_tubes / len(HX.tube.rows)),
                tube_length=HX.tube.length,tube_diameter=HX.tube.tube_OD, fin_thickness=HX.tube.fin_thickness, 
                angle=HX.tube.pitch_angle, pitch_parallel=HX.tube.longitudinal_pitch, pitch_normal=HX.tube.transverse_pitch,
                fin_diameter = HX.tube.fin_OD, fin_density=HX.tube.fin_frequency, parallel_bays=HX.shell.bays,
                bundles_per_bay=HX.shell.bundles_per_bay, fans_per_bay = HX.shell.fans_per_bay, corbels = True, 
                tube_thickness=(HX.tube.tube_OD - HX.tube.tube_ID)*0.5,fan_diameter = HX.shell.diameter)

        dP = -(1/len(HX.tube.rows)) * ht.air_cooler.dP_ESDU_high_fin(m_bank, ACC.A_min, ACC.A_increase, 
                        ACC.flow_area_contraction_ratio, HX.tube.tube_OD, HX.tube.longitudinal_pitch,
                        HX.tube.transverse_pitch, len(HX.tube.rows), ext.rho, ext.mu)

    elif HX.shell.type == 'SHEX':
        effective_area = HX.shell.shell_ID * HX.shell.baffle_spacing * ((HX.tube.transverse_pitch - HX.tube.tube_OD) / HX.tube.transverse_pitch)
        u = (m_ext / ext.rho) / (effective_area)
        effective_diameter = 4 * (HX.tube.transverse_pitch**2 - (np.pi*HX.tube.tube_OD**2*0.25)) / (np.pi*HX.tube.tube_OD)
        Re = effective_diameter * u * ext.rho / ext.mu

        dP = -(1/len(HX.tube.rows)) * ht.conv_tube_bank.dP_Zukauskas(Re,len(HX.tube.rows), HX.tube.transverse_pitch,
                                    HX.tube.longitudinal_pitch, HX.tube.tube_OD, ext.rho, u)
    
    elif HX.shell.type == 'lowfin':
        ACC = fluids.geometry.AirCooledExchanger(tube_rows=len(HX.tube.rows), tube_passes=HX.tube.passes, tubes_per_row=int(HX.tube.N_tubes / len(HX.tube.rows)),
                tube_length=HX.tube.length,tube_diameter=HX.tube.tube_OD, fin_thickness=HX.tube.fin_thickness, 
                angle=HX.tube.pitch_angle, pitch_parallel=HX.tube.longitudinal_pitch, pitch_normal=HX.tube.transverse_pitch,
                fin_diameter = HX.tube.fin_OD, fin_density=HX.tube.fin_frequency, parallel_bays=1,
                bundles_per_bay=1, fans_per_bay = 1, corbels = False, 
                tube_thickness=(HX.tube.tube_OD - HX.tube.tube_ID)*0.5,fan_diameter = 1)

        dP = -(1/len(HX.tube.rows)) * ht.air_cooler.dP_ESDU_low_fin(m_bank, ACC.A_min, ACC.A_increase, 
                        ACC.flow_area_contraction_ratio, HX.tube.tube_OD, 
                        HX.tube.fin_height, ACC.bare_length, HX.tube.longitudinal_pitch,
                        HX.tube.transverse_pitch, len(HX.tube.rows), ext.rho, ext.mu)

    else:
        print('shell type must be one of:','air_cooler, SHEX, lowfin')

    return dP

def get_tube_dPdz(tube, HX, m_tube):

    if tube.phase == 0 or tube.phase == 5:
        dPdz = fluids.friction.one_phase_dP(m_tube, tube.rho, tube.mu, HX.tube.tube_ID, roughness=HX.tube.eps)
    else:
        # should account for non-condensible gases in the bubble/dew calculations 
        # maybe a try except for finding bubble/dew points in the first place.
        bubble = tube.copy()
        dew = tube.copy()
        bubble.update('PQ',tube.P,0.0)
        dew.update('PQ',tube.P,1.0)

        bubble.update('TP',bubble.T-1,bubble.P)
        dew.update('TP',dew.T+1,dew.P)

        tube.Q = min(0.999,max(0.001,tube.Q))

        # dPdz = fluids.two_phase.Xu_Fang(m_tube, tube.Q, bubble.rho, dew.rho, bubble.mu, dew.mu,
        #                                  bubble.sigma, HX.tube.tube_ID, roughness=HX.tube.eps)

        dPdz = fluids.two_phase.two_phase_dP(m_tube, tube.Q, bubble.rho, HX.tube.tube_ID, 1.0, dew.rho, 
                                        bubble.mu, dew.mu, bubble.sigma, tube.P, tube.Pc, HX.tube.eps, angle = 0)

    return dPdz

def ode_shell_section(tube_in, ext_in, HX, L, m_tube, m_bank, m_ext_total, params, tube_H_arr, tube_P_arr, reverse=False, verbose = False):

    tube = tube_in.copy()
    ext_out = ext_in.copy()

    if reverse:
        z_arr = np.linspace(L,0,100)
    else:
        z_arr = np.linspace(0,L,100)

    # Set ode parameters
    rtol = 1e-3
    atol = 1e-3
    max_step = 0.5

    def wrapper(t,y):
        return shell_fun(t,y, HX, tube, ext_in, m_tube, m_bank, m_ext_total, params, tube_H_arr, tube_P_arr, reverse=reverse, verbose = verbose)

    n0 = np.array([ext_in.H, ext_in.P])

    out = solve_ivp(wrapper, (z_arr[0],z_arr[-1]), n0, 'LSODA', t_eval = z_arr, rtol=rtol, atol=atol, max_step = max_step)

    H_out_ext = out.y[0][-1]
    P_out_ext = out.y[1][-1]

    ext_out.update('HP', H_out_ext, P_out_ext)

    return ext_out, out.t, out.y[0], out.y[1]

def shell_fun(z,H, HX, tube, ext, m_tube, m_bank, m_ext, params, tube_H_arr, tube_P_arr, reverse = False, verbose = False):

    ext.update('HP', H[0], H[1])
        
    A_tube = np.pi * HX.tube.tube_ID # [m²/m] one tube
    A_ext = np.pi * HX.tube.tube_OD # [m²/m] one tube
    A = 0.5*(A_tube + A_ext)
    R_int = 0
    if R_int == 0:
        R_i = 0
    else:
        R_i = (R_int/A_tube)

    U = lambda h_ext, h_tube : ((((1/(h_tube*A_tube)) +
                             ((HX.tube.tube_OD - HX.tube.tube_ID) / (HX.tube.k_tube *A)) + 
                                  (1/(h_ext*A_ext)) + 
                                  (R_i/A_tube))**(-1)) / (A))

    tube_arr = [tube.copy() for i in range(len(tube_H_arr))]
    Q_arr = []

    for i in range(len(tube_H_arr)):
        tube_arr[i].update('HP', tube_H_arr[i](z), tube_P_arr[i](z))

        # incorporate baffle short curcuiting here with a lower htc in cross flow with lower mass flow across each bank

        h_ext, h_tube = get_both_htc(tube_arr[i], ext, HX, m_tube, m_bank, params, U)
        U_htc = U(h_ext, h_tube)
        # print('shell U', U_htc)

        Q_arr.append( U_htc*A * (tube_arr[i].T-ext.T) )

    N_baffles = HX.tube.length / HX.shell.baffle_spacing
    effective_area = HX.shell.shell_ID * HX.shell.baffle_spacing * ((HX.tube.transverse_pitch - HX.tube.tube_OD) / HX.tube.transverse_pitch)
    u = (m_bank / ext.rho) / (effective_area)
    effective_diameter = 4 * (HX.tube.transverse_pitch**2 - (np.pi*HX.tube.tube_OD**2*0.25)) / (np.pi*HX.tube.tube_OD)
    Re = effective_diameter * u * ext.rho / ext.mu

    n = HX.shell.shell_ID  / HX.tube.transverse_pitch # very approximate

    dPdz = -(1/HX.tube.length) * ht.conv_tube_bank.dP_Zukauskas(Re, n, HX.tube.transverse_pitch, HX.tube.longitudinal_pitch, HX.tube.tube_OD, ext.rho, u)
    dHdz = ((np.mean(Q_arr))/(m_ext))
        
    if reverse:
        return np.array([-dHdz, -dPdz])
    else:
        return np.array([dHdz, dPdz])

def ode_tube_section(tube_in, ext_in, HX, L, R_f, m_tube, m_bank, m_ext, params, reverse = False, shell_H_fun = 'NA', shell_P_fun ='NA', verbose = False):

    tube_in_T = tube_in.T

    tube = tube_in.copy()
    ext_out = ext_in.copy()

    Te = (tube.T - ext_in.T)*0.5
    h_ext = get_ext_htc(ext_in, HX, m_bank, params, Te) # this is only used in crossflow cases
    #should ideally remove this

    # might be worth guessing the temperature of the external to get a better average value.

    if reverse:
        z_arr = np.linspace(L,0,100)
    else:
        z_arr = np.linspace(0,L,100)

    # Set ode parameters
    rtol = 1e-3
    atol = 1e-3
    max_step = 0.2

    if shell_H_fun == 'NA': # cross flow
        def wrapper(t,y):
            return tube_fun(t,y, HX, m_tube, tube, ext_out, h_ext, m_bank, params, R_f, reverse=reverse, verbose=verbose)
    else: # counter current flow
        def wrapper(t,y):
            return tube_fun(t,y, HX, m_tube, tube, ext_out, h_ext, m_bank, params, R_f, shell_H_fun, shell_P_fun, reverse=reverse, verbose = verbose)

    n0 = np.array([tube_in.H, tube_in.P]) # add inlet pressure here

    out = solve_ivp(wrapper, (z_arr[0],z_arr[-1]), n0, 'LSODA', t_eval = z_arr, rtol=rtol, atol=atol, max_step = max_step)

    H_out_tube = out.y[0][-1]
    P_out_tube = out.y[1][-1]

    H_out_ext = ((m_tube * (tube_in.H - H_out_tube))/m_ext) + ext_in.H
    tube.update('HP', H_out_tube, P_out_tube)
    ext_out.update('HP', H_out_ext, ext_in.P)
    return tube, ext_out, out.t, out.y[0], out.y[1]

def tube_fun(z,H, HX, m_tube, tube, ext_av, h_ext, m_bank, params, R_f, shell_H_fun = 'NA', shell_P_fun ='NA', reverse=False, verbose = False):

    tube.update('HP', H[0], H[1])
    
    A_tube = np.pi * HX.tube.tube_ID # [m²/m] one tube
    A_ext = np.pi * HX.tube.tube_OD # [m²/m] one tube
    A = 0.5*(A_tube + A_ext)
    
    if R_f == 0:
        R_i = 0
    else:
        R_i = (R_f/A_tube)

    U = lambda h_ext, h_tube : ((((1/(h_tube*A_tube)) +
                             ((HX.tube.tube_OD - HX.tube.tube_ID) / (HX.tube.k_tube *A)) + 
                                  (1/(h_ext*A_ext)) + 
                                  (R_i))**(-1)) / (A))

    U2 = lambda h_ext, h_tube : ((((1/(h_tube*A_tube)) +
                             ((HX.tube.tube_OD - HX.tube.tube_ID) / (HX.tube.k_tube *A)) + 
                                  (1/(h_ext*A_ext)))**(-1)) / (A))

    if shell_H_fun == 'NA':
        Te_tube = (ext_av.T - tube.T)*0.5
        h_tube = get_tube_htc(tube, HX, m_tube, params, Te_tube)
    else:
        try:
            ext_av.update('HP', shell_H_fun(z), shell_P_fun(z))
        except:
            raise ImaginaryError
        h_ext, h_tube = get_both_htc(tube, ext_av, HX, m_tube, m_bank, params, U, verbose)

    U_htc = U(h_ext, h_tube)
    
    U2_htc = U2(h_ext, h_tube)

    # if verbose:
    #     print('R_f:', R_f)
    #     print('fouling HTC:', U_htc)
    #     print('w/o fouling HTC:', U2_htc)

    dPdz = get_tube_dPdz(tube, HX, m_tube)
    dHdz = -((U_htc*A)/(m_tube)) * (tube.T-ext_av.T)

    if reverse:
        return np.array([-dHdz, dPdz])
    else:
        return np.array([dHdz, -dPdz])

def crossflow_tube_all(Z, tube_H_fun, tube_P_fun, ext_H_fun, ext_P_fun, tube, ext, m_tube, m_bank, m_ext, params, HX, L, verbose = False):

    H_ext_out_arr = []
    P_ext_out_arr = []

    for i in Z:
        H_tube = tube_H_fun(i)
        P_tube = tube_P_fun(i)

        H_ext_in = ext_H_fun(i)
        P_ext_in = ext_P_fun(i)
        H_ext_out, P_ext_out = crossflow_tube(tube,H_tube, P_tube,ext, H_ext_in, P_ext_in, m_tube, m_bank, m_ext, params, HX,L, verbose)

        if ~np.isreal(H_ext_out):
            print(i)

        H_ext_out_arr.append(H_ext_out)
        P_ext_out_arr.append(P_ext_out)

    

    ext_H_fun, ext_P_fun = interp(Z, np.array(H_ext_out_arr),np.array(P_ext_out_arr))

    return ext_H_fun, ext_P_fun

def crossflow_tube(tube,H_tube, P_tube,ext, H_ext_in, P_ext_in, m_tube, m_bank, m_ext, params, HX,L, verbose = False):

    tube.update('HP',H_tube, P_tube)
    ext.update('HP',H_ext_in, P_ext_in)

    A_tube = np.pi * HX.tube.tube_ID # [m²/m] one tube
    A_ext = np.pi * HX.tube.tube_OD # [m²/m] one tube
    A = 0.5*(A_tube + A_ext)
    R_int = 0
    if R_int == 0:
        R_i = 0
    else:
        R_i = (R_int/A_tube)

    U = lambda h_ext, h_tube : ((((1/(h_tube*A_tube)) +
                             ((HX.tube.tube_OD - HX.tube.tube_ID) / (HX.tube.k_tube *A)) + 
                                  (1/(h_ext*A_ext)) + 
                                  (R_i/A_tube))**(-1)) / (A))

    h_ext, h_tube = get_both_htc(tube, ext, HX, m_tube, m_bank, params, U, verbose)
    U_htc = U(h_ext, h_tube)
    # if verbose:
    #     print('h_ext', h_ext)
    #     print('h_tube', h_tube)

    Q = U_htc*A*L*(tube.T-ext.T)
    dH = Q / m_ext

    dP = -(1/len(HX.tube.rows)) * get_ext_dP(ext, HX, m_bank ,m_ext)

    return H_ext_in + dH, P_ext_in + dP

def interp(z,H,P):
    H_fun = interp1d(z, np.array(H) ,kind='cubic',fill_value='extrapolate')
    P_fun = interp1d(z, np.array(P),kind='cubic',fill_value='extrapolate')
    return H_fun, P_fun

def rigorous_condenser(air_in, condenser_in, m_wf, m_air, params, verbose = False):
    condenser_out = condenser_in.copy()
    air_out = air_in.copy()
    air_in_H = air_in.H
    condenser_in_H = condenser_in.H

    HX = json.load(open('RK21_air_cooler.json'), object_hook=jsondecoder)
    L = HX.tube.length
    R_f = 0

    N_bundles = HX.shell.bays * HX.shell.bundles_per_bay
    N = len(HX.tube.rows)
    # tubes_per_row = HX.tube.N_tubes / len(HX.tube.rows)
    tubes_per_row = HX.tube.rows

    m_tube = m_wf * (1/HX.tube.N_tubes) * (1/N_bundles) # number of tubes is *per bundle*
    
    m_bank = m_air / N_bundles # exernal flow for htc corellation

    Z = np.linspace(0,L,100)

    ext_H_fun, ext_P_fun = interp(Z,np.ones(Z.shape) * air_in.H ,np.ones(Z.shape) * air_in.P)

    tube_out_arr = [0 for i in range(len(HX.tube.rows))]
    ext_in_arr = [0 for i in range(len(HX.tube.rows)+1)]
    ext_in_arr[0] = air_in.copy()

    if verbose:
        plt.figure()

    # for i in range(len(HX.tube.rows)):
    for i,N_tubes in enumerate(tubes_per_row[:N]):
        m_ext = m_air * (1/N_tubes) * (1/N_bundles) # air passing over one tube
        tube_out_arr[i], ext_in_arr[i+1],z,H,P = ode_tube_section(condenser_in.copy(), ext_in_arr[i], 
                                        HX, L, R_f, m_tube, m_bank, m_ext, params, False, ext_H_fun, ext_P_fun,verbose)

        tube_H_fun, tube_P_fun = interp(z,H,P)

        if verbose:
            plt.plot(z, H)

        ext_H_fun, ext_P_fun = crossflow_tube_all(Z, tube_H_fun, tube_P_fun, ext_H_fun, ext_P_fun, 
                                                    condenser_in.copy(), ext_in_arr[i], m_tube, m_bank, m_ext, params, HX, L)

        air_in.update('HP', np.average(ext_H_fun(Z)), np.average(ext_P_fun(Z)))     

        # print('air temperature', air_in.T -273.15)                   

    if verbose:
        plt.show()
    ext_out_arr = ext_in_arr[1:]
    
    condenser_out_T = np.average([tube_out_arr[i].T for i in range(len(tube_out_arr))])
    condenser_out_P = np.average([tube_out_arr[i].P for i in range(len(tube_out_arr))])
    condenser_out_H = np.average([tube_out_arr[i].H for i in range(len(tube_out_arr))])
    condenser_out.update('HP', condenser_out_H, condenser_out_P)

    Q = m_wf * (condenser_in_H - condenser_out.H)

    air_out.update('HP', air_in_H + Q/m_air, np.average(ext_P_fun(Z)))
    # air_out.update('HP', np.average(ext_H_fun(Z)), np.average(ext_P_fun(Z)))

    return air_out, condenser_out

def rigorous_preheater(shell_in, tube_in, m_tube_total, m_shell, shell_DT, shell_DP, params,verbose = False):
    
    tube_in_T = tube_in.T

    HX = json.load(open('RK21_preheater.json'), object_hook=jsondecoder)
    L = HX.tube.length

    t = params['t']
    a = params['vap_fouling_a']
    b = params['vap_fouling_b']
    R_f = a*t*(1-np.exp(-b*t))
    # R_f = 0

    m_tube = m_tube_total / (HX.tube.N_tubes / HX.tube.passes)
    m_ext_tube = m_shell / (HX.tube.N_tubes)

    if 'shell_H_fun' in params:
        shell_H_fun = params['shell_H_fun']
        shell_P_fun = params['shell_P_fun']
        Z = np.linspace(0,L,100)
    else:
        # estimate shell temperature profile - also a check to ensure we dont cross boiling point.
        shell_out = shell_in.copy()
        shell_BP = shell_in.copy()
        shell_BP.update('PQ',shell_out.P - shell_DP, 0)
        if shell_out.T + shell_DT > shell_BP.T:
            shell_DT = shell_BP.T - shell_in.T
        
        shell_out.update('TP',shell_in.T + shell_DT, shell_in.P - shell_DP) # guess shell_out

        Z = np.linspace(0,L,100)
        H_gradient = (shell_out.H- shell_in.H)/L
        P_gradient = (shell_out.P- shell_in.P)/L
        # shell_H_fun, shell_P_fun = interp(np.flip(Z), shell_in.H + H_gradient*(Z), shell_in.P + P_gradient*(Z))
        shell_H_fun, shell_P_fun = interp(Z, shell_in.H + H_gradient*(Z), shell_in.P + P_gradient*(Z))

    stop = True
    i = 0
    max_i = 10

    while stop:

        H_shell_old = shell_H_fun(Z)
        P_shell_old = shell_P_fun(Z)

        # print('tube inlet temperature', tube_in.T-273.15)
        # print('shell inlet temperature', shell_in.T-273.15)
        # print('shell outlet temperature', shell_out.T-273.15)

        # perhaps the 0.95 as bypass can be determined. shortcurcuting. - This assumes to HT in windows to be nil
        tube_out, _, z, H, P  = ode_tube_section(tube_in.copy(), shell_in.copy(), HX, L, R_f, m_tube, m_shell*0.95,
                m_shell, params, False, shell_H_fun, shell_P_fun, verbose) # first pass

        # print('tube midpoint temperature', tube_out.T-273.15)

        tube_out1, _, z1, H1, P1  = ode_tube_section(tube_out.copy(), shell_in.copy(), HX, L, R_f, m_tube, m_shell*0.95,
                m_shell, params, True, shell_H_fun, shell_P_fun, verbose) # second pass

        cp_tube = (tube_in.cp + tube_out1.cp)*0.5

        tube_H_fun, tube_P_fun = interp(z, H, P)
        tube_H1_fun, tube_P1_fun = interp(z1, H1, P1)

        # print('shell inlet temperature', shell_in.T-273.15)

        # tube_in, ext_in, HX, L, m_tube, m_bank, m_ext_total, tube_H_arr, tube_P_arr

        ext_out, z2, H_shell, P_shell  = ode_shell_section(tube_in.copy(), shell_in.copy(), HX, L, m_tube_total, m_shell, m_ext_tube, params, 
                [tube_H_fun, tube_H1_fun], [tube_P_fun, tube_P1_fun], reverse=False, verbose=verbose)

        cp_shell = (shell_in.cp + ext_out.cp)*0.5

        # print('shell outlet temperature', ext_out.T-273.15)

        shell_H_fun, shell_P_fun = interp(z2, H_shell, P_shell)

        H_shell_new = shell_H_fun(Z)
        P_shell_new = shell_P_fun(Z)

        SSD_H = np.sum(((H_shell_old - H_shell_new) / (np.mean(H_shell_old)))**2)
        SSD_P = np.sum(((P_shell_old - P_shell_new) / (np.mean(P_shell_old)))**2)

        i += 1
        if (SSD_H < 1e-2 and SSD_P < 1e-1) or i > max_i:
            if verbose:
                print('Preheater iterations required:',i)
            stop = False
        else:# something to speed up convergence?
            pass

        if verbose and (i == 0 or not stop):
            plt.figure()
            plt.plot(z, H_shell_old/cp_shell + (shell_in.T - 273.15))
            plt.plot(z, H_shell_new/cp_shell + (shell_in.T - 273.15))
            plt.legend(['old', 'new'])
            plt.show()

            plt.figure()
            plt.plot(z, tube_H_fun(z)/cp_tube + (tube_in_T - 273.15))
            plt.plot(z, tube_H1_fun(z)/cp_tube + (tube_in_T - 273.15))
            plt.legend(['tube pass 1', 'tube pass 2'])
            plt.show()

    params['shell_H_fun'] = shell_H_fun
    params['shell_P_fun'] = shell_P_fun

    ext_out.update('HP',H_shell[-1],P_shell[-1])

    if verbose:
        print('Vaporizer inlet temperature (°C)', round(ext_out.T - 273.15,2))
    return ext_out, tube_out1

def rigorous_recuperator(tube_in, shell_in, m_tube_total, m_shell, tube_pass_DT, tube_pass_DP, params, verbose = False):

    shell_in_H = shell_in.H
    tube_in_H = tube_in.H

    HX = json.load(open('RK21_recuperator.json'), object_hook=jsondecoder)
    L = HX.tube.length
    R_f = 0
    tube_out1 = tube_in.copy()
    tube_out = tube_in.copy()
    shell_out = shell_in.copy()

    # guess pass 1 temperature
    tube_out1.update('TP',tube_in.T + tube_pass_DT, tube_in.P - tube_pass_DP)

    # tube count arrays (counting upwards)
    N = int(len(HX.tube.rows)/2)
    bottom_tubes = HX.tube.rows[:N] # pass 2
    top_tubes = HX.tube.rows[N:] # pass 1
    
    N1 = len(bottom_tubes)
    N2 = len(top_tubes)
    Z = np.linspace(0,L,100)

    tube_pass_1 = [tube_in.copy() for i in range(N2)]

    m_tube = m_tube_total / (HX.tube.N_tubes / HX.tube.passes)

    # m_ext = m_shell / (tubes_per_layer) # do this within the iteration
    # m_ext = m_shell / (tubes_per_layer[i]) # do this within the iteration

    stop = True
    ii = 0

    while stop:

        H_pass1 = tube_out1.H
        P_pass1 = tube_out1.P
        T_pass1 = tube_out1.T

        shell = shell_in.copy()
        shell_mid = shell_in.copy()
        tube_pass_2 = [tube_out1.copy() for i in range(N1)]
        tube_pass_2_out = ["" for i in range(N1)]

        ext_H_fun, ext_P_fun = interp(Z, np.ones(Z.shape) * shell_in.H , np.ones(Z.shape) * shell_in.P)
        # do pass 2 of tubes using shell in and estimated tubeside properties
        # plt.figure()
        for i,N_tubes in enumerate(bottom_tubes):
            m_ext = m_shell / (N_tubes)

            tube_pass_2_out[i], _, z, H, P  = ode_tube_section(tube_pass_2[i].copy(), shell,
                    HX, L, R_f, m_tube, m_shell, m_ext, params, True, ext_H_fun, ext_P_fun, verbose=verbose)

            tube_H_fun, tube_P_fun = interp(z, H, P)
            ext_H_fun, ext_P_fun = crossflow_tube_all(Z, tube_H_fun, tube_P_fun, ext_H_fun, ext_P_fun, 
                                                    tube_pass_2[i].copy(), shell, m_tube, m_shell, m_ext, params, HX, L, verbose = False)

            # plt.plot(((tube_H_fun(Z) - tube_out1.H) / (tube_out1.cp)) + (tube_out1.T-273.15))
            # plt.plot(((ext_H_fun(Z) - shell_in.H) / (shell_in.cp))+ (shell_in.T-273.15))

        H_arr2 = np.array([float(tube_pass_2_out[i].H) for i in range(len(bottom_tubes))])
        P_arr2 = np.array([float(tube_pass_2_out[i].P) for i in range(len(bottom_tubes))])

        H2_av  = np.sum(H_arr2  * (bottom_tubes/np.sum(bottom_tubes)))
        P2_av  = np.sum(P_arr2  * (bottom_tubes/np.sum(bottom_tubes)))

        shell_mid.update('HP', np.mean(ext_H_fun(Z)), np.mean(ext_P_fun(Z)))

        tube_out.update('HP', H2_av, P2_av)
        tube_pass_1 = [tube_in.copy() for i in range(N2)]
        tube_pass_1_out = ["" for i in range(N2)]

        # plt.figure()
        for i,N_tubes in enumerate(top_tubes):
            m_ext = m_shell / (N_tubes)

            tube_pass_1_out[i], _, z, H, P  = ode_tube_section(tube_pass_1[i].copy(), shell,
                    HX, L, R_f, m_tube, m_shell, m_ext, params, False, ext_H_fun, ext_P_fun, verbose=verbose)

            tube_H_fun, tube_P_fun = interp(z, H, P)
            ext_H_fun, ext_P_fun = crossflow_tube_all(Z, tube_H_fun, tube_P_fun, ext_H_fun, ext_P_fun, 
                                                    tube_pass_1[i].copy(), shell, m_tube, m_shell, m_ext, params, HX, L)                                   
            # plt.plot(((tube_H_fun(Z) - tube_in.H) / (tube_in.cp)) + (tube_in.T-273.15))
            # plt.plot(((ext_H_fun(Z) - shell_mid.H) / (shell_mid.cp))+ (shell_mid.T-273.15))

        H_arr1 = np.array([float(tube_pass_1_out[i].H) for i in range(len(top_tubes))])
        P_arr1 = np.array([float(tube_pass_1_out[i].P) for i in range(len(top_tubes))])

        H1_av  = np.sum(H_arr1 * (top_tubes/np.sum(top_tubes)))
        P1_av  = np.sum(P_arr1 * (top_tubes/np.sum(top_tubes)))

        tube_out1.update('HP', H1_av, P1_av)
        shell_out.update('HP', np.mean(ext_H_fun(Z)), np.mean(ext_P_fun(Z)))

        SSD_H = (H_pass1 - tube_out1.H)**2
        SSD_P = (P_pass1 - tube_out1.P)**2

        if ii == 0:
            if verbose:
                print('pass1 T out (old) (°C)', round(T_pass1 - 273.15,2))
                print('pass1 T out (new) (°C)', round(tube_out1.T - 273.15,2))

        ii = ii + 1
        if (SSD_H < 1e3 and SSD_P < 1e3) or ii > 10:
            stop = False

        if not stop:
            if verbose: 
                print('recuperator required iterations', ii)
                print('pass1 T out (old) (°C)', round(T_pass1 - 273.15,2))
                print('pass1 T out (new) (°C)', round(tube_out1.T - 273.15,2))

                plt.figure()
                plt.plot(H_arr1)
                plt.plot(H_arr2)
                plt.legend(['pass 1', 'pass 2'])

    params['recuperator_tube_pass_DT'] = abs(tube_out1.T - tube_in.T) # [K]
    params['recuperator_tube_pass_DP'] = abs(tube_out1.P - tube_in.P) # [kPa]

    Q = m_shell * (shell_in.H - shell_out.H)

    tube_out.update('HP', tube_in_H + Q/m_tube_total, P2_av)

    if verbose:
        print('preheater WF inlet', tube_out.T-273.15)
    return tube_out, shell_out

def rigorous_vaporizer(shell_in, tube_in, m_tube_total, m_shell, vaporizer_level, NCG_content, params,verbose = False):
    tube_out = tube_in.copy()

    HX = json.load(open('RK21_vaporizer_large.json'), object_hook=jsondecoder)
    N = len(HX.tube.rows)
    tubes_per_layer = HX.tube.rows

    vaporizer_level = vaporizer_level + params['vaporizer_level_offset']

    # guesses for shell conditions upon entry to large HX
    large_shell_in = shell_in.copy()
    large_shell_in.update('PQ',shell_in.P, 0)
    large_tube_in_array = [tube_in.copy() for i in range(N)]

    iterr = True
    iii = 1

    while iterr:
        large_tube_in_array = [tube_in.copy() for i in range(N)]

        shell_mid_H_old = large_shell_in.H
        shell_mid_P_old = large_shell_in.P

        large_tube_out_array, shell_out = large_vaporizer(large_shell_in, large_tube_in_array, m_tube_total, m_shell, vaporizer_level, params, verbose = verbose)
        tube_out_array, large_shell_in = small_vaporizer(shell_in.copy(), large_tube_out_array, m_tube_total, m_shell, vaporizer_level, params, verbose = verbose)

        shell_mid_H_new = large_shell_in.H
        shell_mid_P_new = large_shell_in.P

        SSD_H = (shell_mid_H_old - shell_mid_H_new)**2
        SSD_P = (shell_mid_P_old - shell_mid_P_new)**2

        if (SSD_H < 1e-2 and SSD_P < 1e-2) or iii > 10:
            iterr = False
        else:
            iii = iii + 1

        if not iterr:
            if verbose:
                if large_shell_in.Q < 1 and large_shell_in.Q > 0:
                    print('fluid leaving small shell has a vapour fraction of:',round(large_shell_in.Q,2))
                else:
                    print('fluid leaving small shell is subcooled')

    tube_out_H = np.array([tube_out_array[i].H for i in range(len(tube_out_array))])
    tube_out_P = np.array([tube_out_array[i].P for i in range(len(tube_out_array))])

    flow_pct = np.array(tubes_per_layer/np.sum(tubes_per_layer))

    tube_out_all_H = np.sum(tube_out_H * flow_pct)
    tube_out_all_P = np.sum(tube_out_P * flow_pct)

    tube_out.update('HP',tube_out_all_H, tube_out_all_P) 

    # Q = m_tube_total * (tube_in.H - tube_out_all_H)
    # H_shell_out = shell_in.H + (Q/m_shell)
    # shell_out.update('HP',H_shell_out, shell_out.P)

    NCG_in  = stream('TP',tube_in.T,tube_in.P,{'CO2':1.0})
    
    Q, water_vapour_content = NCG_flash(NCG_content, tube_out.T, tube_out.P, params['eos'], verbose)

    MW_av_total = NCG_in.Mw * (NCG_content) + tube_in.Mw * (1-NCG_content)  
    MW_av_vapour = NCG_in.Mw * (1-water_vapour_content) + tube_in.Mw * (water_vapour_content) 
    total_molar_flowrate = m_tube_total / (MW_av_total)
    mass_flowrate_vapour = total_molar_flowrate * Q * MW_av_vapour
    m_tube_out = m_tube_total - mass_flowrate_vapour 

    NCG_CO2_content = (1-water_vapour_content)

    if verbose:
        print('mass flow NCG', round(mass_flowrate_vapour,2))
    return shell_out, tube_out, m_tube_out, NCG_CO2_content

def large_vaporizer(shell_in, tube_arr, m_tube_total, m_shell, vaporizer_level, params, verbose=False):
    # assume cross flow

    HX = json.load(open('RK21_vaporizer_large.json'), object_hook=jsondecoder)
    N = len(HX.tube.rows)
    L = HX.tube.length

    t = params['t']
    a = params['vap_fouling_a']
    b = params['vap_fouling_b']
    R_f = a*t*(1-np.exp(-b*t))
    # R_f = 0

    layers = len(HX.tube.rows)
    tubes_per_layer = HX.tube.rows # HX.tube.N_tubes / layers

    # rows_per_layer = HX.tube.N_tubes / N
    Z = np.linspace(0,L,100)

    tube_arr_out = [0 for i in range(N)]
    shell_out = shell_in.copy()

    shell_vapor = shell_in.copy() # superheated vapour
    shell_vapor.update('PQ',shell_in.P,1)
    shell_vapor.update('TP',shell_vapor.T+1,shell_vapor.P)

    shell_boiling = shell_in.copy() # boiling
    shell_boiling.update('PQ',shell_in.P,0.5)

    shell_in_H = shell_in.H
    shell_in_P = shell_in.P

    m_tube = m_tube_total / HX.tube.N_tubes
    # m_ext = m_shell / rows_per_layer

    for i,N_tubes in enumerate(tubes_per_layer):
        m_ext = m_shell / (N_tubes)

    # for i in range(N):
        if i/N > vaporizer_level: # assign exterior as dew point?
            shell_in = shell_vapor.copy()       
        else: # normal\
            shell_in = shell_boiling.copy()
        tube_arr_out[i], _, z, H, P  = ode_tube_section(tube_arr[i].copy(), shell_in.copy(),
                HX, L, R_f, m_tube, m_shell, m_shell, params, False, verbose=verbose)

    tube_out_H = np.array([tube_arr_out[i].H for i in range(len(tube_arr_out))])
    tube_in_H  = np.array([tube_arr[i].H for i in range(len(tube_arr))])

    flow_pct = np.array(tubes_per_layer/np.sum(tubes_per_layer))
    dH = np.sum((tube_in_H - tube_out_H) * flow_pct)
    
    Q = m_tube_total * (dH)
    shell_out_H = shell_in_H + Q/(m_shell)
    shell_out.update('HP', shell_out_H, shell_in_P)

    # might be worth checking for a temperature cross here!
    return tube_arr_out, shell_out

def small_vaporizer(shell_in, tube_in_array, m_tube_total, m_shell, vaporizer_level, params, verbose=False):
    # assume counter-current
    HX = json.load(open('RK21_vaporizer_small.json'), object_hook=jsondecoder)
    N = len(HX.tube.rows)
    L = HX.tube.length

    t = params['t']
    a = params['vap_fouling_a']
    b = params['vap_fouling_b']
    R_f = a*t*(1-np.exp(-b*t))
    # R_f = 0

    m_tube = m_tube_total / (HX.tube.N_tubes)
    tubes_per_layer = HX.tube.rows # HX.tube.N_tubes / layers

    shell_out = shell_in.copy()
    shell_in_H = shell_in.H
    shell_in_P = shell_in.P

    shell_DP = shell_in.copy() # dew point
    shell_DP.update('PQ',shell_in.P,1)
    shell_DP.update('TP',shell_DP.T+1,shell_DP.P)

    tube_arr_out = [0 for i in range(N)]

    for i,N_tubes in enumerate(tubes_per_layer):
        m_ext = m_shell / (N_tubes)
        if i/N > vaporizer_level: # assign exterior as dew point?
            shell_in = shell_DP.copy()          
        else: # normal
            shell_in = shell_in.copy() 
        tube_arr_out[i], _, z, H, P  = ode_tube_section(tube_in_array[i].copy(), shell_in, HX, L, R_f, m_tube, m_shell*0.95,
                m_shell, params, False, verbose = verbose) 

    tube_out_H = np.array([tube_arr_out[i].H  for i in range(len(tube_arr_out))])
    tube_in_H  = np.array([tube_in_array[i].H for i in range(len(tube_in_array))])

    flow_pct = np.array(tubes_per_layer/np.sum(tubes_per_layer))
    dH = np.sum((tube_in_H - tube_out_H) * flow_pct)

    Q = m_tube_total * (dH)
    shell_out_H = shell_in_H + Q/(m_shell)
    shell_out.update('HP', shell_out_H, shell_in_P)

    return tube_arr_out, shell_out
    