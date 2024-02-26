
"""
Kubyshkina & Fossati (2021) grid and interpolator (already included)
Updated grid of planet parameters and mass loss rates based on the
hydrodynamic simulations by Kubyshkina et al. (2018).

Reference: https://ui.adsabs.harvard.edu/abs/2021RNAAS...5...74K/abstract).

Zenodo repository: https://zenodo.org/record/4643823

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
International Public License (https://creativecommons.org/licenses/by-nc/4.0/).
"""

import numpy as np

## Added by Jorge for relative paths
from photoevolver.settings import _MODEL_DATA_DIR
__teq_sma_dataset = np.loadtxt(_MODEL_DATA_DIR+'kubyshkina18/teq-sma.txt')


## astronomical constants        
Lsun=3.9e33;    #solar luminocity        /*erg*s^{-1}*/
Rsun=6.96e10; #solar radius /*cm*/
AU=1.4960e13; #astronomical unit /*cm*/
Re=6.378e8; #earth radius /*cm*/        
Me=5.9722e27; #earth mass /*g*/
kb=1.3807e-16;        #boltzman const /*erg*K^{-1}*/
mh=1.6726e-24;  #hydrogen mass /*g*/
G0=6.6726e-8;   #gravitational constant /* cm^3*g^{-1}*s^{-2}*/

def SMAXIS(Mss,Teq):
#INPUT: Mstar [Msun], Teq [K]
#OUTPUT: d0 [au]
#This function returns the orbital separation for the specific stellar mass and equilibrium temperature, as defined in the grid.
#The value of the orbital separation is the averaged value according to stellar models at different ages (based on MIST stellar isochrone models by Choi et al., 2016; ApJ, 823, 102).
    # mstd = np.loadtxt(PyDir+'teq-sma.txt');
    global __teq_sma_dataset
    mstd = __teq_sma_dataset
    mss00 = mstd[:,0];
    teq00 = mstd[:,1];
    d00 = mstd[:,2];
    #mstd=mstd1;
    Mss_g = [0.4, 0.6, 0.8, 1.0, 1.3];
    T_g = np.linspace(300, 3000, 28); #300:100:3000;
            
    d0_g = np.zeros( (len(Mss_g), len(Teq)));
    
    for i in range(len(Mss_g)):        
        tmp = np.interp( Teq, T_g, d00[np.where(mss00 == Mss_g[i])]);
        d0_g[i,:] = tmp;
    #end for
    
    d0 = np.zeros( (len(Mss), 1));
    for i in range(len(Teq)):
        d0[i] = np.interp( Mss[i], Mss_g, d0_g[:,i]);
    #end for
        
    return d0

def BETA0(rp,mp,t0):
#constants Re, Me, kb, mh, G0 required  
#INPUT: Rpl [Re], Mpl [Me], Teq [K]
#OUTPUT: Lambda
#This function defines the generalised gravitational (Jeans) parameter Lambda as in Fossati et al., 2017; A&A, 598, A90. We used to name it 'beta' until someone has pointed out that it's then looking as plasma parameter, but the name remains in the code.
        bb=G0*mp*Me/(rp*Re*(kb*t0/mh));
        return bb;
                
def unique_rows(a):
# returns unique rows of the array a; to define the available 'grid' of temperatures, euv, etc for the specific interval of planetary mass/radius
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def ADEN(rp,mp):
#constants Re, Me required
#INPUT: Rpl [Re], Mpl [Me]
#OUTPUT: rho_av [g/cm^3]
#average density of a planet
        aveg_den=3.*mp*Me/(rp*rp*rp*Re*Re*Re*3.1416*4);
        return aveg_den;
        
def ADEN_R(mp,aden0):
#constants Re, Me required
#INPUT: Mpl [Me], rho_av [g/cm^3]
#OUTPUT: Rpl [Re]
#define the radius of the planet of given mass and average density
        aveg_R=pow(3.*mp*Me/aden0/(Re*Re*Re*3.1416*4),1./3.);
        return aveg_R;
        
def lin_extpl(x1, x2, y1, y2, x):
        #linear function for two points;
        k = (y1-y2)/(x1-x2);
        b = y1-(k*x1);
        y = k*x + b;
        return y
                
                

########################################################################################
# analytical approximation from Kubyshkina et al., 2018b (HBA)
########################################################################################
                
def Lborder(dau,fxuv,rpre):
#INPUT: d0 [AU], Feuv [erg/s/cm^2], Rpl [RE]
#OUTPUT: Lambda
#border points between two regimes for the given parameters

        zeta=-1.297796148718774+6.861843637445744;
        eta=0.884595403184073+0.009459807206476;

        beta=32.019929264625155 -16.408393523348366;
        alp1=0.422232254541188 -1;
        alp2=-1.748858849270155 +3.286179370395197;
        alp3=3.767941293231585 -2.75;
                
        K = (zeta + eta*np.log(dau)); 
        C = beta + alp1*np.log(fxuv) + alp2*np.log(dau) + alp3*np.log(rpre);

        ld= np.exp(C/K);
        return ld;

def testCF(dau,fxuv,rpre,ld):
#INPUT: d0 [AU], Fxuv [erg/s/cm^2], Rpl [RE], LAMBDA
#OUTPUT: escape rate [g/s]
# approximation for the 'high gravity regime'
        zeta=-1.297796148718774;
        eta=0.884595403184073;

        beta=16.408393523348366 ;
        alp2=-3.286179370395197;
        alp3=2.75;

        K = (zeta + eta*np.log(dau)); 
        C = beta + np.log(fxuv) + alp2*np.log(dau) + alp3*np.log(rpre);

        mdot= np.exp(C + K*np.log(ld));
        return mdot;

def testCFJ(dau,fxuv,rpre,ld):
#INPUT: d0 [AU], Fxuv [erg/s/cm^2], Rpl [RE], LAMBDA
#OUTPUT: escape rate [g/s]
# approximation for the 'low gravity regime'

        zeta=-6.861843637445744;
        eta=-0.009459807206476;

        beta=32.019929264625155;
        alp1=0.422232254541188;
        alp2=-1.748858849270155;
        alp3=3.767941293231585;

        K = (zeta + eta*np.log(dau)); 
        C = beta + alp1*np.log(fxuv) + alp2*np.log(dau) + alp3*np.log(rpre);

        mdot= np.exp(C + K*np.log(ld));
        return mdot;
                

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

__txtdata = None

def INTERPOL(Mss,EUV,T_i,r_i,m_i, dataset_file = None):   #,NTEST
## INPUT: Mstar [Msun], EUV [erg/cm/s], Teq [K], Rpl [Re], Mpl [Me]
## import numpy as np
## import scipy.optimize as fitit
##  
## OUTPUT: escape rate [g/s]

        AllowExtrapolation = True # 0 -- strictly prohibit extrapolation; 1 -- allow extrapolation (see readme.pdf)
        AllowExtrapolationRM = True #same for Rpl and Mpl only
        NotifyIfExtrapolate = True # set 0 to switch off the notifications
        
        Mssg0 = np.array([0.4, 0.6, 0.8, 1.0, 1.3])
        if(Mss>=0.6):
            Tg0 = np.array([300, 700, 1100, 1500, 2000])
        else:
            Tg0 = np.array([300, 700, 1100, 1500])
        
        rg0 = np.linspace(1, 10, 10); 
        mg0 = np.array([1., 1.6, 2.1, 3.2, 4.3, 5., 6.7, 7.8, 9., 12.1, 14., 16.2, 21.7, 29.1, 39., 45.1, 60.5, 81.0, 93.8, 108.6]);
        
        ## first check of input parameters
        use = ((Mss > max(Mssg0) or Mss<min(Mssg0)) or (T_i>max(Tg0) or T_i<min(Tg0)) or (r_i>max(rg0) or r_i<min(rg0)) or (m_i>max(mg0) or m_i<min(mg0)));

        if(use and not AllowExtrapolation):
            
            ITPL = -1;
            if(NotifyIfExtrapolate):
                print('Planet parameters are not in the grid range, please check the input or set AllowExtrapolation=1')
                print('Stellar mass range: 0.4 ... 1.3 Msun')
                print('Equilibrium temperature range: 300 ... 2000 K; 300 ... 1500 K in case of Mstar = 0.4 Msun')
                print('Planetary mass range: 1 ... 108.6 Mearth')
                print('Planetary radius range: 1 ... 10 Rearth')
            return ITPL
        
        ## loading the input file; to speed up the calculations lines 186-195 can be moved to the main program, and then passes to the function

        try: input0
        except NameError: input0 = None
        
        # ADDED BY JORGE
        global __txtdata
        if dataset_file is None:
            if __txtdata is None: 
                __txtdata = np.loadtxt(_MODEL_DATA_DIR+'kubyshkina18/input_interpol_26-03-2021.dat');
            input0 = __txtdata
        else: input0 = np.loadtxt(PyDir + dataset_file)
        
        #if input0 is None:
        if 1:
          #input0 = np.loadtxt('input_interpol_26-03-2021.dat');
          mss = input0[:,0]; #stellar mass [Msun]
          teq = input0[:,1]; #equilibrium temperature [K]
          sma = input0[:,2]; #orbital separation [AU]
          euv = input0[:,3]; #EUV flux at the planetary orbit [erg/s/cm^2]
          rpl = input0[:,4]; #planetary radius [Rearth]
          mpl = input0[:,5]; #planetary mass [Mearth]
          bt = input0[:,7]; # reduced Jeans escape parameter Lambda
          lhy = input0[:,6]; #hydrodynamical mass loss [g/s]
          mindex = input0[:,-1]; #technical index for extrapolation 0/1/2 []
        
       
          #bt = BETA0(rpl, mpl, teq);
        # END
        
        beta_i = BETA0( r_i, m_i, T_i);#
        
        ## find closest Mstar in grid, set subgrid

        n_tmp, = np.where( abs(Mssg0-Mss) == 0);   #equal

        if(len(n_tmp) == 0):
            dmssg = np.abs(Mssg0-Mss);
            mssg = np.array(sorted(Mssg0[np.argsort(dmssg)][0:2]));
        else:
            mssg = Mssg0[n_tmp];
        

        lhy_ii_mss = np.linspace(0, 0, len(mssg));#np.zeros((len(mssg),1));
        

#Mstar cycle
        for imss in range(len(mssg)):
            ## reset Teq grid for specific Mss, check parameters again
            usg, = np.where(mss == mssg[imss]); 

            #print( np.where((Tg0<=max(teq[usg]))*(Tg0>=min(teq[usg]))) )
            Tg1 = Tg0[np.where( (Tg0<=max(teq[usg]))*(Tg0>=min(teq[usg])) )];
            use = (T_i>max(Tg1) or T_i<min(Tg1));
            if(use and NotifyIfExtrapolate and AllowExtrapolation):
                print('Extrapolation by Teq!')
            if(use and not AllowExtrapolation):
                if(NotifyIfExtrapolate):
                    print('Planet parameters are not in the grid range, the requested Teq is not available for the given stellar mass')
                    print('Equilibrium temperature range: 300 ... 2000 K; 300 ... 1500 K in case of Mstar = 0.4 Msun')
                ITPL = -1;
                return ITPL
                
            ## find closest Teq and set Teq subgrid
            n_tmp, = np.where(np.abs(Tg1-T_i) == 0);   #equal

            if(len(n_tmp) == 0):
                dtg = np.abs(Tg1-T_i);
                tg = np.array(sorted(Tg1[np.argsort(dtg)][0:2]));#sortmult([abs(Tg1-T_i) Tg1],1,0);
            else:
                tg = Tg1[n_tmp];
                 
                
            lhy_ii_teq = np.linspace(1, len(tg), len(tg));#np.zeros((len(tg),1));
            
                        
#Teq cycle  
            for iteq in range(len(tg)):
                ## check euv range
                #  print(np.where((mss == mssg[imss])*(teq == tg[iteq])) )
                usg, = np.where( (mss == mssg[imss])*(teq == tg[iteq]) );
                use = EUV<min(euv[usg]) or EUV>max(euv[usg]);
                if(use and NotifyIfExtrapolate and AllowExtrapolation):
                    print('Extrapolation by EUV')
                if(use and not AllowExtrapolation):
                    if(NotifyIfExtrapolate):
                      print(f'Planet parameters are not in the grid range, this Feuv is not available for the Mstar = {mssg[imss]}Msun and Teq = {tg[iteq]}');
                      print(f'Available Feuv interval is {min(euv[usg])} - {max(euv[usg])} erg/s/cm^2')
                        
                    ITPL = -1;
                    return ITPL
                        
##
                ## set euv grid
                xuvg1 = sorted(set(euv[usg]));
                
                if((not xuvg1) == False):
                    xuvg = np.array(xuvg1);
                                                    
                    lhy_ii_xuv = np.linspace(1, len(xuvg), len(xuvg))#np.zeros((len(xuvg),1));
                    
#euv cycle                                
                    for ixuv in range(len(xuvg)):
                        usg, = np.where( (mss == mssg[imss])*(teq==tg[iteq])*(euv==xuvg[ixuv]) );

                        #interpol_b0 for bt(usg), rpl(usg)
                        n_tmp, = np.where((mss == mssg[imss])*(teq==tg[iteq])*(euv==xuvg[ixuv])*(r_i==rpl)*(m_i==mpl));
                                        
                        if((len(n_tmp) == 0) and len(xuvg)>1):
                            rplg = np.array(sorted(set(rpl[usg])));
                            lhy_ii_rpl = 0*np.linspace(1, len(rplg), len(rplg));#np.zeros((len(rplg),1));
                            
                                        
                            irpl = np.min(rplg)-1;
#rpl cycle                                            
                            while (irpl < max(rplg) ):
                                    irpl += 1;
                                    usg1, = np.where( (mss == mssg[imss])*(teq==tg[iteq])*(euv==xuvg[ixuv])*(rpl==irpl) );
                                
                                    if(len(usg1)>1):
#lhy(bt)                          
                                        use = beta_i<min(bt[usg1]) or beta_i>max(bt[usg1])
                                        #print(use)
                                        if(not use):
                                          lhy_ii_rpl[int(irpl-1)] = np.exp(np.interp( np.log(beta_i), np.log(bt[usg1]), np.log(lhy[usg1])));
                                        elif(use and AllowExtrapolationRM):
                                          if(NotifyIfExtrapolate and (abs(irpl - r_i)<1.)):
                                            print('Extrapolation by Lambda')
                                            print(f'Mass range is {min(mpl[usg1])} - {max(mpl[usg1])} Mearth\n(Lambda = {min(bt[usg1]):.2f} - {max(bt[usg1]):.2f}, Lambda_pl = {beta_i:.2f})\nat Teq = {tg[iteq]} K, Mstar = {mssg[imss]} Msun, Rpl = {irpl} Rearth\n')
                                          if(beta_i<min(bt[usg1])):
                                            mindex_i = 0
                                          else:
                                            mindex_i = 1
                                          usg_e, = np.where( (mss == mssg[imss])*(teq==tg[iteq])*(euv==xuvg[ixuv])*(rpl==irpl)*(mindex==mindex_i) );
                                          if(len(usg_e)<=2 and mindex_i==0):
                                            usg_e = usg1[0:3]
                                          if(len(usg_e)<=2 and mindex_i==1):
                                            usg_e = usg1[-3:len(usg1)]
                                          par = np.polyfit(np.log(bt[usg_e]), np.log(lhy[usg_e]), 1);
                                          par_fit = np.poly1d(par);
                                          lhy_ii_rpl[int(irpl-1)] = np.exp(par_fit(np.log(beta_i)));
                                        else:
                                          if(NotifyIfExtrapolate):
                                            print("The planetary mass is out of range for the given Mstar, Teq, Feuv and Rpl")
                                          ITPL = -1
                                          return ITPL


                                    

                                        
                            rplg = (rplg[np.where(lhy_ii_rpl>0)]);
                                            
                            lhy_ii_rpl    = lhy_ii_rpl[np.where(lhy_ii_rpl>0)];
                            

                
                            if( (r_i>max(rplg) or r_i<min(rplg)) and AllowExtrapolationRM):
                                us, = np.where(np.diff(lhy_ii_rpl) > 0);                                    
                                ind0 = 0*np.linspace(1,len(rplg),len(rplg));
                                ind0[us] = 1;
                                #print((rplg>min(rplg))*(ind0.transpose()>0))
                                us = np.where( (rplg>min(rplg))*(ind0) );
#lhy(rpl) ext.          
                                if(r_i<min(rplg)):
                                      us1 = us[0:2];
                                else:
                                      us1 = us[-2:len(us)];
                                par = np.polyfit(np.log(rplg[us1]), np.log(lhy_ii_rpl[us1]), 1);
                                par_fit = np.poly1d(par);
                                lhy_ii_xuv[ixuv] = np.exp(par_fit(np.log(r_i)));
                                                        
                            else:
#lhy(rpl)                        
                                lhy_ii_xuv[ixuv] = np.exp(np.interp( np.log(r_i), np.log(rplg), np.log(lhy_ii_rpl)));
                                                     
                            
                        else: #if 122
                            if(len(n_tmp) == 0):
                                lhy_ii_xuv[ixuv]    = 0;
                                
                            else:                    
                                lhy_ii_xuv[ixuv]    = lhy[n_tmp];
                                
                                            
                xuvg = xuvg[np.where(lhy_ii_xuv > 0)];
                lhy_ii_xuv = lhy_ii_xuv[np.where(lhy_ii_xuv > 0)];
                
                        
                n_tmp, = np.where(np.abs(xuvg-EUV)<0.01*EUV);
                        
                                                        
                ## here set par_ii(Tg(iteq))
                if(len(xuvg)>1 and (len(n_tmp) == 0)):
                    if(EUV<=max(xuvg) and EUV>=min(xuvg)):
        #lhy(xuv)       This can also be linear, but the logarithm works better at the low Feuv    
                        lhy_ii_teq[iteq] = np.exp(np.interp( np.log(EUV), np.log(xuvg), np.log(lhy_ii_xuv)));

                    else:
                        par = np.polyfit(np.log(xuvg), np.log(lhy_ii_xuv), 1);
                        par_fit = np.poly1d(par);
                        lhy_ii_teq[iteq] = np.exp(par_fit(np.log(EUV)));
 
                                                                
                else: #if 285
                    if(len(n_tmp) == 0):
                        lhy_ii_teq[iteq]    = 0;
                        
                    else:
                        lhy_ii_teq[iteq]    = lhy_ii_xuv[n_tmp];
                        
                    
                
                ## here define par(T_i) : 
            tg = tg[np.where(lhy_ii_teq>0)];
            lhy_ii_teq    = lhy_ii_teq[np.where(lhy_ii_teq>0)];
            
                
            n_tmp, = np.where(tg==T_i);
                            
            #here set par_ii(Mssg(imss))
            if( len(tg)>1 and (len(n_tmp) == 0)):
                if(len(lhy_ii_teq)>0):
                        if( T_i>1000 and T_i <= max(tg)):#base on semi-axis
                          tg1 = SMAXIS(np.ones((len(tg),1))*mssg[imss], tg);
                          tg1 = tg1[:,0];
                          T_i1 = SMAXIS([mssg[imss]], [T_i]);
                          T_i1 = T_i1[0,0];
                        
#lhy(teq) T>1000   
                        #print(np.log(tg1));print(np.log(T_i1))      
                        #print(np.log(lhy_ii_teq))
                          lhy_ii_mss[imss] = np.exp( np.interp( -np.log(T_i1), -np.log(tg1), np.log(lhy_ii_teq)));  
                        #print(np.log(lhy_ii_mss[imss]))
      
                        else: #if 342 %base on teq
#lhy(teq) T<=1000            
                          lhy_ii_mss[imss] = np.exp( np.interp( T_i, tg, np.log(lhy_ii_teq)));
#    
                    
            else: #if 340
                if(len(n_tmp) == 0):
                    lhy_ii_mss[imss]    = 0;
                    
                else:
                    lhy_ii_mss[imss]    = lhy_ii_teq[n_tmp[0]];
                    

        ## here define par(Mss)
        mssg = mssg[np.where(lhy_ii_mss>0)];
        lhy_ii_mss    = lhy_ii_mss[np.where(lhy_ii_mss>0)];
        #print(mssg)
        #print(lhy_ii_mss)
        
        
        n_tmp, = np.where(mssg==Mss);
                
        if(len(mssg)>1 and (len(n_tmp) == 0)):
            Lhy_i    = np.interp( Mss, mssg, lhy_ii_mss);
            

            # Len_i = Len_K( Mss, SMAXIS( Mss, T_i), m_i, r_i, Reff_i, EUV, 0);
        else: #424
            if(len(n_tmp) == 0):
                Lhy_i    = 0;              
                
            else:
                Lhy_i    = lhy_ii_mss[n_tmp[0]];
                            
        ##
        #ITPL = [Lhy_i, Reff_i, Len_i, Nmin_i, Rdis_i, Tmax_i, Vmax_i/1e5, KSImax_i];
        ITPL = Lhy_i;
        return ITPL
#end INTERPOL

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
