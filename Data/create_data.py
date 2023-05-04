# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:17:25 2022

@author: ba8641
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
import sys

#from amplpy import AMPL, modules
#modules.load() # load all modules
from amplpy import AMPL
ampl = AMPL() # instantiate AMPL object

from amplpy import AMPL
ampl = AMPL() # instantiate AMPL object

ampl.option["solver"] = "gurobi"


#Hyperparameters##############################
#132 countries; 55 vaccines; 33 PAHO; 74 Gavi; 6 overlap of PAHO and Gavi
#   N_entities = 132-33-74+6 = 33

exp_from = 1
exp_to = 150
n_markets_list = [132]
n_ent_list=[2,3]
unc_high_list = [10]
interest_rate_list = [0]
alpha_list = [1]
ROI = 1.15
HIC_GNI = 13205
UMIC_GNI = 4255
LMIC_GNI = 1085
au_demand_list = []         #
gen_supply = 1e9

min_price_rate = [1,2]
commitment = [0,1]

create_file = True
main_path = r'/home/ba8641/'
path_regression_coeficients = ''

au_vac = ['AU_DTwP-HepB-Hib', 'AU_MR', 'AU_PCV13', 'AU_Rota', 'AU_Hib'] #Vaccines offered by AU
####What monetary incentive would be necessary not to disrupt the capacity; Quantity for supply as current demand;
##Look at the QUALI metric as the new possible metric for an intervention
#Build a slide presentation
au_price_rate = [1]         #Price multiplier of AU price over Gavi one

comb_vaccines_dict = {'tOPV':[['mOPV2','mOPV1']],
                      'MR':[['Measles','Rubella']],
                      'IPV':[['mOPV2','mOPV1']],
                      'DTwP-HepB-Hib':[['HepB','DTwP','Hib']],
                      'bOPV':[['mOPV1']],
                      'DTaP-HepB-Hib-IPV':[['HepB','IPV','DTaP','Hib']],
                      'DTaP':[['DT']],
                      'DTwP':[['DT']],
                      'DTaP-HepB-Hib':[['DTaP','HepB']],
                      'DTaP-HepB-IPV':[['DTaP', 'HepB', 'IPV']],
                      'DTaP-Hib':[['DTaP','Hib']],
                      'DTaP-Hib-IPV':[['DTaP','IPV','Hib']],
                      'DTaP-IPV':[['DTaP','IPV']],
                      'DTwP-HepB':[['DTwP','HepB']],
                      'DTwP-Hib':[['DTwP','Hib']],
                      'Hib-MenAC':[['Hib','MenAC']],
                      'Hib-MenC':[['Hib','MenC']],
                      'HPV2':[['HPV']],
                      'HPV2 & HPV4':[['HPV2','HPV4']],
                      'HPV4':['HPV2'],
                      'HPV9':[['HPV4']],
                      'MenAC':[['MenA','MenC']],
                      'MenAC Ps':[['MenA Ps','MenC']],
                      'MenACW-135Ps':['MenAC Ps'],
                      'MenACYW-135':[['MenAC']],
                      'MenACYW-135 Ps':[['MenACW-135Ps']],
                      'MenBC':[['MenB','MenC']],
                      'MM':[['Measles','Mumps']],
                      'MMR':[['MM', 'Rubella'],['MR','Mumps']],
                      'MMRV':[['MMR']],
                      'PCV10':[['PCV']],
                      'PCV10 & PCV13':[['PCV13']],
                      'PCV13':[['PCV10']],
                      'PPSV':[['PCV13']],
                      'Td':[['TT']],
                      'Td-IPV':[['Td','IPV']],
                      'Tdap':[['Td']],
                      'Tdap-IPV':[['Tdap','IPV']]
                      }

antigen_dict = {'mOPV':[['IPV', 'DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'bOPV']],
                      'Measles':[['MR','MMR', 'Measles']],
                      'Rubella':[['MR','MMR']],
                      'HepB':[['HepB','DTaP-HepB-Hib-IPV','DTwP-HepB-Hib']],
                      'HepA':[['HepA']],
                      'DT':[['DT', 'DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'DTwP', 'DTwP-HepB-Hib', 'DTwP-Hib']],
                      'P':[['DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'DTwP', 'DTwP-HepB-Hib', 'DTwP-Hib']],
                      'HPV':[['HPV2','HPV4']],
                      'Hib':[['DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTwP-HepB-Hib', 'DTwP-Hib', 'Hib']],
                      'MenA':[['MenA', 'MenACW-135Ps', 'MenACYW-135']],
                      'Tet':[['Td', 'Tdap']],
                      'PCV':[['PCV10', 'PCV13']],
                      'Mumps':[['MMR']],
                      'Var':[['Varicella']],
                      'YF':[['YF']],
                      'Typ':[['TyphoidPs','TCV']],
                      'BCG':[['BCG']],
                      'Rota': [['Rota']],
                      'OCV': [['OCV']]
                      }

au_antigen_dict = {'mOPV':[['IPV', 'DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'bOPV']],
                      'Measles':[['MR','MMR', 'Measles', 'AU_MR']],
                      'Rubella':[['MR','MMR', 'AU_MR']],
                      'HepB':[['HepB','DTaP-HepB-Hib-IPV','DTwP-HepB-Hib']],
                      'HepA':[['HepA']],
                      'DT':[['DT', 'DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'DTwP', 'DTwP-HepB-Hib', 'DTwP-Hib', 'AU_DTwP-HepB-Hib']],
                      'P':[['DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTaP-IPV', 'DTwP', 'DTwP-HepB-Hib', 'DTwP-Hib']],
                      'HPV':[['HPV2','HPV4']],
                      'Hib':[['DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTwP-HepB-Hib', 'DTwP-Hib', 'Hib', 'AU_Hib']],
                      'MenA':[['MenA', 'MenACW-135Ps', 'MenACYW-135']],
                      'Tet':[['Td', 'Tdap']],
                      'PCV':[['PCV10', 'PCV13', 'AU_PCV']], #Pneumococcal
                      'Mumps':[['MMR']],
                      'Var':[['Varicella']],
                      'YF':[['YF']],
                      'Typ':[['TyphoidPs','TCV']],
                      'BCG':[['BCG']],
                      'Rota': [['Rota','AU_Rota']],
                      'OCV': [['OCV']]          #Cholera
                      }

ant = pd.DataFrame(antigen_dict).transpose()
au_ant = pd.DataFrame(au_antigen_dict).transpose()
ant.index.name = 'A'
au_ant.index.name = 'A'
ant.columns = ['V1']
au_ant.columns = ['V1']

#Import vaccine demand data########################################################################

dt = pd.read_csv(main_path+'/Detailed_Demand_Forecast_data.csv')
all_countries = pd.unique(dt.iloc[:,1])
countries_2021 = pd.unique(dt[dt['Year']==2021]['Country (group)'])
all_vaccines = pd.unique(dt['Vaccine'])
vaccines_2021 = pd.unique(dt[dt['Year']==2021]['Vaccine'])

dat = dt[(dt['Year']==2021) & (dt['Current WB Status & Current Gavi Eligibility'].str.contains('Gavi'))]

ct = pd.read_csv(main_path+'/Country-data.csv')
ct['PAHO'] = (ct['WHO Region']=='AMR') & (~ct['Country'].isin(['USA', 'Canada']))
ct = ct.sort_values('Country')

pt = pd.read_csv(main_path+'/Prices.csv')
prt = pt[(pt['Year (Price)']==2021) & (pt['Source']!='V3P') & (pt['Price Tier (group) (4.2)']!='USA')].drop(['Price Tier (group) (4.2)','Year (Price)'],axis=1)


pt_PAHO = pd.read_csv(main_path+'/99_PAHO_Pricing_data.csv')
pt_PAHO = pt_PAHO[(pt_PAHO['Year']==2021) & (pt_PAHO['Avg. Price'].notnull())].drop(['Vial Size','% Difference in Avg. Price Color','Year'], axis = 1)

pt_PAHO['Avg. Price'] = pt_PAHO['Avg. Price'].replace('[/$,]', '', regex=True).astype(float) #Already price/dose
pt_PAHO = pt_PAHO.groupby('Vaccine').min()
pt_PAHO.reset_index(inplace=True)
pt_PAHO = pt_PAHO.rename(columns = {'index':'Vaccine'})

dat['Price']=-1
#ct_PAHO = dat[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country']))]
dat.loc[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country'])),'Price'] = dat[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country']))].merge(pt_PAHO[['Vaccine','Avg. Price']], left_on='Vaccine Sub-type', right_on='Vaccine', how='left')['Avg. Price'].values
#ct_PAHO.merge(pt_PAHO[['Vaccine','Avg. Price']], on='Vaccine', how='left')

dat = dat.rename(columns={'Current WB Status & Current Gavi Eligibility':'Gavi', 'Country (group)':'Country', 'Total Required Supply': 'Demand'})

#UNICEF prices
pt_UN = pd.read_csv(main_path+'/910_UNICEF_Pricing_data.csv')
pt_UN = pt_UN[(pt_UN['Year']==2021) & (pt_UN['Min. Price'].notnull())].drop(['Year','% Difference in Min. Price Color','Company  ', 'Vial Size'], axis = 1)
pt_UN['Min. Price'] = pt_UN['Min. Price'].replace('[/$,]', '', regex=True).astype(float) #Price per dose
pt_UN = pt_UN.groupby('Vaccine ').min()
pt_UN.reset_index(inplace=True)
pt_UN = pt_UN.rename(columns = {'index':'Vaccine'})

#Merge dat with UNICEF
dat.loc[(dat['Gavi'].isin(['Gavi Eligible','Gavi Transitioning','Gavi Fully Self-financing']))&(dat['Price']==-1),'Price'] = dat[(dat['Gavi'].isin(['Gavi Eligible','Gavi Transitioning','Gavi Fully Self-financing']))&(dat['Price']==-1)].merge(pt_UN[['Vaccine ','Min. Price']], left_on='Vaccine Sub-type', right_on='Vaccine ', how='left')['Min. Price'].values
dat = dat.reset_index(drop=True)

#Correct missing prices

#For now just deleting non-gavis
dat = dat[~((dat['Gavi']=='Non-Gavi MIC') & (dat['Price']==-1))]
dat = dat[~dat['Country'].isin(ct[ct['GNI per Capita (USD)']==0]['Country'])] #North Korea

##Find coefficients ##############################################################

countries = pd.unique(dat['Country']).tolist()

con = pd.DataFrame({'Market' : ['PAHO','Gavi']})
con_au = pd.DataFrame({'Market' : ['PAHO','Gavi','AU']})
con['GNI/cap'] = con_au['GNI/cap'] = 0
con.loc[con['Market']=='PAHO','GNI/cap'] = con_au.loc[con_au['Market']=='PAHO','GNI/cap'] = ct.groupby('PAHO')['GNI per Capita (USD)'].mean().loc[True]
con.loc[con['Market']=='Gavi','GNI/cap'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False)]['GNI per Capita (USD)'].mean()
con_au.loc[con_au['Market']=='Gavi','GNI/cap'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False) & (ct['WHO Region']!='AFR')]['GNI per Capita (USD)'].mean()
con_au.loc[con_au['Market']=='AU','GNI/cap'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False) & (ct['WHO Region']=='AFR')]['GNI per Capita (USD)'].mean()


def get_coef(df):
    gni = df['GNI/cap'].tolist()
    M=np.zeros([len(df['Market']),len(df['Market'])], dtype=float)
    for i in range(0,len(df['Market'])):
        for j in range(0,len(df['Market'])):
            M[i,j]=gni[i]/gni[j]
            
    w,v=np.linalg.eig(M) 
    princ=[]
    princ=np.where((abs(w.real)>=0.0000001)&(abs(w.imag)<0.00000001))[0]

    ind_interest=-1

    if len(princ)>0: 
        ind_interest=princ[0]
    else:
        print("Warning: No real eigen vector exists")
        
    f = pd.DataFrame(v[:,ind_interest].real/np.average(v[:, ind_interest].real), index=[df['Market']])
    f.columns=["Weight"]
    return f

coef = get_coef(con)
coef_au = get_coef(con_au)
    
Frame = pd.read_csv("FrameRP.csv")


##################################################################################
#Consolidate markets into PAHO, Gavi, and AU
#################################################################################

#Define "baseline vaccine prices
dat=dat.merge(Frame, on="Country", how="left")
b_pr = pd.DataFrame(dat['Price']/dat['Weight'])
b_pr.columns=["Corrected Price"]
b_pr['Vaccine'] = dat['Vaccine Sub-type']
b_pr.loc[b_pr['Vaccine'] == 'MenACW-135 Ps','Vaccine'] = 'MenACW-135Ps'
b_pr.loc[b_pr['Vaccine'] == 'Typhoid Ps','Vaccine'] = 'TyphoidPs'

def MakeVaccines(alpha):
    vac = b_pr.dropna().groupby('Vaccine').mean()
    min_vac = b_pr.dropna().groupby('Vaccine').min()
    auv = vac.copy()
    for i in au_vac:
        auv.loc[i,'Corrected Price'] = alpha*auv.loc[i[3:],'Corrected Price']
        min_vac.loc[i,'Corrected Price'] = min_vac.loc[i[3:],'Corrected Price']
    return vac, auv, min_vac
    
vaccines, au_vaccines, min_vac = MakeVaccines(1)

coef.to_csv("CoefRP.csv")
coef_au.to_csv("CoefRP_AU.csv")
vaccines.to_csv("Baseline.csv")

rp = pd.DataFrame(np.matmul(vaccines[['Corrected Price']], coef[['Weight']].transpose()))
rp.columns = coef.index

min_pr = pd.DataFrame(np.matmul(min_vac[['Corrected Price']], coef[['Weight']].transpose()))
min_pr.columns = coef.index

rp_au = pd.DataFrame(np.matmul(au_vaccines[['Corrected Price']], coef_au[['Weight']].transpose()))
rp_au.columns = coef_au.index

min_pr_au = pd.DataFrame(np.matmul(min_vac[['Corrected Price']], coef_au[['Weight']].transpose()))
min_pr_au.columns = coef_au.index


exp_list = range(exp_from,exp_to+1)
number = len(unc_high_list)*len(interest_rate_list)*len(n_markets_list)*len(exp_list)

con.loc[con['Market']=='PAHO','Births'] = con_au.loc[con_au['Market']=='PAHO','Births'] = ct.groupby('Country').mean().groupby('PAHO')['Birth Cohort'].sum().loc[1.0]
con.loc[con['Market']=='Gavi','Births'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False)].groupby('Country').mean()['Birth Cohort'].sum()
con_au.loc[con_au['Market']=='Gavi','Births'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False) & (ct['WHO Region']!='AFR')]['Birth Cohort'].sum()
con_au.loc[con_au['Market']=='AU','Births'] = ct[(ct['Gavi Co-financing'] != 'Non-Gavi') & (ct['PAHO'] == False) & (ct['WHO Region']=='AFR')]['Birth Cohort'].sum()

con.index = con['Market']
con = con.drop('Market', axis = 1)
con.columns = ['g', 'd']

con_au.index = con_au['Market']
con_au = con_au.drop('Market', axis = 1)
con_au.columns = ['g', 'd']
dem_v = dat.groupby(['Vaccine Sub-type']).sum(['Demand'])
comp = [dem_v.index.values[z] in vaccines.index.values for z in range(len(dem_v.index))]
vaccines['k'] = dem_v.loc[comp,'Demand']
vaccines.loc[vaccines['k'].isna(),'k'] = vaccines.loc[~vaccines['k'].isna(),'k'].mean()
vaccines.loc[vaccines['k']<2*1e8,'k'] = 2*1e8

comp = [dem_v.index.values[z] in au_vaccines.index.values for z in range(len(dem_v.index))]
comp[-3] = comp[-2] = comp[-1] = True
au_vaccines['k'] = dem_v.loc[comp,'Demand']
au_vaccines.loc[au_vaccines['k'].isna(),'k'] = au_vaccines.loc[~au_vaccines['k'].isna(),'k'].mean()
au_vaccines.loc[au_vaccines['k']<2*1e8,'k'] = 2*1e8

Var = pd.DataFrame()
individual = pd.DataFrame()
summary = pd.DataFrame()
fail = pd.DataFrame()
inf = pd.DataFrame()

def ampl_solve(Var, individual, i, j, e, o, rpd, con, p, mi, fail):
    
    ampl.reset()
    ampl.eval("option rel_boundtol 1e-3;")
    
    ampl.read(r"/home/ba8641/Model/ABP.mod")
    
    #Treat Data
    ampl.set_data(vaccines,"V") if o[-1] != 'Y' else ampl.set_data(au_vaccines,"V")
    
    ampl.set_data(con,"M")
    
    #Create antigen table
    ampl.set_data(ant.drop('V1', axis=1),"A") if o[-1] != 'Y' else ampl.set_data(au_ant.drop('V1', axis=1),"A")
    if o[-1] != 'Y':
        k = pd.DataFrame(ant['V1'])
        k.index = ant['V1']
    else:
        k = pd.DataFrame(au_ant['V1'])
        k.index = au_ant['V1']

    original_stdout = sys.stdout
    with open('data.dat','w') as f:
        sys.stdout = f
        for a in range(len(ant)):
            if o[-1]!='Y':
                st = 'set V1['+ ant.index[a] + '] := ' 
            else:
                st = 'set V1['+ au_ant.index[a] + '] := '
            for b in ant.iloc[a,0]:
                st+= str(b) + ' '
            st += ';'
            print(st)
        sys.stdout = original_stdout
    
    ampl.read_data('data.dat')
    ampl.get_parameter('r').set_values(rpd.unstack())
    
    ampl.get_parameter('c').set_values(p.unstack())
    
    #mi.loc[:,:]=0
    ampl.get_parameter('u').set_values(mi.unstack())
    
    # Solve
    ampl.eval("solve PO;")
    #ampl.solve("PO")
    solve_result = ampl.get_value("solve_result")
    if solve_result != "solved":
        pd.concat([inf, pd.DataFrame([i,j,e,o]).transpose()])
        return Var, individual
        print("Failed to solve (solve_result: {})".format(solve_result))

    # Get objective entity by AMPL name
    tss = ampl.get_objective("TSSS").value()
    # Print it
    #print("Objective is:", tss)
    ampl.get_parameter('XC').set_values(ampl.get_variable("X").get_values())
    


    # Get the values of the variable Buy in a dataframe object
    s=pd.DataFrame([tss],columns = ['TSS'])
    #v = ampl.get_variable("X").get_values().to_pandas()
    v = ampl.get_parameter("XC").get_values().to_pandas()
    v.columns = ['X']
    t=v.copy()
    f = ampl.get_variable("F").get_values().to_pandas()
    fail = pd.concat([fail,f])
    
    t.index = [x[1] for x in v.index]
    if o[-1] != 'Y':
        vaccines['k']=(vaccines.loc[:,'k']-t.loc[:,'X']).astype(int)
    else:
        au_vaccines['k']=(au_vaccines.loc[:,'k']-t.loc[:,'X']).astype(int)
   
    ampl.eval("reset data Y;")
    ampl.eval("reset data O;")
    ampl.eval("unfix Y;")
    ampl.eval("unfix O;")
    ampl.eval("solve PTw;")
    tcs = ampl.get_objective("TCSS").value()
    s['TCS'] = tcs
    x = ampl.get_variable("Y").get_values().to_pandas()
    x.columns = ['Yl']
    v = pd.concat([v,x], axis=1)
    f = ampl.get_parameter("u").get_values().to_pandas()
    f.columns = ['u']
    x = ampl.get_variable("O").get_values().to_pandas()
    x.columns = ['Ol']
    v = pd.concat([v,f,x],axis=1)
    
    ampl.eval("solve PTh;")
    tpf = ampl.get_objective("TPFS").value()
    s['TPF'] = tpf
    x = ampl.get_variable("Y").get_values().to_pandas()
    x.columns = ['Yh']
    v = pd.concat([v,x], axis=1)
    f = ampl.get_parameter("r").get_values().to_pandas()
    f.columns =['r']
    x = ampl.get_variable("O").get_values().to_pandas()
    x.columns = ['Oh']
    v = pd.concat([v,f,x], axis=1)
    
    v['Market'] = s['Market'] = con.index[0]
    v['Exp'] = s['Exp'] = e+1
    v['Min Price Rate'] = s['Min Price Rate'] = i
    v['RP uncertainty'] = s['RP uncertainty'] = j/100
    v['Markets'] = s['Markets'] = o
    v['Commitment'] = s['Commitment'] = p.iloc[-5,0]
    
    Var = pd.concat([Var,v])
    individual = pd.concat([individual, s])
    
    return Var, individual, fail

c = con.copy()
c_au=con_au.copy()
r=rp.copy()
r_au = rp_au.copy()
m = min_pr.copy()
m_au = min_pr_au.copy()
t = pd.DataFrame()

def solve_order(drops, j, e, name, rpk, co, mi, i = 0, commi = 0):
    rpd = rpk.copy()
    co = co.copy()
    mi = mi.copy()
    p = rpd.copy()
    p.loc[:,:] = 0
    p.iloc[-5:,:] = commi
    for d in drops:
        co = co.drop(d)
        rpd = rpd.drop(d, axis=1, level = 0)
        p = p.drop(d, axis=1, level = 0)
        mi = mi.drop(d, axis = 1, level = 0)
    
    #vaccines.loc[vaccines['u'].to_list()>rpd.min(axis=1).values,'u'] = rpd.min(axis=1).loc[vaccines['u'].to_list()>rpd.min(axis=1).values]
    #else:
    #    au_vaccines.loc[au_vaccines['u'].to_list()>rpd.min(axis=1).values,'u'] = rpd.min(axis=1).loc[au_vaccines['u'].to_list()>rpd.min(axis=1).values]
    return ampl_solve(Var,summary,i,j,e, name,rpd,co,p, mi, fail)


#vaccines, au_vaccines, min_vac = MakeVaccines(1)
vaccines = vaccines.drop('Corrected Price', axis = 1)
au_vaccines = au_vaccines.drop('Corrected Price', axis = 1)
vaccines.index.name = au_vaccines.index.name = min_pr.index.name = 'V'
vac=vaccines.copy()
au_vacc = au_vaccines.copy()
for j in unc_high_list:
    for e in range(exp_to):
        min_pr = m.copy()
        m_au = min_pr_au.copy()
        t = r.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
        t_au = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
        t[t<min_pr.drop(min_pr.index[-5:])] = min_pr.drop(min_pr.index[-5:])[t<min_pr.drop(min_pr.index[-5:])]
        t_au[t_au<m_au] = m_au[t_au<m_au]
        ########################################################
        #Do without AU
        #########################################################
        vaccines = vac.copy()
        rp = t.copy()
        min_pr = min_pr.drop(min_pr.index[-5:])
        #PAHO-Gavi order
        Var, summary, fail = solve_order(['Gavi'],j,e, 'PAHO-Gavi',rp, c, min_pr)
        #Do Gavi
        Var, summary, fail = solve_order(['PAHO'], j,e, 'PAHO-Gavi',rp, c, min_pr)
        
        #Gavi-PAHO order
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['PAHO'],j,e, 'Gavi-PAHO',rp, c, min_pr)
        #Do Gavi
        Var, summary, fail = solve_order(['Gavi'],j,e, 'Gavi-PAHO',rp, c, min_pr)
        
        ################################################################
        #With AU, no production
        ###############################################################
        rp = t_au.copy()
        rp = rp.drop(rp.index[-5:])
        m_au = m_au.drop(m_au.index[-5:])
        #PAHO-Gavi-AU order
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'PAHO-Gavi-AU-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','AU'], j,e, 'PAHO-Gavi-AU-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'PAHO-Gavi-AU-N',rp, c_au, m_au)
        #PAHO-AU-Gavi
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'PAHO-AU-Gavi-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','Gavi'],j,e, 'PAHO-AU-Gavi-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'PAHO-AU-Gavi-N',rp, c_au, m_au)
        #Gavi-PAHO-AU order
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'Gavi-PAHO-AU-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['Gavi','AU'], j,e, 'Gavi-PAHO-AU-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'Gavi-PAHO-AU-N',rp, c_au, m_au)
        #Gavi-AU-PAHO
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'Gavi-AU-PAHO-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','Gavi'],j,e, 'Gavi-AU-PAHO-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'Gavi-AU-PAHO-N',rp, c_au, m_au)
        #AU-Gavi-PAHO order
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['PAHO','Gavi'],j,e, 'AU-Gavi-PAHO-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'AU-Gavi-PAHO-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'AU-Gavi-PAHO-N',rp, c_au, m_au)
        #AU-PAHO-Gavi
        vaccines['k'] = vac['k']
        Var, summary, fail = solve_order(['PAHO','Gavi'],j,e, 'AU-PAHO-Gavi-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'AU-PAHO-Gavi-N',rp, c_au, m_au)
        Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'AU-PAHO-Gavi-N',rp, c_au, m_au)
        
        ################################################################
        #With AU, AND production
        ###############################################################
        
        rp=t_au.copy()
        #PAHO-Gavi-AU order
        for alpha in min_price_rate:
            for commi in commitment:
                au_vaccines['k'] = au_vacc['k']
                for ind in au_vac:
                    min_pr_au.loc[ind,:] = alpha*min_pr_au.loc[ind[3:],:]

                Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'PAHO-Gavi-AU-Y',rp, c_au, min_pr_au, alpha)
                Var, summary, fail = solve_order(['PAHO','AU'], j,e, 'PAHO-Gavi-AU-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'PAHO-Gavi-AU-Y',rp, c_au, min_pr_au,alpha,commi)
                #PAHO-AU-Gavi
                au_vaccines['k'] = au_vacc['k']
                Var, summary, fail = solve_order(['Gavi','AU'],j,e, 'PAHO-AU-Gavi-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'PAHO-AU-Gavi-Y',rp, c_au, min_pr_au,alpha,commi)
                Var, summary, fail = solve_order(['PAHO','AU'], j,e, 'PAHO-AU-Gavi-Y',rp, c_au, min_pr_au,alpha)
                #Gavi-PAHO-AU order
                au_vaccines['k'] = au_vacc['k']
                Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'Gavi-PAHO-AU-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['Gavi','AU'], j,e, 'Gavi-PAHO-AU-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'Gavi-PAHO-AU-Y',rp, c_au, min_pr_au,alpha,commi)
                #Gavi-AU-PAHO
                au_vaccines['k'] = au_vacc['k']
                Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'Gavi-AU-PAHO-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'Gavi-AU-PAHO-Y',rp, c_au, min_pr_au,alpha,commi)
                Var, summary, fail = solve_order(['Gavi','AU'], j,e, 'Gavi-AU-PAHO-Y',rp, c_au, min_pr_au,alpha)
                #AU-Gavi-PAHO order
                au_vaccines['k'] = au_vacc['k']
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'AU-Gavi-PAHO-Y',rp, c_au, min_pr_au,alpha,commi)
                Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'AU-Gavi-PAHO-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['Gavi','AU'], j,e, 'AU-Gavi-PAHO-Y',rp, c_au, min_pr_au,alpha)
                #AU-PAHO-Gavi
                au_vaccines['k'] = au_vacc['k']
                Var, summary, fail = solve_order(['PAHO','Gavi'], j,e, 'AU-PAHO-Gavi-Y',rp, c_au, min_pr_au,alpha,commi)
                Var, summary, fail = solve_order(['Gavi','AU'], j,e, 'AU-PAHO-Gavi-Y',rp, c_au, min_pr_au,alpha)
                Var, summary, fail = solve_order(['PAHO','AU'],j,e, 'AU-PAHO-Gavi-Y',rp, c_au, min_pr_au,alpha)                                
        
            '''
            #AU-Gavi-PAHO
            con = c_au.drop('Gavi')
            con = con.drop('PAHO')
            rp = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
            t = rp
            rp = t.drop('Gavi', axis=1)
            rp = rp.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            vaccines['k']=au_vaccines['k'] = gen_supply
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            #Do Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            
            #AU-PAHO-Gavi
            con = c_au.drop('Gavi')
            con = con.drop('PAHO')
            rp = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
            t = rp
            rp = t.drop('Gavi', axis=1)
            rp = rp.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            vaccines['k']=au_vaccines['k'] = gen_supply
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            #Do PAHO
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            #Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            
            #With AU and new manufacturers
            con = c_au.drop('Gavi')
            con = con.drop('PAHO')
            rp = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
            t = rp
            rp = t.drop('Gavi', axis=1)
            rp = rp.drop('PAHO', axis=1)
            au_vaccines.loc[au_vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[au_vaccines['u'].to_list()>rp.min(axis=1).values]
            vaccines['k']=au_vaccines['k'] = gen_supply
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')
            #Do Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1)
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1)
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            au_vaccines.loc[au_vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[au_vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary, fail = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')'''
            
            
#################################################################
#Save results
#################################################################

avg=summary.groupby(['Market','Min Price Rate', 'RP uncertainty', 'Markets']).mean().drop('Exp', axis=1)
summary.to_csv('Extended Analysis/summary.csv')
Var.to_csv("Extended Analysis/var.csv")
inf.to_csv("Extended Analysis/inf.csv")
fail.to_csv("Extended Analysis/fail.csv")
