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

from amplpy import AMPL, modules
modules.load() # load all modules
ampl = AMPL() # instantiate AMPL object
ampl.option["solver"] = "highs"


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

min_price_rate = [.9,1,1.25,2]
commitment = [0,.25,.5,.75,.1]

create_file = True
main_path = r'C:\Users\BrunoAlvesMaciel\Dropbox\BrunoAlvesMaciel\RIT\Au-Project\Data'
path_regression_coeficients = ''

au_vac = ['AU_DTwP-HepB-Hib', 'AU_MR', 'AU_PCV13', 'AU_Rota'] #Vaccines offered by AU
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
                      'Hib':[['DTaP-HepB-Hib-IPV', 'DTaP-Hib-IPV', 'DTwP-HepB-Hib', 'DTwP-Hib', 'Hib']],
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

dt = pd.read_csv(main_path+'\Detailed_Demand_Forecast_data.csv')
all_countries = pd.unique(dt.iloc[:,1])
countries_2021 = pd.unique(dt[dt['Year']==2021]['Country (group)'])
all_vaccines = pd.unique(dt['Vaccine'])
vaccines_2021 = pd.unique(dt[dt['Year']==2021]['Vaccine'])

dat = dt[(dt['Year']==2021) & (dt['Current WB Status & Current Gavi Eligibility'].str.contains('Gavi'))]

ct = pd.read_csv(main_path+'\Country-data.csv')
ct['PAHO'] = (ct['WHO Region']=='AMR') & (~ct['Country'].isin(['USA', 'Canada']))
ct = ct.sort_values('Country')

pt = pd.read_csv(main_path+'\Prices.csv')
prt = pt[(pt['Year (Price)']==2021) & (pt['Source']!='V3P') & (pt['Price Tier (group) (4.2)']!='USA')].drop(['Price Tier (group) (4.2)','Year (Price)'],axis=1)


pt_PAHO = pd.read_csv(main_path+'\99_PAHO_Pricing_data.csv')
pt_PAHO = pt_PAHO[(pt_PAHO['Year']==2021) & (pt_PAHO['Avg. Price'].notnull())].drop(['Vial Size','% Difference in Avg. Price Color','Year'], axis = 1)

pt_PAHO['Avg. Price'] = pt_PAHO['Avg. Price'].replace('[\$,]', '', regex=True).astype(float) #Already price/dose
pt_PAHO = pt_PAHO.groupby('Vaccine').min()
pt_PAHO.reset_index(inplace=True)
pt_PAHO = pt_PAHO.rename(columns = {'index':'Vaccine'})

dat['Price']=-1
#ct_PAHO = dat[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country']))]
dat.loc[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country'])),'Price'] = dat[(dat['Country (group)'].isin(ct[ct['PAHO']]['Country']))].merge(pt_PAHO[['Vaccine','Avg. Price']], left_on='Vaccine Sub-type', right_on='Vaccine', how='left')['Avg. Price'].values
#ct_PAHO.merge(pt_PAHO[['Vaccine','Avg. Price']], on='Vaccine', how='left')

dat = dat.rename(columns={'Current WB Status & Current Gavi Eligibility':'Gavi', 'Country (group)':'Country', 'Total Required Supply': 'Demand'})

#UNICEF prices
pt_UN = pd.read_csv(main_path+'\910_UNICEF_Pricing_data.csv')
pt_UN = pt_UN[(pt_UN['Year']==2021) & (pt_UN['Min. Price'].notnull())].drop(['Year','% Difference in Min. Price Color','Company  ', 'Vial Size'], axis = 1)
pt_UN['Min. Price'] = pt_UN['Min. Price'].replace('[\$,]', '', regex=True).astype(float) #Price per dose
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
    
#Creates a python frame with information on the countries and their weights which will have to be multiplied by the 
#average price per vaccine in a single price market to dissagregate it to the country levels
'''Frame=pd.DataFrame(v[:,ind_interest].real/np.average(v[:, ind_interest].real), index=[countries])
Frame.columns=["Weight"]
Frame['Country']=countries
Frame.to_csv("FrameRP.csv")'''
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
    auv = vac.copy()
    auv.loc['AU_DTwP-HepB-Hib','Corrected Price'] = alpha*auv.loc['DTwP-HepB-Hib','Corrected Price']
    auv.loc['AU_MR', 'Corrected Price'] = alpha*auv.loc['MR','Corrected Price']
    auv.loc['AU_PCV13', 'Corrected Price'] = alpha*auv.loc['PCV13','Corrected Price']
    auv.loc['AU_Rota', 'Corrected Price'] = alpha*auv.loc['Rota','Corrected Price']
    return vac, auv
    
vaccines, au_vaccines = MakeVaccines(1)

coef.to_csv("CoefRP.csv")
coef_au.to_csv("CoefRP_AU.csv")
vaccines.to_csv("Baseline.csv")

rp = pd.DataFrame(np.matmul(vaccines[['Corrected Price']], coef[['Weight']].transpose()))
rp.columns = coef.index

rp_au = pd.DataFrame(np.matmul(au_vaccines[['Corrected Price']], coef_au[['Weight']].transpose()))
rp_au.columns = coef_au.index


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

comp = [dem_v.index.values[z] in au_vaccines.index.values for z in range(len(dem_v.index))]
comp[-3] = comp[-2] = comp[-1] = True
au_vaccines['k'] = dem_v.loc[comp,'Demand']
au_vaccines.loc[au_vaccines['k'].isna(),'k'] = au_vaccines.loc[~au_vaccines['k'].isna(),'k'].mean()

Var = pd.DataFrame()
individual = pd.DataFrame()
summary = pd.DataFrame()
inf = pd.DataFrame()

def ampl_solve(Var, individual, i, j, e, o, rpd, con, p):
    
    ampl.reset()
    
    ampl.read(r"C:\Users\BrunoAlvesMaciel\Dropbox\BrunoAlvesMaciel\RIT\Au-Project\Model\ABP.mod")
    
    #Treat Data
    ampl.set_data(vaccines,"V") if o != 'AU-Gavi-PAHO-NewMan' else ampl.set_data(au_vaccines,"V")
    
    ampl.set_data(con,"M")
    
    #Create antigen table
    ampl.set_data(ant.drop('V1', axis=1),"A") if o != 'AU-Gavi-PAHO-NewMan' else ampl.set_data(au_ant.drop('V1', axis=1),"A")
    if o != 'AU-Gavi-PAHO-NewMan':
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
    
    # Solve
    ampl.eval("solve PO;")
    #ampl.solve("PO")
    solve_result = ampl.get_value("solve_result")
    if solve_result != "solved":
        pd.concat([inf, pd.DataFrame([i,j,e,o]).transpose()])
        return Var, individual
        print("Failed to solve (solve_result: {})".format(solve_result))

    # Get objective entity by AMPL name
    tss = ampl.get_objective("TSS").value()
    # Print it
    #print("Objective is:", tss)
    ampl.get_parameter('XC').set_values(ampl.get_variable("X").get_values())
    


    # Get the values of the variable Buy in a dataframe object
    s=pd.DataFrame([tss],columns = ['TSS'])
    v= ampl.get_variable("X").get_values().to_pandas()
    
    v.index = [x[1] for x in v.index]
    vaccines.loc[:,'k']=vaccines.loc[:,'k']-v.loc[:,'X.val']
   
    
    ampl.eval("solve PTw;")
    tcs = ampl.get_objective("TCS").value()
    s['TCS'] = tcs
    v['Yl'] = ampl.get_variable("Y").get_values().to_pandas()
    v['Ol'] = ampl.get_variable("Y").get_values().to_pandas()
    
    ampl.eval("solve PTh;")
    tpf = ampl.get_objective("TPF").value()
    s['TPF'] = tpf
    v['Yh'] = ampl.get_variable("Y").get_values().to_pandas()
    v['Oh'] = ampl.get_variable("Y").get_values().to_pandas()
    
    
    v['Market'] = s['Market'] = con.index[0]
    v['Exp'] = s['Exp'] = e+1
    v['Min Price Rate'] = s['Min Price Rate'] = i
    v['RP uncertainty'] = s['RP uncertainty'] = j/100
    v['Markets'] = s['Markets'] = o
    
    Var = pd.concat([Var,v])
    individual = pd.concat([individual, s])
    
    return Var, individual

c = con.copy()
c_au=con_au.copy()
r=rp_au.copy()
t = pd.DataFrame()

def solve_order(drops, j, e, name, rpk, i = 0, commi = 0):
    rpd = rpk.copy()
    co = c_au.copy()
    p = rpd.copy()
    p.loc[:,:] = 0
    p.iloc[-4:,:] = commi
    for d in drops:
        co = co.drop(d)
        rpd = rpd.drop(d, axis=1, level = 0)
        p = p.drop(d, axis=1, level = 0)
    
    vaccines.loc[vaccines['u'].to_list()>rpd.min(axis=1).values,'u'] = rpd.min(axis=1).loc[vaccines['u'].to_list()>rpd.min(axis=1).values]
    #else:
    #    au_vaccines.loc[au_vaccines['u'].to_list()>rpd.min(axis=1).values,'u'] = rpd.min(axis=1).loc[au_vaccines['u'].to_list()>rpd.min(axis=1).values]
    return ampl_solve(Var,summary,i,j,e, name,rpd,co,p)


vaccines, au_vaccines = MakeVaccines(1)
vaccines.index.name = 'V'
vaccines.columns = ['u']
vac=vaccines
au_vaccines.index.name = 'V'
au_vaccines.columns = ['u']
for j in unc_high_list:
    for e in range(exp_to):
        t = r.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
        ########################################################
        #Do without AU
        #########################################################
        vaccines = vac
        rp = t.drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
        #PAHO-Gavi order
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['Gavi','AU'],j,e, 'PAHO-Gavi',rp)
        #Do Gavi
        Var, summary = solve_order(['PAHO','AU'], j,e, 'PAHO-Gavi',rp)
        
        #Gavi-PAHO order
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['PAHO','AU'],j,e, 'Gavi-PAHO',rp)
        #Do Gavi
        Var, summary = solve_order(['Gavi','AU'],j,e, 'Gavi-PAHO',rp)
        
        ################################################################
        #With AU, no production
        ###############################################################
        #PAHO-Gavi-AU order
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['Gavi','AU'],j,e, 'PAHO-Gavi-AU-N',rp)
        Var, summary = solve_order(['PAHO','AU'], j,e, 'PAHO-Gavi-AU-N',rp)
        Var, summary = solve_order(['PAHO','Gavi'], j,e, 'PAHO-Gavi-AU-N',rp)
        #PAHO-AU-Gavi
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['Gavi','AU'],j,e, 'PAHO-AU-Gavi-N',rp)
        Var, summary = solve_order(['PAHO','Gavi'],j,e, 'PAHO-AU-Gavi-N',rp)
        Var, summary = solve_order(['PAHO','AU'],j,e, 'PAHO-AU-Gavi-N',rp)
        #Gavi-PAHO-AU order
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['PAHO','AU'],j,e, 'Gavi-PAHO-AU-N',rp)
        Var, summary = solve_order(['Gavi','AU'], j,e, 'Gavi-PAHO-AU-N',rp)
        Var, summary = solve_order(['PAHO','Gavi'], j,e, 'Gavi-PAHO-AU-N',rp)
        #Gavi-AU-PAHO
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['PAHO','AU'],j,e, 'Gavi-AU-PAHO-N',rp)
        Var, summary = solve_order(['PAHO','Gavi'],j,e, 'Gavi-AU-PAHO-N',rp)
        Var, summary = solve_order(['Gavi','AU'],j,e, 'Gavi-AU-PAHO-N',rp)
        #AU-Gavi-PAHO order
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['PAHO','Gavi'],j,e, 'AU-Gavi-PAHO-N',rp)
        Var, summary = solve_order(['PAHO','AU'],j,e, 'AU-Gavi-PAHO-N',rp)
        Var, summary = solve_order(['Gavi','AU'],j,e, 'AU-Gavi-PAHO-N',rp)
        #AU-PAHO-Gavi
        vaccines['k']=au_vaccines['k'] = gen_supply
        Var, summary = solve_order(['PAHO','Gavi'],j,e, 'AU-PAHO-Gavi-N',rp)
        Var, summary = solve_order(['Gavi','AU'],j,e, 'AU-PAHO-Gavi-N',rp)
        Var, summary = solve_order(['PAHO','AU'],j,e, 'AU-PAHO-Gavi-N',rp)
        
        ################################################################
        #With AU, AND production
        ###############################################################
        
        rp=t
        #PAHO-Gavi-AU order
        for alpha in min_price_rate:
            for commi in commitment:
                vaccines['k']=au_vaccines['k'] = gen_supply
                vaccines = au_vaccines.copy()
                vaccines.loc['AU_DTwP-HepB-Hib','u'] = alpha*vaccines.loc['DTwP-HepB-Hib','u']
                vaccines.loc['AU_MR', 'u'] = alpha*vaccines.loc['MR','u']
                vaccines.loc['AU_PCV13', 'u'] = alpha*vaccines.loc['PCV13','u']
                vaccines.loc['AU_Rota', 'u'] = alpha*vaccines.loc['Rota','u']
                Var, summary = solve_order(['Gavi','AU'],j,e, 'PAHO-Gavi-AU-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','AU'], j,e, 'PAHO-Gavi-AU-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'PAHO-Gavi-AU-Y',rp,alpha,commi)
                #PAHO-AU-Gavi
                vaccines['k']=au_vaccines['k'] = gen_supply
                Var, summary = solve_order(['Gavi','AU'],j,e, 'PAHO-AU-Gavi-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'PAHO-AU-Gavi-Y',rp,alpha,commi)
                Var, summary = solve_order(['PAHO','AU'], j,e, 'PAHO-AU-Gavi-Y',rp,alpha)
                #Gavi-PAHO-AU order
                vaccines['k']=au_vaccines['k'] = gen_supply
                Var, summary = solve_order(['PAHO','AU'],j,e, 'Gavi-PAHO-AU-Y',rp,alpha)
                Var, summary = solve_order(['Gavi','AU'], j,e, 'Gavi-PAHO-AU-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'Gavi-PAHO-AU-Y',rp,alpha,commi)
                #Gavi-AU-PAHO
                vaccines['k']=au_vaccines['k'] = gen_supply
                Var, summary = solve_order(['PAHO','AU'],j,e, 'Gavi-AU-PAHO-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'Gavi-AU-PAHO-Y',rp,alpha,commi)
                Var, summary = solve_order(['Gavi','AU'], j,e, 'Gavi-AU-PAHO-Y',rp,alpha)
                #AU-Gavi-PAHO order
                vaccines['k']=au_vaccines['k'] = gen_supply
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'AU-Gavi-PAHO-Y',rp,alpha,commi)
                Var, summary = solve_order(['PAHO','AU'],j,e, 'AU-Gavi-PAHO-Y',rp,alpha)
                Var, summary = solve_order(['Gavi','AU'], j,e, 'AU-Gavi-PAHO-Y',rp,alpha)
                #AU-PAHO-Gavi
                vaccines['k']=au_vaccines['k'] = gen_supply
                Var, summary = solve_order(['PAHO','Gavi'], j,e, 'AU-PAHO-Gavi-Y',rp,alpha,commi)
                Var, summary = solve_order(['Gavi','AU'], j,e, 'AU-PAHO-Gavi-Y',rp,alpha)
                Var, summary = solve_order(['PAHO','AU'],j,e, 'AU-PAHO-Gavi-Y',rp,alpha)                                
        
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
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            #Do Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-No')
            
            #AU-PAHO-Gavi
            con = c_au.drop('Gavi')
            con = con.drop('PAHO')
            rp = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
            t = rp
            rp = t.drop('Gavi', axis=1)
            rp = rp.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            vaccines['k']=au_vaccines['k'] = gen_supply
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            #Do PAHO
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            #Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1).drop(['AU_DTwP-HepB-Hib','AU_MR', 'AU_PCV13', 'AU_Rota'])
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            vaccines.loc[vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-PAHO-Gavi-No')
            
            #With AU and new manufacturers
            con = c_au.drop('Gavi')
            con = con.drop('PAHO')
            rp = r_au.applymap(lambda x: x*(1+j*(np.random.random()-.5)/100))
            t = rp
            rp = t.drop('Gavi', axis=1)
            rp = rp.drop('PAHO', axis=1)
            au_vaccines.loc[au_vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[au_vaccines['u'].to_list()>rp.min(axis=1).values]
            vaccines['k']=au_vaccines['k'] = gen_supply
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')
            #Do Gavi
            con = c_au.drop('PAHO')
            rp = t.drop('PAHO', axis=1)
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')
            con = c_au.drop('Gavi')
            rp = t.drop('Gavi', axis=1)
            con = con.drop('AU')
            rp = rp.drop('AU', axis=1)
            au_vaccines.loc[au_vaccines['u'].to_list()>rp.min(axis=1).values,'u'] = rp.min(axis=1).loc[au_vaccines['u'].to_list()>rp.min(axis=1).values]
            Var, summary = ampl_solve(Var,summary,i,j,e, 'AU-Gavi-PAHO-NewMan')'''
            
            
#################################################################
#Save results
#################################################################

avg=summary.groupby(['Market','Min Price Rate', 'RP uncertainty', 'Markets']).mean().drop('Exp', axis=1)
summary.to_csv('DataIn\summary.csv')
Var.to_csv("DataIn/var.csv")
inf.to_csv("DataIn/inf.csv")
        
#avg.boxplot('TSS',by=['Markets', 'Market', 'Min Price Rate'])
#avg[avg.index.get_level_values('Min Price Rate')==.2].boxplot('TSS',by=['Markets', 'Market'], fontsize = 6, rot=90)


##################################################################################
#Generate Data files ###########################################################
######################################################################################


'''
#Define data creation function
def data_creation(exp,m,n_l,n_h,i,num):
    num_y = 20
    interest_rate = i

    uncertainty_low = 1 + n_l/100.0
    uncertainty_high = 1 + n_h/100.0
    GNI_births_data = dat['Demand']
    Reg_coef_data = pd.read_csv(path_regression_coeficients)
    
    markets = len(pd.unique(dat['Country']))
    GNI_limits_list = [max(ct.iloc[:,4])+1,HIC_GNI, UMIC_GNI, LMIC_GNI, 0]
    #GNI_limits_list_opt = [max(GNI_births_data['GNI per capita']),4125,0]
    
    #if num_HIC>=1 and num_UMIC>=1 and num_LMIC>=1 and num_LIC>=1:
    #    limits = limits_from_segments(GNI_limits_list,[num_HIC,num_UMIC,num_LMIC,num_LIC])
    #else:
    #    limits = limits_from_segments(GNI_limits_list_opt,[num_HIC,num_LIC])
    
    #new_data = GNI_AnnualBirths(GNI_births_data,limits)
    #segmentation_dict = new_data[0]
    countries = pd.DataFrame(pd.unique(dat['Country']))
    #countries = new_data[1][new_data[1].columns[2:]]
    #countries.index = range(1,countries.shape[0]+1)
    
    GNI = ct[ct['Country'].isin(pd.unique(dat['Country']).tolist())][['GNI per Capita (USD)']]
    demand = ct[ct['Country'].isin(pd.unique(dat['Country']).tolist())][['Birth Cohort']]
    
    #GNI_L = pd.DataFrame.from_dict(segmentation_dict, orient='index', dtype=None)
    GNI_L = pd.concat(GNI, demand)
    GNI_L.columns = ['gni_p','l:=']
    GNI_L.index.name = 'param:'
    #GNI_L.to_csv(main_path+r'\gni_p and l.dat',sep='\t')
    
    Reg_coef_data["C"] = (Reg_coef_data["RD_cost"]*ROI)/(1-(1+interest_rate/100.0)**-num_y)
    Reg_coef_data["C"] = Reg_coef_data["C"].round(0)*interest_rate/100.0
    
    
    #for key in segmentation_dict:
   #     if key == len(segmentation_dict):
   #         Reg_coef_data['%s:='%key] = Reg_coef_data["Intercept"] + Reg_coef_data["GNI_coef"]*segmentation_dict[key][0]
    #    else:
     #       Reg_coef_data[key] = Reg_coef_data["Intercept"] + Reg_coef_data["GNI_coef"]*segmentation_dict[key][0]
    
    C_b = Reg_coef_data[Reg_coef_data.columns[4:5]]
    #C_b.index.name = 'param C'
    C_b.columns = [':=']
    #C_b.to_csv(main_path+r'\Cb.dat',sep='\t')
    
    R_b_m = Reg_coef_data[Reg_coef_data.columns[5:]]
    #R_b_m.index.name = 'param R:'
    
    
    R_b_m_list = R_b_m.values.tolist()
   # print R_b_m_list
    
    for i in range(len(R_b_m_list)):
        for j in range(len(R_b_m_list[i])):
            if j == 0:
                temp3 = 0
                if (i+1) in comb_vaccines_dict:
                    temp1 = []
                    for k in comb_vaccines_dict[i+1]:
                        temp2 = 0
                        for l in comb_vaccines_dict[i+1][comb_vaccines_dict[i+1].index(k)]:
                            temp2 +=  R_b_m_list[l-1][j]
                        temp1.append(temp2)
                    #print "temp1 = ",temp1
                    temp3 = max(temp1)
                R_b_m_list[i][j] = [R_b_m_list[i][j]*uncertainty_low,np.random.uniform(max(temp3,R_b_m_list[i][j]*uncertainty_low),R_b_m_list[i][j]*uncertainty_high),R_b_m_list[i][j]*uncertainty_high,temp3]
       #         R_b_m_list[i][j] = [round_function(R_b_m_list[i][j][0]),round_function(R_b_m_list[i][j][1]),round_function(R_b_m_list[i][j][2]),round_function(R_b_m_list[i][j][3])]
                #print i,j,R_b_m_list[i][j]
                R_b_m_list[i][j] = R_b_m_list[i][j][1]
                
            
            if j > 0:
                if (i+1) in comb_vaccines_dict:
                    temp1 = []
                    for k in comb_vaccines_dict[i+1]:
                        temp2 = 0
                        for l in comb_vaccines_dict[i+1][comb_vaccines_dict[i+1].index(k)]:
                            temp2 +=  R_b_m_list[l-1][j]
                        temp1.append(temp2)
                    temp3 = max(temp1)
                R_b_m_list[i][j] = [R_b_m_list[i][j]*uncertainty_low,np.random.uniform(max(temp3,R_b_m_list[i][j]*uncertainty_low),min(R_b_m_list[i][j]*uncertainty_high,R_b_m_list[i][j-1])),R_b_m_list[i][j]*uncertainty_high,temp3]
        #        R_b_m_list[i][j] = [round_function(R_b_m_list[i][j][0]),round_function(R_b_m_list[i][j][1]),round_function(R_b_m_list[i][j][2]),round_function(R_b_m_list[i][j][3])]
                #print i,j,R_b_m_list[i][j]
                R_b_m_list[i][j] = R_b_m_list[i][j][1]

            #if i == 0 or i == 6: 
             #   print "Actual= ",R_b_m_list[i]     
    #print R_b_m_list
    header = []
    for i in range(m):
        if i == range(m)[-1]:
            header.append(str(i+1)+":=")
            continue
        header.append(str(i+1))
    #print header
    #R_b_m_list.insert(0, header)
    #print R_b_m_list
    
    R_b_m = pd.DataFrame(R_b_m_list, columns = header) 
    R_b_m.index += 1
    #print R_b_m1
    #print R_b_m
    #R_b_m = R_b_m*np.random.uniform(uncertainty_low, uncertainty_high,size=(R_b_m.shape[0],R_b_m.shape[1]))
    #R_b_m.to_csv(main_path+r'\Rbm.dat',sep='\t')
    #print R_b_m
    
    D_b_m = Reg_coef_data[Reg_coef_data.columns[3:4]]
    #D_b_m.index.name = 'param D:'
    D_b_m.columns = [1]
    for n in range (2,markets+1):
        if n == markets:
            D_b_m['%s:='%n] = Reg_coef_data['D']
        else:
            D_b_m[n] = Reg_coef_data['D']
    #D_b_m.to_csv(main_path+r'\Dbm.dat',sep='\t')
    
    d_a_m = pd.DataFrame.from_dict({1:[3,3,3,3,2,2]}, orient='columns', dtype=None)
    #d_a_m.index.name = 'param d:'
    d_a_m.index += 1 
    for n in range (2,markets+1):
        if n == markets:
            d_a_m['%s:='%n] = d_a_m[1]
        else:
            d_a_m[n] = d_a_m[1]
    #d_a_m.to_csv(main_path+r'\dam.dat',sep='\t')
    
    if create_file == True:
        file_name = 'Test_m%s_ul%s_uh%s_i%s_e%s.dat'%(markets,uncertainty_low,uncertainty_high,1+interest_rate/100.0,exp)
        with open(main_path+'\Data_generation\data_original.dat', 'r') as input_file, open(main_path+'\DataIn'+'\%s'%file_name, 'w') as output_file:
            for line in input_file:
                if 'param n_markets  := ' in line:
                    output_file.write('param n_markets  := %s;\n'%markets)
                    output_file.write('param uncertainty_low  := %s;\n'%uncertainty_low)
                    output_file.write('param uncertainty_high  := %s;\n'%uncertainty_high)
                    output_file.write('param interest_rate  := %s;\n'%(1+interest_rate/100.0))
                else:
                    output_file.write(line)
    
        with open(main_path+'\DataIn'+'\%s'%file_name, 'a') as data:
            mode = csv.writer(data, delimiter=' ')
            mode.writerow('')
            GNI_L.to_csv(data,sep='\t')
            mode.writerow(';')
            C_b.index.name = 'param C'
            C_b.to_csv(data,sep='\t')
            mode.writerow(';')
            R_b_m.index.name = 'param R:'
            R_b_m.to_csv(data,sep='\t')
            mode.writerow(';')
            D_b_m.index.name = 'param D:'
            D_b_m.to_csv(data,sep='\t')   
            mode.writerow(';') 
            d_a_m.index.name = 'param d:'
            d_a_m.to_csv(data,sep='\t')   
            mode.writerow(';') 
            countries.index.name = 'param Countries:'
            countries.to_csv(data,sep='\t')   
            mode.writerow(';') 
            
        with open(main_path+'\DataIn'+'\%s'%file_name, 'a') as data:
            mode = csv.writer(data, delimiter=' ')

        with open(main_path+'\ExpFileDetailsTestFL.txt', 'ab') as data:
            mode = csv.writer(data, delimiter=' ',quotechar=' ',lineterminator='')
            mode.writerow(['%s'%file_name+'\n'])

with open(main_path+'\DataIn\ExpDetails.txt', 'r+') as output_file:
    mode = csv.writer(output_file, delimiter=' ',quotechar=' ',lineterminator='')
    mode.writerow(['%s\n'%number])

for mar in n_ent_list:
    n_markets = 1
    for i in exp_list:
        for unc in unc_high_list:
            unc_high = unc
            unc_low = 0.0
            for int_r in interest_rate_list:
                interest_rate = int_r
                data_creation(i,n_markets,unc_low,unc_high,interest_rate,number)

with open(main_path+'\ExpDetails.txt', 'r+') as data:
            mode = csv.writer(data, delimiter=' ',quotechar=' ',lineterminator='')
            mode.writerow([r'"%s\"'%main_path+'\n'])
'''