# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:08:04 2023

@author: ba8641
"""
from amplpy import AMPL, modules
modules.load() # load all modules
ampl = AMPL() # instantiate AMPL object



################################################################
#ExaAMPLe
##################################################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os


def main():
    # You can install amplpy with "python -m pip install amplpy"
    from amplpy import AMPL

    # Create an AMPL instance
    ampl = AMPL()
    
    ampl.reset()
    
    ampl.read(r"C:\Users\BrunoAlvesMaciel\Dropbox\BrunoAlvesMaciel\RIT\Au-Project\Model\ABP.mod")
    
    #Treat Data
    vaccines.index.name = 'V'
    vaccines.columns = ['u']
    ampl.set_data(vaccines,"V")
    con.index = con['Market']
    con = con.drop('Market', axis = 1)
    con.columns = ['g', 'd']
    ampl.set_data(con,"M")
    
    #Create antigen table
    ampl.set_data(ant.drop('V1', axis=1),"A")
    k = pd.DataFrame(ant['V1'])
    k.index = ant['V1']
    #t = 0
    #for i in ant.index:
    #    ampl.set_data(k.drop('V1', axis=1)[t],"V1["+i+']')
    #    t+=1
    #st = ''
    original_stdout = sys.stdout
    with open('data.dat','w') as f:
        sys.stdout = f
        for a in range(len(ant)):
            st = 'set V1['+ ant.index[a] + '] := '
            for b in ant.iloc[a,0]:
                st+= str(b) + ' '
            st += ';'
            print(st)
        sys.stdout = original_stdout
    
    ampl.read_data('data.dat')
    ampl.get_parameter('r').set_values(rp.unstack())
    
    # Solve
    ampl.option["solver"] = "highs"
    ampl.solve()
    solve_result = ampl.get_value("solve_result")
    if solve_result != "solved":
        raise Exception("Failed to solve (solve_result: {})".format(solve_result))

    # Get objective entity by AMPL name
    tss = ampl.get_objective("TSS").value()
    # Print it
    print("Objective is:", tss)


    # Get the values of the variable Buy in a dataframe object
    X = ampl.get_variable("X").get_values().to_pandas()
    Y = ampl.get_variable("Y").get_values().to_pandas()
    # Print as pandas dataframe


try:
    main()
except Exception as e:
    print(e)
    raise