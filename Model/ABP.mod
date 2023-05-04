set V;
set A;
set M;
set V1{A};
set Q{V} default {}; 
set QQ:= union {v in V} Q[v];
set N{QQ} within V;  

param r{M,V};					#Reservation price
param d{M};						#Doses demanded per antigen
param s{V} default 10e10;		#Total supply
param k{v in V} default s[v]; 	#Remaining supply
param g{M};						#GNI per capita
param u{M,V} default 0;					#Minimum price/Cost per dose?
param c{M,V} default 0;			#Commitment of market m to vaccine v.
param XC{M,V} default 0;
param alpha{m in M,v in V} default (r[m,v]/u[m,v] - 1);

var X{M,V} >= 0;
var Y{m in M, v in V} >= u[m,v];
var F{M,A} >= 0;				#Antigen deficit
var O{M,V} >= 0;				#Overpayment from RP

#maximize TSS: sum{v in V,m in M}r[m,v]*X[m,v] + sum{v in V, m in M}(r[m,v]-Y[m,v])- sum{a in A, m in M} F[m,a];
maximize TSS: sum{v in V, m in M}(r[m,v]-u[m,v])*X[m,v] - 10e10*sum{a in A,m in M}F[m,a];
maximize TCS: sum{v in V, m in M} (r[m,v]-Y[m,v])*XC[m,v] - 10e10*sum{v in V,m in M}O[m,v];
maximize TPF: sum{v in V, m in M} (Y[m,v]-u[m,v])*XC[m,v] - 10e10*sum{v in V,m in M}O[m,v];

maximize TSSS: sum{v in V, m in M}(r[m,v]-u[m,v])*X[m,v];
maximize TCSS: sum{v in V, m in M} (r[m,v]-Y[m,v])*XC[m,v];
maximize TPFS: sum{v in V, m in M} (Y[m,v]-u[m,v])*XC[m,v];

#s.t. MultiAntigen{v in V, q in Q[v], m in M}: Y[v,m] >= sum{t in N[q]}Y[t,m];
s.t. Supply{v in V}: sum{m in M}X[m,v]<= k[v];
s.t. RP{v in V, m in M}: Y[m,v] - O[m,v] <= r[m,v];
s.t. RP2{v in V, m in M}: Y[m,v] <= r[m,v];
s.t. Dem{a in A, m in M}:sum{v in V1[a]}X[m,v] + F[m,a] = d[m];
s.t. Com{v in V, m in M:d[m]<k[v]}: X[m,v] >= c[m,v]*d[m];
s.t. Com2{v in V, m in M:d[m]>k[v]}: X[m,v] >= c[m,v]*k[v];
#s.t. Elasticity{v in V, m in M}: Y[m,v] >= u[v] + (r[m,v]-u[v])*(k[v]-X[m,v])/(k[v]+1e-10)
s.t. Elasticity{v in V, m in M:k[v]>0}: Y[m,v] >= ((k[v]-X[m,v])*alpha[m,v]+s[v])/s[v] * (u[m,v]); 

problem PO: X, Y, O, F, TSS, Supply, RP, Elasticity, Dem, Com, Com2;
#problem PO: X, F, TSS, Supply, Dem, Com, Com2;
problem PTw: Y, O, TCS, RP, Elasticity;
problem PTh: Y, O, TPF, RP, Elasticity;
