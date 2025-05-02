#!/usr/bin/env python3
#coding:utf-8
import time
import csv
import numpy as np
import pandas as pd
import scipy.sparse as sp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# Read edges from the edge file
def ReadEdges(EdgeFile, size):
    df = pd.read_csv(EdgeFile, skiprows=2)
    row = df['node1'].values
    col = df['node2'].values
    data = [1] * len(row)  
    A = sp.coo_matrix((data, (row, col)), shape=(size, size), dtype=np.int32).tocsr()  # Get the adjacency matrix
    A = A + A.T
    return A

# Calculate the noisy Adjacency matrix(RR) --> ns_A
def CalcNSA(A, p):   
    n = A.shape[0] 
    rand_matrix = np.triu(np.random.rand(n, n), 1) 
    flip_mask = rand_matrix < p
    ns_A = A.copy().toarray()
    ns_A[flip_mask] = 1 - ns_A[flip_mask]
    return sp.triu(ns_A, k=1, format='csr')

# Add Laplace noise
def AddLaplaceNS(data, mu, b):
    laplace_noise = np.random.laplace(mu, b, len(data))          # 为原始数据添加Laplace噪声
    return data+laplace_noise

# Calculate Assort_Coeff --> r
def CalcAC(A, deg):
    EdgeNum = sum(deg)/2
    idx = A.nonzero()
    k1 = np.array([deg[i] for i in idx[0]])
    k2 = np.array([deg[i] for i in idx[1]])
    ru = (np.dot(k1, k2)/2/EdgeNum-(sum(k1+k2)/4/EdgeNum)**2)
    rd = (sum(k1*k1+k2*k2)/4/EdgeNum-(sum(k1+k2)/4/EdgeNum)**2)
    if rd == 0:
        return 0
    return ru/rd

# Calculate Assort_Coeff_Numerator --> rn
def CalcACN(A, deg):
    EdgeNum = sum(deg)/2
    idx = A.nonzero()
    k1 = np.array([deg[i] for i in idx[0]])
    k2 = np.array([deg[i] for i in idx[1]])
    ru = (np.dot(k1, k2)/2/EdgeNum-(sum(k1+k2)/4/EdgeNum)**2)
    return ru

# Read the numerical upper-bound on epsilon in the local randomizer [Feldman+, FOCS21]
def ReadNumericalBound(NumerBoundFile, n, eps):
    with open(NumerBoundFile, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if float(row[2]) <= eps:
                break
    epsL = float(row[0])
    return epsL

# Calculate the closed-form upper bound on epsilon in local randomizer [Feldman+, FOCS21]
def CalcEpsL(n, eps):
    epsL_min = 0.001
    epsL_sht = 0.001
    epsL = epsL_max = np.log(n/(16*np.log(2/delta)))
    while epsL>=epsL_min:
        alpha = np.exp(epsL)
        x1 = (alpha-1)/(alpha+1)
        x2 = 8*np.sqrt(alpha*np.log(4/delta))/np.sqrt(n)
        x3 = 8*alpha/n
        eps_tmp = np.log(1+x1*(x2+x3))
        if eps_tmp <= eps:
            break
        epsL -= epsL_sht
    return epsL

# Local_{ru}: Calculate Assort_Coeff_Numerator in the non-interactive local model(RR+Lap) 
def CalcNLocACN(A, deg, epsRR, epsLap):
    NodeNum = A.shape[0]
    EdgeNum = sum(deg)/2
    # Flip probability --> p
    p = 1/(np.exp(epsRR)+1)  
    # Calculate the noisy Adjacency matrix --> ns_A
    ns_A = CalcNSA(A, p)  # # Upper triangular matrix
    # Calculate the noisy degree --> lap_deg
    lap_deg = AddLaplaceNS(deg, 0, 1/epsLap)
    
    hat_A = (ns_A-p*sp.triu(np.ones((NodeNum, NodeNum)), k=1, format='csr'))/(1-2*p)  # Upper triangular matrix

    # Calculate the first and the second item --> item1, item2
    item1 = hat_A.dot(lap_deg).dot(lap_deg)/EdgeNum
    item2 = ((0.5*sum(lap_deg*lap_deg)-(NodeNum+2)/epsLap**2)/EdgeNum)**2-(5*NodeNum+4)/epsLap**4/EdgeNum**2
    return ns_rn

# Calculate Assort_Coeff_Numerator in the interactive local model (Shuffle_{ru})
def CalcILocACN(A, deg, epsRR, epsLap):
    NodeNum = A.shape[0]
    EdgeNum = sum(deg)/2
    p = 1/(np.exp(epsRR)+1)
    ns_A = CalcNSA(A, p)          # 上三角矩阵
    lap_deg = AddLaplaceNS(deg, 0, 1/epsLap)
    hatA = (ns_A-p*sp.triu(np.ones((NodeNum, NodeNum)), k=1, format='csr'))/(1-2*p)  
    # Calculate the first and the second item --> item1, item2
    item1 = hatA.dot(lap_deg).dot(deg)/EdgeNum
    item2 = ((0.5*sum(lap_deg*lap_deg)-(NodeNum+2)/epsLap**2)/EdgeNum)**2-(5*NodeNum+4)/epsLap**4/EdgeNum**2
    ns_rn = item1-item2
    return ns_rn

# Calculate Assort_Coeff_Numerator in the decentralized model (Decentral_{re})
def CalcDecACN(A, deg, EpsDeg, EpsNBDegSum):
    delta_1 = delta/2
    NodeNum = A.shape[0]
    EdgeNum = sum(deg)/2
    # Calculate the noisy degree --> lap_deg
    lap_deg = AddLaplaceNS(deg, 0, 2/EpsDeg)
    # Calculate the upper bound of the local sensitivity of the sum of neighbor degrees --> sen_NBDegSum
    ProbUpperBound_deg = lap_deg + 2/EpsDeg*np.log(1.0/(2*delta_1))
    sorted_ProbUpperBound_deg = np.sort(ProbUpperBound_deg)
    first_max = sorted_ProbUpperBound_deg[-1]
    second_max = sorted_ProbUpperBound_deg[-2]
    sen_NBDegSum = 2*(first_max+second_max)
    # Calculate the noisy sum of degrees of all neighbors --> lap_nb_deg_sum
    nb_deg_sum = A.dot(deg)
    lap_nb_deg_sum = AddLaplaceNS(nb_deg_sum, 0, sen_NBDegSum/EpsNBDegSum)
    # Calculate Assort_Coeff_Numerator --> ns_rn (With empirical estimation)
    item1 = 0.5*np.dot(lap_deg, lap_nb_deg_sum)/EdgeNum
    item2 = (0.5*sum(lap_nb_deg_sum)/EdgeNum)**2-0.5*NodeNum*sen_NBDegSum**2/EpsNBDegSum**2/EdgeNum**2
    item3 = ((0.5*sum(lap_deg*lap_deg)-4*(NodeNum+2)/EpsDeg**2)/EdgeNum)**2-16*(5*NodeNum+4)/EpsDeg**4/EdgeNum**2
    ns_rn = item1-item2
    ns_rn1 = item1-item3
    return ns_rn, ns_rn1



if __name__ == "__main__":
    # Edge file (in)
    EdgeFile = input('[EdgeFile (in)]:')
    # Numericalbound File (in)
    NumerBoundFile = input('[NumerBoundFile (in)]:')
    # ResultDir
    ResultDir = input('[Output directory]:')
    
    # Total number of nodes --> all_node_num
    with open(EdgeFile, 'r') as f:
        for i in range(2):
            num = f.readline().rstrip(",\n")
    all_node_num = int(num)

    ItrNum = 100  # Number of iterations
    delta = 10**(-8)

    a_mat = ReadEdges(EdgeFile, all_node_num)  # csr_matrix
    NodeNum = all_node_num
    deg = a_mat.dot([1]*all_node_num)
    print('max_degree =', max(deg))
    print('avg_degree =', sum(deg)/NodeNum)
    r = CalcAC(a_mat, deg)
    ru = CalcACN(a_mat, deg)
    print('r =', r)
    print('ru =', ru)

    
    eps_lst = np.arange(0.1, 2.1, 0.1)
    
    avg_re_lc = 0
    avg_l2_lc = 0
    
    avg_re_sfn = 0
    avg_l2_sfn = 0
    avg_re_sfc = 0
    avg_l2_sfc = 0
    
    avg_re_dct = 0
    avg_l2_dct = 0
    avg_re_dct1 = 0
    avg_l2_dct1 = 0
    
    sign_lc = 0
    sign_sfn = 0
    sign_sfc = 0
    sign_dct = 0
    sign_dct1 = 0
    

    # with ThreadPoolExecutor(max_workers = 1) as pool:
    with ThreadPoolExecutor() as pool:
        epsL_nlst = list(pool.map(ReadNumericalBound, [NumerBoundFile]*20, [NodeNum]*20, eps_lst)) 
        epsL_clst = list(pool.map(CalcEpsL, [NodeNum-1]*20, eps_lst)) 
    print('Numerical bounds:', epsL_nlst)
    print('Closed form bounds:', epsL_clst)
    epsL_nlst = np.array(epsL_nlst)
    epsL_clst = np.array(epsL_clst)
    

    lc_partial = partial(CalcNLocACN, a_mat, deg)
    sf_partial = partial(CalcILocACN, a_mat, deg)
    dct_partial = partial(CalcDecACN, a_mat, deg)


    with ProcessPoolExecutor(max_workers = 10) as pool:
    # with ProcessPoolExecutor() as pool:
        for itr in range(ItrNum):
            print('-'*10, itr+1, '-'*10)
            # Local_{ru}
            _lc = list(pool.map(lc_partial, 0.6*eps_lst, 0.4*eps_lst))
            print('_lc: ', _lc)
            re_lc = np.fabs(np.array(_lc)-ru)/max(ru, 0.001*NodeNum)
            l2_lc = (np.array(_lc)-ru)**2
            avg_re_lc += re_lc
            avg_l2_lc += l2_lc
            sign_lc += (np.array(_lc)*ru > 0)

            # Shuffle_{ru}[N]
            _sfn = list(pool.map(sf_partial, 0.6*epsL_nlst, 0.4*epsL_nlst))
            print('_sfn: ', _sfn)
            re_sfn = np.fabs(np.array(_sfn)-ru)/max(ru, 0.001*NodeNum)
            l2_sfn = (np.array(_sfn)-ru)**2
            avg_re_sfn += re_sfn
            avg_l2_sfn += l2_sfn
            sign_sfn += (np.array(_sfn)*ru > 0)

            # Shuffle_{ru}[C]
            _sfc = list(pool.map(sf_partial, 0.6*epsL_clst, 0.4*epsL_clst))
            print('_sfc: ', _sfc)
            re_sfc = np.fabs(np.array(_sfc)-ru)/max(ru, 0.001*NodeNum)
            l2_sfc = (np.array(_sfc)-ru)**2
            avg_re_sfc += re_sfc
            avg_l2_sfc += l2_sfc
            sign_sfc += (np.array(_sfc)*ru > 0)

            # Decentral_{ru}
            _dct = list(pool.map(dct_partial, 0.4*2*eps_lst, 0.6*2*eps_lst))
            print('_dct: ', _dct)
            re_dct = np.fabs(np.array(_dct)[:,0]-ru)/max(ru, 0.001*NodeNum)
            l2_dct = (np.array(_dct)[:,0]-ru)**2
            re_dct1 = np.fabs(np.array(_dct)[:,1]-ru)/max(ru, 0.001*NodeNum)
            l2_dct1 = (np.array(_dct)[:,1]-ru)**2
            avg_re_dct += re_dct
            avg_l2_dct += l2_dct
            avg_re_dct1 += re_dct1
            avg_l2_dct1 += l2_dct1
            sign_dct += (np.array(_dct)[:,0]*ru > 0)
            sign_dct1 += (np.array(_dct)[:,1]*ru > 0)

    avg_re_lc /= ItrNum
    avg_l2_lc /= ItrNum
    
    avg_re_sfn /= ItrNum
    avg_l2_sfn /= ItrNum
    avg_re_sfc /= ItrNum
    avg_l2_sfc /= ItrNum
    
    avg_re_dct /= ItrNum
    avg_l2_dct /= ItrNum
    avg_re_dct1 /= ItrNum
    avg_l2_dct1 /= ItrNum
    
    
    # Output result
    ResultFile1 = ResultDir + '/eps(itr%d)_l2.csv'%(ItrNum)
    ResultFile2 = ResultDir + '/eps(itr%d)_re.csv'%(ItrNum)
    ResultFile3 = ResultDir + '/sign(itr%d).csv'%(ItrNum)
   
    print("Outputting result.")
    f = open(ResultFile1, "w")
    print("eps,Local,Shuffle[N],Shuffle[C],Decentral,1@Decentral@1", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for i in range(len(eps_lst)):
        lst = [eps_lst[i], avg_l2_lc[i], avg_l2_sfn[i], avg_l2_sfc[i], avg_l2_dct[i], avg_l2_dct1[i]]
        writer.writerow(lst)
    f.close()
    
    f = open(ResultFile2, "w")
    print("eps,Local,Shuffle[N],Shuffle[C],Decentral,1@Decentral@1", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for i in range(len(eps_lst)):
        lst = [eps_lst[i], avg_re_lc[i], avg_re_sfn[i], avg_re_sfc[i], avg_re_dct[i], avg_re_dct1[i]]
        writer.writerow(lst)
    f.close()

    f = open(ResultFile3, "w")
    print("eps,Local,Shuffle[N],Shuffle[C],Decentral,1@Decentral@1", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for i in range(len(eps_lst)):
        lst = [eps_lst[i], sign_lc[i], sign_sfn[i], sign_sfc[i], sign_dct[i], sign_dct1[i]]
        writer.writerow(lst)
    f.close()



