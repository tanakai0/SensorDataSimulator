import numpy as np

Norm_EXP = 10.
Norm_IntVal = 10000.
N_Times = Norm_IntVal / Norm_EXP
D_H, H_M = 24, 60
D_M = D_H * H_M

Bre_Lun_bound = 9.5
balance_min = ((D_H/2)-Bre_Lun_bound)*H_M
eat_toil_lag = 2.5 * 60
bath_wash_lag = 1 * 60

def actseq_gen(Savedlist, speedpara, sleepdur, is_sleep_noon, GRWdurs, fres, order):
    """
    This generates activity sequence by the way of [1, 2] using probability motivation based plannning.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021).
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
    
    Parameters
    ----------
    Savedlist : list of str
        For example, Savedlist[i] = '0/WAR0000000KSCB0000000/1104,680,131,91KB'.
    sppedpara : 
    sleepdur : 
    is_sleep_noon : 
    GRWdurs : 
    fres : 
    order : 
    
    Returns
    -------
    Actseqs : 
    """
    raise ValueError('Now At Work.')
    Actseqs = []
    for path in Savedlist:
        [stj, fpc, houspara] = path.split('/')
        ActSeq = main(fpc, speedpara, sleepdur, is_sleep_noon, GRWdurs, fres, order)
        Actseqs.append(ActSeq)
    return Actseqs

def main(floorplancode, speedpara, sleepdur, is_sleep_noon, GRWdurs, fres, order):
    """
    This generates activity sequence by the way of [1, 2] using probability motivation based plannning.
    [1] C. Jiang and A. Mita, "SISG4HEI_Alpha: Alpha version of simulated indoor scenario generator for houses with elderly individuals." Journal of Building Engineering, 35-101963(2021).
    [2] https://github.com/Idontwan/SISG4HEI_Alpha.
    
    Parameters
    ----------
    floorplancode : str
        For example, 'WAR0000000KSCB0000000'.
    sppedpara : float
        Resident's activtiy speed.
    sleepdur : float
        Average duration time of sleep.
    is_sleep_noon : boolean
        Whether the resident sleeps at noon then.
    GRWdurs : list of float
        [Average duration of going out,
         Average duration of reading,
         Aaverage duration of watching TV]
    fres : lsit of int
        [Average frequency of goint to toilet, 
         Average frequency of wandering]
    order : list of int
        For example, [4, 0, 3, 1, 2], 0:'Bath', 1:'Go out', 2:'Clean', 3:'Read', 4:'Watch TV'
    
    Returns
    -------
    Actseq : list of tuple
        Activity sequnce. Actseq[i] = [i-th activity code number, duration time of i-th activity]
        activitie codes = [0 Sleep_Noon, 1 Sleep_Evening, 2 Wash_Self(in Bathroom), 3 Cooking0(in KS), 4 Cooking1(in CB), 5 Cooking2(in RFA), 6 Eat, 7 Bath, 8 Dress_up, 9 Go_out, 10 Toilet_Short, 11Toilet_Long, 12 Clean, 13 Work, 14 Watch_TV, 15 Wash_Clothing(in WM), 16 Wandering, 17 Resting]
    """
    Durs = RP.sam_bas_dura(speedpara, sleepdur, GRWdurs)
    Oft_Days, Toi_shor_Times, eatT4TL, Twan = RP.sam_fres(fres, order)
    MDurs, Tot_Sle_Dur, MD2D = RA.Mid_dur(Durs, floorplancode, Oft_Days, is_sleep_noon)
    weights = RA.Mid_weight(MDurs, Tot_Sle_Dur, Oft_Days, Toi_shor_Times, eatT4TL, Twan)
    MVs, start_minu = Or.origin_MVs_minu(weights, MDurs, Oft_Days, Tot_Sle_Dur, eatT4TL)
    ReMD2D, Redurs, Reweights = RA.real_act_dur(MDurs, weights, MD2D)
    start_node = 1 if ReMD2D[0] == [0] else 0
    Markchains = MP.Markchain(start_minu, start_node, MVs, ReMD2D, Reweights, Redurs)
    Actseq = post_process(Durs, Markchains, ReMD2D)
    return Actseq

def post_process(Durs, Markchains, ReMD2D):
    ACTSEQ = []
    N = len(Markchains)
    for i in range(N-1):
        code = Markchains[i][0]
        if code == len(ReMD2D)-1:
            num, smin = 17, Markchains[i][1]
            ACTSEQ.append([num, smin])
        else:
            smin, emin = Markchains[i][1], Markchains[i+1][1]
            actdur = emin - smin
            nums = ReMD2D[code]
            ThisDurs = [Durs[j] for j in nums]
            sumTD = sum(ThisDurs)
            ws = [TD/sumTD for TD in ThisDurs]
            for j in range(len(nums)):
                ACTSEQ.append([nums[j], smin])
                smin += ws[j]*actdur
    return ACTSEQ




# ------------Repo--------------------------------------------------------------
# Resident Profile

def sam_bas_dura(paraDoActFast, paraSleepDur, parasGRW):
    # unit minutes
    #[ 0 Sleep_Noon, Sleep_Evening, Wash_Self(in Bathroom), Cooking0(in KS), Cooking1(in CB),
    #  5 Cooking2(in RFA), Eat, Bath, Dress_up, Go_out,
    #  10 Toilet_Short, Toilet_Long, Clean, Work, Watch_TV,
    #  15 Wash_Clothing(in WM), Wandering]
    paras = [[15., 60.], [360., 180.], [3., 4.], [5., 10.], [0.5, 1.],
             [0.5, 0.5], [10., 20.], [5., 20.], [0.5, 3.5], [210., 180.],
             [0.6, 1.], [4., 16.], [10., 20.], [180., 150.], [120., 120.],
             [0.2, 0.8], [0.3, 0.4]]
    N = len(paras)
    rts = np.random.rand(N)
    t = 0.3 - 0.3*paraDoActFast
    for i in range(2, N-3):
        rts[i] = t + 0.7*rts[i]
    rts[1] = paraSleepDur
    rts[9], rts[13], rts[14] = parasGRW[0], parasGRW[1], parasGRW[2]
    Durs = np.zeros(N)
    for i in range(N):
        Durs[i] = 0.1 + paras[i][0] + paras[i][1]*rts[i] #0.1min is passing time
    return Durs


def sam_fres(fresTandW, orders):
    # Determining how often the resident bath/ go out/ clean/ work/ watch_TV
    Oft_Days = [1, 1, 1, 1, 1]
    AddDay_num1 = Tl.sample([0.2, 0.8, 1]) + 2
    for i in range(AddDay_num1): Oft_Days[i] += 1
    AddDay_num2 = Tl.sample([0.2, 0.8, 1])
    for i in range(AddDay_num2): Oft_Days[i] += 1
    if np.random.rand() < 0.5: Oft_Days[0] += 1 + Tl.sample([0.1, 0.3, 0.7, 0.9, 1])
    n_Oft_Days = [Oft_Days[4-orders.index(i)] for i in range(5)]
    # Determining how many times the resident go toilet(short) every day
    Toi_shor_Times = Tl.sample([0.15, 0.45, 0.75, 0.95, 1], fresTandW[0]) + 3
    # Determining how many times the resident eat between go toilet(long)
    eatT4TL = 1/(0.07 + 0.28*fresTandW[0])
    # Determing how often the resident wander
    exp_Twan = 4.5 * (1-fresTandW[1]) - 1.1
    Twan = np.exp(exp_Twan)
    return n_Oft_Days, Toi_shor_Times, eatT4TL, Twan




# -----------Real_Act------------------------------


def Mid_dur(durs, floorplancode, oft_Days, sleep_noon):
    MD2D = [[0], [2, 1, 2], [5, 3, 4, 6], [7, 8, 15],
            [8, 9, 8], [10], [11], [12], [13], [14], [15], [16]] #map Mid_dur to dur
    MDurs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 0 sleep_noon, sleep, eat, bath, go_out, toilet(short), toilet(long), clean, work,
    # 9 watch_TV, Washing, Wandering
    MDurs[1] = durs[1] + 2*durs[2] #Go_to_Bathroom->Sleep->Go_to_Bathroom
    if sleep_noon:
        MDurs[0] = durs[0]
        MDurs[1] -= durs[0]/2
        durs[1] -= durs[0]/2

    MDurs[2] = durs[6] #Cooking2->Cooking0->Cooking1->Dining
    if floorplancode[10] != '0':
        MDurs[2] += durs[3] + durs[4] + durs[5]
    else: MD2D[2] = [6]
    if floorplancode[12] == '0': MDurs[2] -= 0.1
    if floorplancode[14] == '0': MDurs[2] -= 0.1

    MDurs[3] = durs[7] + durs[8] #Go_to_Bathroom->Dress_up->Washing,
    if floorplancode[0] == '0': MDurs[3] -= 0.1 # once pass to WAR
    if floorplancode[17] != '0':
        MDurs[3] += durs[15] # time to wash cloth
        MDurs[10] = durs[15] # time to wash cloth
    else: MD2D[3] = [7, 8]

    MDurs[4] = durs[9] + 2*durs[8] #Dress_up->Go_out->Dress_up
    if floorplancode[0] == '0': MDurs[4] -= 0.2
    if floorplancode[3] == '0': MDurs[4] += oft_Days[1]*durs[13]/oft_Days[3]
    else: MDurs[8] += durs[13] # Work, Desk
    if floorplancode[5] == '0': MDurs[4] += oft_Days[1]*durs[14]/oft_Days[4]
    else: MDurs[9] += durs[14] # Watch_TV, Sofa

    for i in [5, 6, 7, 11]:
        MDurs[i] = durs[i+5]

    Tot_Sle_Dur = MDurs[0] + MDurs[1]
    return MDurs, Tot_Sle_Dur, MD2D


def Mid_weight(Mdurs, Tot_Sle_Dur, oftDays, Toi_shor_Times, eatT4TL, Twan):
    weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 0 sleep_noon, sleep, eat, bath, go_out, toilet(short), toilet(long), clean, work,
    # 9 watch_TV, Washing, Wandering
    for i in range(2):
        if Mdurs[i] != 0: weights[i] = Norm_IntVal/(D_M-Mdurs[i])
    weights[2] = Norm_IntVal/(3*(D_M-0.9*Tot_Sle_Dur-3*Mdurs[2])/8)
    js = [3, 4, 7, 8, 9]
    for i in range(5):
        if Mdurs[js[i]] != 0:
            weights[js[i]] = Norm_IntVal/(oftDays[i]*(D_M-Tot_Sle_Dur) - Mdurs[js[i]])
    weights[5] = Toi_shor_Times*Norm_IntVal / (D_M - 0.9*Tot_Sle_Dur)
    weights[6] = Norm_IntVal / eatT4TL # each meal, the Motivation value increase weights[6]
    if Mdurs[10] != 0: weights[10] = Norm_IntVal # each bath, the Motivation value increase
    weights[11] = Norm_IntVal / (Twan*(D_M-Tot_Sle_Dur))
    return weights


def real_act_dur(Mdurs, weights, MD2D):
    ReMD2D, Redurs, Reweights = [], [], []
    N = len(Mdurs)
    for i in range(N):
        if Mdurs[i] != 0:
            Redurs.append(Mdurs[i])
            Reweights.append(weights[i])
            ReMD2D.append(MD2D[i])
    ReMD2D.append([17])
    return ReMD2D, Redurs, Reweights



# ----------------Origin---------------------------------


def origin_MVs_minu(weights, MDurs, Oft_days, Tot_sle_dur, eatT4TL):
    # determining origin_MVs by determining when will them first time be happened
    O_Mvs = [0, Norm_IntVal, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 0 sleep_noon, sleep, eat, bath, go_out, toilet(short), toilet(long), clean, work,
    # 9 watch_TV, Washing, Wandering
    if weights[0] != 0: # sleep_noon after 11 to 15 hours after sleep_evening
        O_Mvs[0] = Norm_IntVal - (11+4*np.random.rand())*60*weights[0]
    O_Mvs[2] = Norm_IntVal - (0.1*MDurs[1]+15+30*np.random.rand())*weights[2]

    ord_nodes, n2oft = Tl.oft_order(Oft_days)
    minus_0 = MDurs[2] + 15 + 60*np.random.rand()
    O_Mvs[5] = Norm_IntVal - (0.1*MDurs[1]+minus_0)*weights[5]
    for node in ord_nodes:
        if weights[node] != 0:
            O_Mvs[node] = Norm_IntVal - minus_0*weights[node]
            minus_0 = minus_0 + (int(n2oft[str(node)])*(D_M-Tot_sle_dur)-minus_0)*np.random.rand()

    N = int(eatT4TL)
    O_Mvs[6] = weights[6]*np.random.randint(N)
    O_Mvs[11] = Norm_IntVal/3*np.random.rand()
    if weights[10] != 0: O_Mvs[10] = 0.02*Norm_IntVal

    start_minus = H_M - 3.5*H_M*np.random.rand()
    Mvs, NN = [], len(weights)
    for i in range(NN):
        if weights[i] != 0: Mvs.append(O_Mvs[i])
    Mvs.append(1.02*Norm_IntVal)
    return Mvs, start_minus


# -------- MarkovProcess ------------------------------------------------------------


'''
MD2D = [[0], [2, 1, 2], [3, 4, 5, 6], [7, 8, 15],
            [8, 9, 8], [10], [11], [12], [13], [14], [15], [16]]
    #[ 0 Sleep_Noon, Sleep_Evening, Wash_Self(in Bathroom), Cooking0(in KS), Cooking1(in CB),
    #  5 Cooking2(in RFA), Eat, Bath, Dress_up, Go_out,
    #  10 Toilet_Short, Toilet_Long, Clean, Work, Watch_TV,
    #  15 Wash_Clothing(in WM), Wandering]
'''



def Markchain(start_absminu, start_node, MVs, Node2Act, weights, ReDurs,
              lags=[[False, -1],[False, -1]], MVlog=False, update_Days=30):
    # lags for toilet_L and Washing

    if MVlog:
        MVlog = []
        for i in range(len(MVs)):
            MVlog.append([MVs[i]])

    Markchains = [[start_node, start_absminu]]
    absminu = Markchains[-1][-1]

    while absminu < update_Days*D_M:
        act_code = Markchains[-1][0]
        act_dur = 5+15*np.random.rand() if act_code==len(MVs)-1 else ReDurs[act_code]
        duration = Sample_duration(act_code, Node2Act, act_dur)
        updateMV(MVs, lags, absminu, act_code, weights, Node2Act, ReDurs, act_dur, duration)

        if lags[0][1] <= 0: lags[0][0] = False
        if lags[1][1] <= 0: lags[1][0] = False

        act_code = sample_next_actcode(MVs)
        absminu += duration
        if act_code!= Markchains[-1][0]:
            Markchains.append([act_code, absminu])

            if MVlog:
                for i in range(len(MVs)):
                    MVlog[i].append(MVs[i])
    if MVlog: return MVlog, Markchains

    return Markchains


def Sample_duration(act_code, ReMD2D, act_dur):
    act = ReMD2D[act_code]
    rt = np.random.rand()
    if act == [2, 1, 2]: w = 0.97+0.06*rt
    elif act == [8, 9, 8]: w = 0.4 + 0.5*rt
    elif act == [13] or act == [14]: w = 0.3 + 0.4*rt
    else: w = 0.95+0.1*rt
    return w*act_dur


def updateMV(MVs, lags, absminu, actcode, weights, ReMD2D, ReDurs, act_dur, redur):
    if ReMD2D[0] == [0]: Tol_l_code = 6
    else: Tol_l_code = 5
    N = len(MVs) - 1
    act = ReMD2D[actcode]
    if act == [0] or act == [2, 1, 2]: # sleep
        for i in range(N):
            if i == actcode: MVs[i] = MvU.norm_decrese(MVs[i], act_dur, redur)
            elif ReMD2D[i] == [0] or ReMD2D[i] == [2, 1, 2]:
                MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
            elif i== Tol_l_code-1 or 6 in ReMD2D[i]:
                MVs[i] = MvU.sleeptimeinc(MVs[i], redur, weights[i])
            elif i == Tol_l_code:
                if lags[0][0]: MVs[i], lags[0][1] = MvU.lag(MVs[i], weights[i], redur, lags[0][1])
            elif ReMD2D[i] == [15]:
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
    elif 6 in act: # eat
        for i in range(N):
            if i == actcode: MVs[i] = MvU.eat_hungry(absminu, MVs[i])
            elif i == Tol_l_code:
                lags[0][0], lags[0][1] = True, eat_toil_lag
            elif ReMD2D[i] == [15]:
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
    elif actcode == Tol_l_code - 3: # bath
        for i in range(N):
            if i == actcode: MVs[i] = MvU.norm_decrese(MVs[i], act_dur, redur)
            elif ReMD2D[i] == [15]:
                lags[1][0], lags[1][1] = True, bath_wash_lag
            elif i == Tol_l_code:
                if lags[0][0]: MVs[i], lags[0][1] = MvU.lag(MVs[i], weights[i], redur, lags[0][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
    elif act == [8, 9, 8]: # go_out
        eat_code = actcode - 2
        hun_v, toiL_v, toL_lag, remins = MvU.goout_eat_toilL_value(absminu, redur, MVs[eat_code], ReDurs[eat_code],
                                                                   weights[eat_code], MVs[Tol_l_code], weights[Tol_l_code],
                                                                   ReDurs[Tol_l_code], toL_lag=lags[0][0],
                                                                   remain_minu=lags[0][1])
        for i in range(N):
            if i == actcode: MVs[i] = MvU.norm_decrese(MVs[i], act_dur, redur)
            elif i == eat_code: MVs[i] = hun_v
            elif i == Tol_l_code: MVs[i], lags[0][0], lags[0][1] = toiL_v, toL_lag, remins
            elif i == Tol_l_code-1: MVs[i] = MvU.goout_toilS_val(MVs[i], redur, ReDurs[i], weights[i])
            elif ReMD2D[i] == [15]:
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
    elif actcode == Tol_l_code: # Toilet(long)
        for i in range(N):
            if i == actcode:
                MVs[i] -= Norm_IntVal
                if lags[0][0]: MVs[i], lags[0][1] = MvU.lag(MVs[i], weights[i], redur, lags[0][1])
            elif ReMD2D[i] == [15]:
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
    elif act == [15]: # washing
        for i in range(N):
            if i == actcode:
                MVs[i] -= Norm_IntVal
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
            elif i == Tol_l_code:
                if lags[0][0]: MVs[i], lags[0][1] = MvU.lag(MVs[i], weights[i], redur, lags[0][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])
    else:
        for i in range(N):
            if i == actcode: MVs[i] = MvU.norm_decrese(MVs[i], act_dur, redur)
            elif i == Tol_l_code:
                if lags[0][0]: MVs[i], lags[0][1] = MvU.lag(MVs[i], weights[i], redur, lags[0][1])
            elif ReMD2D[i] == [15]:
                if lags[1][0]: MVs[i], lags[1][1] = MvU.lag(MVs[i], weights[i], redur, lags[1][1])
            else: MVs[i] = MvU.daytimeinc(MVs[i], redur, weights[i])


def sample_next_actcode(MVs):
    N = len(MVs)
    expweights = []
    for i in range(N):
        expwei = np.exp(MVs[i]/N_Times-9.8) if MVs[i]/N_Times > 9.8 else 0
        expweights.append(expwei)
    sum_expwei = sum(expweights)
    possIs = []
    for i in range(N):
        possI = sum(expweights[0:i+1])/sum_expwei
        possIs.append(possI)
    actcode = Tl.sample(possIs)
    return actcode


# ------------------Tools----------------------------------------


def sample(I_Weights, t=0):
    N = len(I_Weights)
    if not t: t = np.random.rand()
    for i in range(N):
        if t < I_Weights[i]: return i


def Mval2TranP(Moti_value, Act2Moti_Map):
    N = len(Act2Moti_Map)
    TranPossI = []
    for i in range(N):
        num_l = Act2Moti_Map[i]
        P = 0
        for v in num_l:
            P += np.exp(Moti_value[v])
        TranPossI.append(P)
    for i in range(N-1):
        TranPossI[i+1] += TranPossI[i]
    TranPossI = [TranPossI[i]/TranPossI[-1] for i in range(N)]
    return TranPossI


def is_lundin(abs_mins):
    # determine whether this meal is lunch or dinner (not breakfast)
    return (abs_mins+balance_min)//(D_M//2)%2


def oft_order(oft_days):
    rs = np.random.rand(5)
    node2oft = {}
    nodes = ['3', '4', '7', '8', '9']
    for i in range(5):
        node2oft[nodes[i]] = oft_days[i] + rs[i]/5
    sort_nodes = sorted(nodes, key=lambda node:node2oft[node])
    orderednodes = [int(n) for n in sort_nodes]
    return orderednodes, node2oft


def gen_subtxt(absminu, act, dur):
    day, left_minu = int(absminu//D_M), absminu%D_M
    PorA = 'AM' if left_minu<D_M//2 else 'PM'
    hour, minu = int(left_minu//H_M), int(left_minu%H_M)
    if PorA == 'PM': hour -= int(D_H//2)
    second = int(60*((left_minu%H_M)-minu))
    d_hour, d_minu = int(dur//H_M), int(dur%H_M)
    d_second = int(60*((dur%H_M)-d_minu))
    txt1 = str(day)+'d '+PorA+' '+str(hour)+'h '+str(minu)+'m '+str(second)+'s'
    txt2 = act
    txt3 = str(d_hour)+'h '+str(d_minu)+'m '+str(d_second)+'s'
    txt_l = [txt1, txt2, txt3, '\n']
    return '   '.join(txt_l)

