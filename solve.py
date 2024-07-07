import pandas as pd
import numpy as np
import re
from datetime import datetime
from gurobipy import GRB, Model, quicksum
import time
import random

def read_inst_file_to_dataframe(file_path):
    data = {
        "LOC": [],
        "STE": [],
        "MOD": [],
        "SEG": [],
        "PTH": [],
        "PTHSG": [],
        "TSL": [],
        "PTR": [],
        "TRO": [],
        "TRONXT": [],
        "TRORTE": []
    }

    column_headers = {
        "LOC": ["ID", "Code", "Longitude", "Latitude"],
        "STE": ["ID", "Code", "Alias", "Type", "LocationCode", "VehicleCapacity", "Operator", "City", "Country"],
        "MOD": ["ID", "Code", "ModelAlias", "Make", "LengthMM", "WidthMM", "HeightMM", "WeightKG", "CapDemand:TRUCK",
                "CapDemand:TRAIN", "CapDemand:VESSEL", "CapDemand:SHUNTING", "CapDemand:SELF_PROPELLED",
                "CapDemand:UNKNOWN"],
        "SEG": ["ID", "Code", "OriginCode", "DestinationCode", "TransportMode", "DefaultLeadTimeHours"],
        "PTH": ["ID", "PathCode", "PathOriginCode", "PathDestinationCode"],
        "PTHSG": ["PathID", "PathCode", "SegmentSequenceNumber", "SegmentCode", "SegmentOriginCode",
                  "SegmentDestinationCode"],
        "TSL": ["Index", "Date"],
        "PTR": ["PathSegmentCode", "TimeSlotIndex", "TimeSlotDate", "LeadTimeHours", "Capacity", "MaxExtraCapacity"],
        "TRO": ["ID", "Code", "ModelCode", "OriginCode", "DestinationCode", "AvailableDateOrigin", "DesignatedPathCode",
                "DueDateDestinaton"],
        "TRONXT": ["TransportObjectCode", "NextDispatchSiteCode", "NextDispatchSiteArrivalDate"],
        "TRORTE": ["TransportObjectCode", "OptionIndex", "RouteCode"]
    }

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")[:-1]
            keyword = parts[0]
            if keyword in data:
                data[keyword].append(parts[1:])

    # transfer to DataFrame
    dataframes = {}
    for keyword, records in data.items():
        if records:
            num_columns = len(records[0])
            df = pd.DataFrame(records, columns=column_headers[keyword][:num_columns])
            dataframes[keyword] = df

    return dataframes
def split_route(route):
    s_t, by,_=route.split("-")
    # countrycode+num+name
    pattern = re.compile(r'([A-Z]{3}\d{2}(PLANT|TERM|DEAL|PORT))([A-Z]{3}\d{2}(PLANT|TERM|DEAL|PORT))')
    match = pattern.match(s_t)
    return match.group(1), match.group(3),by
def read_data_from_dataframes(dataframes):
    def segmentList(path, dataframes):
        return dataframes["PTHSG"][dataframes["PTHSG"]["PathCode"] == path]["SegmentCode"].to_list()

    def date_to_int(date):
        date0 = dataframes["PTR"][dataframes["PTR"]["TimeSlotIndex"] == "0"]["TimeSlotDate"].iloc[0]
        date_format = "%d/%m/%Y-%H:%M:%S"
        datetime0 = datetime.strptime(date0, date_format)
        datetime1 = datetime.strptime(date, date_format)
        int_day = datetime1 - datetime0
        return int_day.days

    V = [int(v[3:]) - 1 for v in dataframes["TRONXT"]["TransportObjectCode"]]
    Path = [dataframes["TRO"][dataframes["TRONXT"]["TransportObjectCode"] == v]["DesignatedPathCode"].iloc[0] for v in
            dataframes["TRONXT"]["TransportObjectCode"]]
    vehs = pd.DataFrame({"path": np.array(Path)}, index=pd.Series(V))
    vehs["type"] = dataframes["TRO"]["ModelCode"].to_list()
    vehs["size"] = [
        float(dataframes["MOD"][dataframes["MOD"]["Code"] == vehs.loc[v, "type"]]["CapDemand:TRUCK"].iloc[0]) for v in
        V]
    vehs["has_due"] = [0 if dataframes["TRO"].loc[v, "DueDateDestinaton"] == "-" else 1 for v in V]
    vehs["due_time"] = [date_to_int(dataframes["TRO"].loc[v, "DueDateDestinaton"]) if vehs.loc[v, "has_due"] else "-"
                        for v in V]
    vehs["produce_time"] = [date_to_int(dataframes["TRO"].loc[v, "AvailableDateOrigin"]) for v in V]
    vehs["product_place"] = dataframes["TRO"]["OriginCode"]
    vehs["origin"] = dataframes["TRO"]["OriginCode"]
    vehs["destination"] = dataframes["TRO"]["DestinationCode"]
    vehs["pathList"] = [dataframes["PTH"][(dataframes["PTH"]["PathOriginCode"]==vehs.loc[v, 'origin']) &
                                   (dataframes["PTH"]["PathDestinationCode"]==vehs.loc[v, 'destination'])]["PathCode"].to_list() for v in V]
    vehs["segmentLists"] = [[segmentList(p, dataframes) for p in vehs.loc[v, "pathList"]] for v in V]
    vehs["netTransTimeList"] =[
            [sum(float(dataframes["SEG"][dataframes["SEG"]["Code"] == seg]["DefaultLeadTimeHours"].iloc[0]) / 24 for seg in seglist)
            for seglist in  vehs.loc[v, "segmentLists"]]for v in V]
    vehs["netTransTime"] = [min(vehs.loc[v, "netTransTimeList"]) for v in V]
    vehs["netTransTimeIndex"] = [vehs.loc[v, "netTransTimeList"].index(vehs.loc[v, "netTransTime"]) for v in V]

    P = [[idx for idx, _ in enumerate(vehs.loc[v, "pathList"])] for v in V]
    #S = [[[seg for seg in segmentList(vehs.loc[v, "pathList"][p], dataframes)] for p in P[v]] for v in V]
    S = [[[idx for idx, _ in enumerate(segmentList(vehs.loc[v, "pathList"][p], dataframes))] for p in P[v]] for v in V]
    T = [i for i in range(len(dataframes["PTR"]))]


    trans = pd.DataFrame(index=pd.Series(T), columns=["code", "from", "to", "by", "start_date", "duration_days", "capacity"])
    for t in range(len(trans)):
        trans.loc[t, "code"] = dataframes["PTR"].loc[t, "PathSegmentCode"]
        trans.loc[t, "from"], trans.loc[t, "to"], trans.loc[t, "by"] \
            = split_route(dataframes["PTR"].loc[t, "PathSegmentCode"])
        trans.loc[t, "start_date"] = int(dataframes["PTR"].loc[t, "TimeSlotIndex"])
        trans.loc[t, "duration_days"] = float(dataframes["PTR"].loc[t, "LeadTimeHours"]) / 24
        trans.loc[t, "capacity"] = float(dataframes["PTR"].loc[t, "Capacity"])

    l = [
            [
                [int(float(dataframes["SEG"][dataframes["SEG"]["Code"] == seg]["DefaultLeadTimeHours"].iloc[0])/24) if "DEAL" not in seg
           else int(float(dataframes["SEG"][dataframes["SEG"]["Code"] == seg]["DefaultLeadTimeHours"].iloc[0])/24-0.1)+1
                 for seg in segmentList(p, dataframes)]
                for p in vehs.loc[v, "pathList"]]
        for v in V]

    T_vps = [
        [[trans['from'][trans['from'] == split_route(vehs.loc[v, "segmentLists"][p][s])[0]].index.to_list() for s in
          S[v][p]]
         for p in P[v]] for v in V]

    def find_routes(segment_index, current_route, date):
        if segment_index == len(seglist):
            all_route.append(current_route)
            dates.append(date)
            return
        current_segment = seglist[segment_index]
        segment_data = dataframes["PTR"][dataframes["PTR"]['PathSegmentCode'] == current_segment]
        for idx, row in segment_data.iterrows():
            if float(dataframes["PTR"].loc[idx,"TimeSlotIndex"]) + 0.05 >= date:
                new_route = current_route + [idx]
                date_NXT = float(dataframes["PTR"].loc[idx, "TimeSlotIndex"]) + float(dataframes["PTR"].loc[idx, "LeadTimeHours"])/24
                date_NXT = round(date_NXT + 0.1)
                find_routes(segment_index + 1, new_route, date_NXT+1)

    plants = []
    deals = []
    for place in dataframes["LOC"]["Code"]:
        if "PLANT" in place:
            plants.append(place)
        elif "DEAL" in place:
            deals.append(place)
    transportation_combination = pd.DataFrame(columns=["route", "origin", "destination", "start_date", "end_date"])
    for origin in plants:
        for destination in deals:
            paths = dataframes["PTH"][
                (dataframes["PTH"]["PathOriginCode"] == origin) & (dataframes["PTH"]["PathDestinationCode"] == destination)][
                "PathCode"].to_list()
            all_route = []
            dates = []
            for p in paths:
                seglist = segmentList(p, dataframes)
                find_routes(0, [], 0)
            new_data = pd.DataFrame({
                'route': all_route,
                'origin': [origin] * len(all_route),
                'destination': [destination] * len(all_route),
                'start_date' : 0,
                'end_date': dates
            })
            transportation_combination = pd.concat([transportation_combination, new_data], ignore_index=True)
    transportation_combination["start_date"] = [trans.loc[rt[0], "start_date"] for rt in transportation_combination["route"]]


    return V, vehs, T, T_vps, trans, l, P, S, transportation_combination, plants, deals


def solve_normal(V, vehs, T, T_vps, trans, l, P, S):
    VD = []
    VN = []
    for v in V:
        if vehs.loc[v, "has_due"]:
            VD.append(v)
        else:
            VN.append(v)
    a = vehs["produce_time"].to_list()

    ST = trans["start_date"].to_list()
    CD = vehs["size"].to_list()
    C = trans["capacity"].to_list()
    DT = vehs["due_time"].to_list()
    N = vehs["netTransTime"]


    model = Model("IP_Model")
    model.setParam('MIPGap', 0.01)
    # Variables
    x = {}
    for v in V:
        for p in P[v]:
            for s in S[v][p]:
                x[v, p, s] = model.addVar(vtype='i',lb=0)
    w = {}
    for v in V:
        for p in P[v]:
            w[v, p] = model.addVar(vtype='b')
    y = {}
    for v in V:
        for p in P[v]:
            for s in S[v][p]:
                for t in T:
                    y[v, p, s, t] = model.addVar(vtype='b')
    d = [model.addVar(vtype='b') for v in V]
    R = [model.addVar(vtype='i', lb=0) for v in V]
    B = [model.addVar(vtype='i', lb=0) for v in V]

    # Constrains
    model.addConstrs(quicksum(x[v,p,0] for p in P[v]) >= a[v] for v in V)  # 第一次离开时间>=出厂时间
    model.addConstrs(
        x[v, p, s-1] + w[v, p]*(l[v][p][s-1] + 1/2) <= x[v,p,s] for v in V for p in P[v] for s in S[v][p] if s>=1)# 后一次出发时间>=前一次出发+dura+1
    model.addConstrs(quicksum(w[v,p] for p in P[v]) <= 1 for v in V)
    model.addConstrs(quicksum(w[v, p] for p in P[v]) >= 1 for v in V)
    model.addConstrs(quicksum(y[v,p,s,t] for t in T_vps[v][p][s]) == w[v,p] for v in V for p in P[v] for s in S[v][p])
    model.addConstrs(quicksum(y[v,p,s,t] * ST[t] for t in T_vps[v][p][s]) == x[v,p,s] for v in V for p in P[v] for s in S[v][p])
    model.addConstrs(y[v,p,s,t] <= w[v,p] for v in V for p in P[v] for s in S[v][p] for t in T_vps[v][p][s])
    model.addConstrs(quicksum(quicksum(quicksum(y[v,p,s,t] * CD[v] for s in S[v][p])for p in P[v])for v in V) <= C[t] for t in T)
    M = max(trans["start_date"] + trans["duration_days"])
    model.addConstrs(quicksum(x[v,p, S[v][p][-1]] + w[v,p]*l[v][p][S[v][p][-1]] for p in P[v]) - DT[v] <= M*d[v] for v in VD)
    model.addConstrs(quicksum(x[v,p, S[v][p][-1]] + w[v,p]*l[v][p][S[v][p][-1]] for p in P[v]) - DT[v] <= R[v] for v in VD)
    model.addConstrs(quicksum(x[v,p, S[v][p][-1]] + w[v,p]*l[v][p][S[v][p][-1]] for p in P[v]) - a[v]-2*N[v] <=B[v] for v in VD)
    model.addConstrs(x[v,p, S[v][p][-1]] + l[v][p][S[v][p][-1]] <= M for v in V for p in P[v])

    model.setObjective(quicksum(quicksum(x[v,p, S[v][p][-1]] + w[v,p]*l[v][p][S[v][p][-1]]for p in P[v]) - a[v] for v in VN)
                       +quicksum(100*d[v]+25*R[v]+5*B[v] for v in VD), GRB.MINIMIZE)


    model.optimize()

    # due rate
    print(sum([1 if d[v].X else 0 for v in VD])/len(VD))
    # utilisation
    utilisation = pd.DataFrame(index=T, columns=["used", "capacity", "utilisation"],data=0.)
    for v in V:
        for p in P[v]:
            for s in S[v][p]:
                for t in T_vps[v][p][s]:
                    if y[v, p, s,t].X == 1:
                        utilisation.loc[t,"used"] += vehs.loc[v, "size"]
    utilisation["capacity"] = trans["capacity"]
    utilisation["utilisation"] = utilisation["used"]/utilisation["capacity"]
    print(utilisation["utilisation"])
    total_utilisation = sum(utilisation["used"]) / sum(utilisation["capacity"])
    print(f"total utilisation = {total_utilisation}")
    #utilisation.to_excel("utilisation_normal.xlsx")
    return model.ObjVal



def greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals):
    end_date = max(trans["start_date"] + trans["duration_days"])
    # sort veh has due
    vehs_due = vehs[vehs["has_due"] == True].sort_values(by="due_time").index.to_list()
    # sort veh has no due
    vehs_ndue = vehs[vehs["has_due"] == False].sort_values(by="produce_time").index.to_list()
    vehs_or = vehs_due + vehs_ndue
    cap_trans = trans["capacity"].to_list()
    arrival_date = [end_date+1 for v in V]
    df = pd.DataFrame(index=plants,columns=deals)
    df_tr = []
    for origin in plants:
        for destination in deals:
            df.loc[origin, destination] = len(df_tr)
            df_tr.append(transportation_combination[(transportation_combination["origin"]==origin) &
                                        (transportation_combination["destination"]==destination)].sort_values(by="end_date"))
    selected_route = [[]for v in V]
    for v in vehs_or:
        st = 0
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            if st == 1: break
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and (not sum([cap_trans[r] <= size for r in rlist])):
                for r in rlist:
                    cap_trans[r] -= size
                arrival_date[v] = tc["end_date"]
                selected_route[v] = tc["route"]
                st = 1
                # print(f"choose {v} to {rlist},pt={vehs.loc[v, 'produce_time']} has_due={vehs.loc[v, 'has_due']}, due_time={vehs.loc[v, 'due_time']}, at={arrival_date[v]}")

    sol = 0
    for v in vehs_ndue:
        sol += arrival_date[v]-vehs.loc[v,"produce_time"]
    for v in vehs_due:
        delay = max(0, arrival_date[v]-vehs.loc[v, "due_time"])
        if delay >0.05: sol += 100 + 25*delay
        exceedDoubleNetworkTime = max(0, arrival_date[v] - vehs.loc[v,"produce_time"] - 2 * vehs.loc[v, "netTransTime"])
        sol += 5 * exceedDoubleNetworkTime
    print(sol)
    plan = pd.DataFrame(index=V, columns=T,data=0)
    for v in V:
        for rt in selected_route[v]:
            plan.loc[v, rt] = 1
    return arrival_date, cap_trans, selected_route, sol, df, df_tr
def CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals):
    end_date = max(trans["start_date"] + trans["duration_days"])
    vehs_due = vehs[vehs["has_due"] == True].sort_values(by="due_time").index.to_list()
    vehs_ndue = vehs[vehs["has_due"] == False].sort_values(by="produce_time").index.to_list()
    vehs_or = vehs_due + vehs_ndue

    cap_trans = trans["capacity"].to_list()
    arrival_date = [end_date + 1 for v in V]

    df = pd.DataFrame(index=plants, columns=deals)
    df_tr = []
    for origin in plants:
        for destination in deals:
            df.loc[origin, destination] = len(df_tr)
            df_tr.append(transportation_combination[(transportation_combination["origin"] == origin) &
                                                    (transportation_combination[
                                                         "destination"] == destination)].sort_values(by="end_date"))
    selected_route = [[] for v in V]

    unassigned_count = 0
    for v in vehs_or:
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        RCL = []  # Restricted candidate list
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist):  # (not sum([cap_trans[r] <= size for r in rlist])):
                RCL.append((tc, rlist))
                if len(RCL) > 3:
                    break
        if RCL:
            chosen_route, chosen_rlist = random.choice(RCL)
            for r in chosen_rlist:
                cap_trans[r] -= size
            arrival_date[v] = chosen_route["end_date"]
            selected_route[v] = chosen_route["route"]
        else:
            print(f"Warning: Vehicle {v} could not be assigned within constraints.")
            unassigned_count += 1

    sol = 0
    for v in vehs_ndue:
        sol += arrival_date[v] - vehs.loc[v, "produce_time"]
    for v in vehs_due:
        delay = max(0, arrival_date[v] - vehs.loc[v, "due_time"])
        if delay > 0.05: sol += 100 + 25 * delay
        exceedDoubleNetworkTime = max(0,
                                      arrival_date[v] - vehs.loc[v, "produce_time"] - 2 * vehs.loc[v, "netTransTime"])
        sol += 5 * exceedDoubleNetworkTime
    print(f"greedy_sol : {sol}")
    plan = pd.DataFrame(index=V, columns=T, data=0)
    for v in V:
        for rt in selected_route[v]:
            plan.loc[v, rt] = 1
    return arrival_date, cap_trans, selected_route, sol, df, df_tr
def VL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals, alpha):
    end_date = max(trans["start_date"] + trans["duration_days"])
    vehs_due = vehs[vehs["has_due"] == True].sort_values(by="due_time").index.to_list()
    vehs_ndue = vehs[vehs["has_due"] == False].sort_values(by="produce_time").index.to_list()
    vehs_or = vehs_due + vehs_ndue

    cap_trans = trans["capacity"].to_list()
    arrival_date = [end_date + 1 for v in V]

    df = pd.DataFrame(index=plants, columns=deals)
    df_tr = []
    for origin in plants:
        for destination in deals:
            df.loc[origin, destination] = len(df_tr)
            df_tr.append(transportation_combination[(transportation_combination["origin"] == origin) &
                                                    (transportation_combination[
                                                         "destination"] == destination)].sort_values(by="end_date"))
    selected_route = [[] for v in V]
    unassigned_count = 0
    for v in vehs_or:
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        RCL = []  # Restricted candidate list
        min_end_date = float('inf')
        max_end_date = float('-inf')

        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist):  # (not sum([cap_trans[r] <= size for r in rlist])):
                RCL.append((tc, rlist))
                min_end_date = min(min_end_date, tc["end_date"])
                max_end_date = max(max_end_date, tc["end_date"])

        end_date_threshold = min_end_date + alpha * (max_end_date - min_end_date)
        RCL = [(tc, rlist) for (tc, rlist) in RCL if tc["end_date"] <= end_date_threshold]

        if RCL:
            chosen_route, chosen_rlist = random.choice(RCL)
            for r in chosen_rlist:
                cap_trans[r] -= size
            arrival_date[v] = chosen_route["end_date"]
            selected_route[v] = chosen_route["route"]
        else:
            print(f"Warning: Vehicle {v} could not be assigned within constraints.")
            unassigned_count += 1

    sol = 0
    for v in vehs_ndue:
        sol += arrival_date[v] - vehs.loc[v, "produce_time"]
    for v in vehs_due:
        delay = max(0, arrival_date[v] - vehs.loc[v, "due_time"])
        if delay > 0.05: sol += 100 + 25 * delay
        exceedDoubleNetworkTime = max(0,
                                      arrival_date[v] - vehs.loc[v, "produce_time"] - 2 * vehs.loc[v, "netTransTime"])
        sol += 5 * exceedDoubleNetworkTime
    print(f"greedy_sol: {sol}")
    plan = pd.DataFrame(index=V, columns=T, data=0)
    for v in V:
        for rt in selected_route[v]:
            plan.loc[v, rt] = 1
    return arrival_date, cap_trans, selected_route, sol, df, df_tr


def local_search_1best(arrival_date, cap_trans, selected_route, sol, df, df_tr, V, vehs):
    change_list = [[] for v in V]
    sol_list = [sol for v in V]
    new_date = [0 for v in V]
    for v in V:
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        dt = arrival_date[v]
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist) and rlist != selected_route[v] and dt > tc["end_date"]:
                dt = min(dt, tc["end_date"])
                change_list[v] = rlist
        if change_list[v]:
            s_origin = cost_v(v, arrival_date[v],vehs)
            s_now = cost_v(v, dt,vehs)
            sol_list[v] += s_now - s_origin
            new_date[v] = dt
    new_sol = min(sol_list)
    print(f"new_val: {new_sol}")
    if sol - new_sol >= 0.5:
        min_v = sol_list.index(new_sol)
        sz = vehs.loc[min_v, "size"]
        for rt in selected_route[min_v]:
            cap_trans[rt] += sz
        for rt in change_list[min_v]:
            cap_trans[rt] -= sz
        selected_route[min_v] = change_list[min_v]
        arrival_date[min_v] = new_date[min_v]
        arrival_date, cap_trans, selected_route, sol = local_search_1best(arrival_date, cap_trans, selected_route, new_sol,df, df_tr, V, vehs)
        return arrival_date, cap_trans, selected_route, sol
    else:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
def local_search_1half(arrival_date, cap_trans, selected_route, sol, df, df_tr, V, vehs, trans, T):
    # consider neighborhood of veh-air
    change_list = []
    ipst = 0
    ipv = -1
    ipdate = max(trans["start_date"] + trans["duration_days"]) + 1
    for v in V:
        if ipst: break
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        dt = arrival_date[v]
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist) and rlist != selected_route[v] and dt > tc["end_date"]:
                dt = tc["end_date"]
                change_list = rlist
                ipst = 1
                ipv = v
                ipdate = dt
    new_sol = sol
    if change_list and ipv != -1:
        s_origin = cost_v(ipv, arrival_date[ipv],vehs)
        s_now = cost_v(ipv, ipdate,vehs)
        new_sol += s_now - s_origin

    print(f"new_val: {new_sol}")
    if sol - new_sol >= 0.5:
        print(f"new_val(veh-air): {new_sol}")
        # accept veh-air
        sz = vehs.loc[ipv, "size"]
        print(f"arrange {ipv} from {selected_route[ipv]} to {change_list}")
        for rt in selected_route[ipv]:
            cap_trans[rt] += sz
        for rt in change_list:
            cap_trans[rt] -= sz
        selected_route[ipv] = change_list
        arrival_date[ipv] = ipdate
        if not all([cap_trans[t] >= 0 for t in T]):
            val = min(cap_trans)
            print(f"idx = {cap_trans.index(val)}, val = {val}")
        arrival_date, cap_trans, selected_route, sol = local_search_1half(arrival_date, cap_trans,
                                                                               selected_route, new_sol, df, df_tr, V, vehs, trans, T)
        return arrival_date, cap_trans, selected_route, sol
    else:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
def local_search_op2(arrival_date, cap_trans, selected_route, sol, df, df_tr, V, vehs, trans, T):
    # consider neighborhood of veh-veh
    exchange_combi = []
    res = 0
    for v0 in V:
        origin0 = vehs.loc[v0, "origin"]
        destination0 = vehs.loc[v0, "destination"]
        size0 = vehs.loc[v0, "size"]
        for v1 in V[v0+1:]:
            origin1 = vehs.loc[v1, "origin"]
            destination1 = vehs.loc[v1, "destination"]
            size1 = vehs.loc[v1, "size"]
            st = 0
            if size0 > size1:
                if all([cap_trans[rt]+size1>=size0 for rt in selected_route[v1]]):
                    st = 1
            else:
                if all([cap_trans[rt]+size0>=size1 for rt in selected_route[v0]]):
                    st = 1
            if st and origin0 == origin1 and destination0 == destination1:
                if vehs.loc[v0, "produce_time"]<=trans.loc[selected_route[v1][0], "start_date"] and vehs.loc[v1, "produce_time"]<=trans.loc[selected_route[v0][0], "start_date"]:
                    new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                    if new_res < res:# cost would be less
                        res = new_res
                        exchange_combi = [v0, v1]
    if res >= -0.5:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
    else:
        print(f"new_val(veh-veh): {sol + res}")
        # accept veh-veh
        v0 = exchange_combi[0]
        v1 = exchange_combi[1]
        size0 = vehs.loc[v0, "size"]
        size1 = vehs.loc[v1, "size"]
        print(f"switch {v0} and {v1}, by {selected_route[v0]}<----->{selected_route[v1]}")
        for rt in selected_route[v0]:
            cap_trans[rt] += size0 - size1
        for rt in selected_route[v1]:
            cap_trans[rt] += size1 - size0
        term_r = selected_route[v0]
        selected_route[v0] = selected_route[v1]
        selected_route[v1] = term_r
        term_t = arrival_date[v0]
        arrival_date[v0] = arrival_date[v1]
        arrival_date[v1] = term_t
        if not all([cap_trans[t] >= 0 for t in T]):
            val = min(cap_trans)
            print(f"idx = {cap_trans.index(val)}, val = {val}")
        arrival_date, cap_trans, selected_route, sol = local_search_op2(arrival_date, cap_trans, selected_route,
                                                                    sol+res, df, df_tr, V, vehs, trans, T)
        return arrival_date, cap_trans, selected_route, sol
def local_search_op12(arrival_date, cap_trans, selected_route, sol, df, df_tr, V, vehs, trans, T):
    # consider neighborhood of veh-air
    change_list = [[] for v in V]
    sol_list = [sol for v in V]
    new_date = [0 for v in V]
    for v in V:
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        dt = arrival_date[v]
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist) and rlist != selected_route[v] and dt > tc["end_date"]:
                dt = tc["end_date"]
                change_list[v] = rlist
        if change_list[v]:
            s_origin = cost_v(v, arrival_date[v],vehs)
            s_now = cost_v(v, dt,vehs)
            sol_list[v] += s_now - s_origin
            new_date[v] = dt
    new_sol = min(sol_list)

    # consider neighborhood of veh-veh
    exchange_combi = []
    res = 0
    for v0 in V:
        origin0 = vehs.loc[v0, "origin"]
        destination0 = vehs.loc[v0, "destination"]
        size0 = vehs.loc[v0, "size"]
        for v1 in V[v0+1:]:
            origin1 = vehs.loc[v1, "origin"]
            destination1 = vehs.loc[v1, "destination"]
            size1 = vehs.loc[v1, "size"]
            st = 0
            if size0 > size1:
                if all([cap_trans[rt]+size1>=size0 for rt in selected_route[v1]]):
                    st = 1
            else:
                if all([cap_trans[rt]+size0>=size1 for rt in selected_route[v0]]):
                    st = 1
            if st and origin0 == origin1 and destination0 == destination1:
                if vehs.loc[v0, "produce_time"]<=trans.loc[selected_route[v1][0], "start_date"] and vehs.loc[v1, "produce_time"]<=trans.loc[selected_route[v0][0], "start_date"]:
                    new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                    if new_res < res:# cost would be less
                        res = new_res
                        exchange_combi = [v0, v1]
    if new_sol - sol >= -0.5 and res >= -0.5:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
    else:
        if new_sol - sol < res:
            print(f"new_val(veh-air): {new_sol}")
            # accept veh-air
            min_v = sol_list.index(new_sol)
            sz = vehs.loc[min_v, "size"]
            print(f"arrange {min_v} from {selected_route[min_v]} to {change_list[min_v]}")
            for rt in selected_route[min_v]:
                cap_trans[rt] += sz
            for rt in change_list[min_v]:
                cap_trans[rt] -= sz
            selected_route[min_v] = change_list[min_v]
            arrival_date[min_v] = new_date[min_v]
            if not all([cap_trans[t]>=0 for t in T]):
                val = min(cap_trans)
                print(f"idx = {cap_trans.index(val)}, val = {val}")

            arrival_date, cap_trans, selected_route, sol = local_search_op12(arrival_date, cap_trans, selected_route,
                                                                        new_sol, df, df_tr, V, vehs, trans, T)
            return arrival_date, cap_trans, selected_route, sol
        else:
            print(f"new_val(veh-veh): {sol + res}")
            # accept veh-veh
            v0 = exchange_combi[0]
            v1 = exchange_combi[1]
            size0 = vehs.loc[v0, "size"]
            size1 = vehs.loc[v1, "size"]
            print(f"switch {v0} and {v1}, by {selected_route[v0]}<----->{selected_route[v1]}")
            for rt in selected_route[v0]:
                cap_trans[rt] += size0 - size1
            for rt in selected_route[v1]:
                cap_trans[rt] += size1 - size0
            term_r = selected_route[v0]
            selected_route[v0] = selected_route[v1]
            selected_route[v1] = term_r
            term_t = arrival_date[v0]
            arrival_date[v0] = arrival_date[v1]
            arrival_date[v1] = term_t

            if not all([cap_trans[t] >= 0 for t in T]):
                val = min(cap_trans)
                print(f"idx = {cap_trans.index(val)}, val = {val}")
            arrival_date, cap_trans, selected_route, sol = local_search_op12(arrival_date, cap_trans, selected_route,
                                                                        sol+res, df, df_tr, V, vehs, trans, T)
            return arrival_date, cap_trans, selected_route, sol
def local_search_1half2best(arrival_date, cap_trans,selected_route,sol, df, df_tr, V, vehs, trans, T):
    # consider neighborhood of veh-air
    change_list = []
    ipst = 0
    ipv = -1
    ipdate = max(trans["start_date"] + trans["duration_days"]) + 1
    for v in V:
        if ipst: break
        origin = vehs.loc[v, "origin"]
        destination = vehs.loc[v, "destination"]
        size = vehs.loc[v, "size"]
        dt = arrival_date[v]
        for _, tc in df_tr[df.loc[origin, destination]].iterrows():
            rlist = tc["route"]
            if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                    cap_trans[r] >= size for r in rlist) and rlist != selected_route[v] and dt > tc["end_date"]:
                dt = tc["end_date"]
                change_list = rlist
                ipst = 1
                ipv = v
                ipdate = dt
    new_sol = sol
    if change_list and ipv != -1:
        s_origin = cost_v(ipv, arrival_date[ipv],vehs)
        s_now = cost_v(ipv, ipdate,vehs)
        new_sol += s_now - s_origin

    # consider neighborhood of veh-veh
    exchange_combi = []
    res = 0
    for v0 in V:
        origin0 = vehs.loc[v0, "origin"]
        destination0 = vehs.loc[v0, "destination"]
        size0 = vehs.loc[v0, "size"]
        for v1 in V[v0 + 1:]:
            origin1 = vehs.loc[v1, "origin"]
            destination1 = vehs.loc[v1, "destination"]
            size1 = vehs.loc[v1, "size"]
            st = 0
            if size0 > size1:
                if all([cap_trans[rt] + size1 >= size0 for rt in selected_route[v1]]):
                    st = 1
            else:
                if all([cap_trans[rt] + size0 >= size1 for rt in selected_route[v0]]):
                    st = 1
            if st and origin0 == origin1 and destination0 == destination1:
                if vehs.loc[v0, "produce_time"] <= trans.loc[selected_route[v1][0], "start_date"] and vehs.loc[
                    v1, "produce_time"] <= trans.loc[selected_route[v0][0], "start_date"]:
                    new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[
                        v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                    if new_res < res:  # cost would be less
                        res = new_res
                        exchange_combi = [v0, v1]

    if new_sol - sol >= -0.5 and res >= -0.5:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
    else:
        if new_sol - sol < res:
            print(f"new_val(veh-air): {new_sol}")
            # accept veh-air
            sz = vehs.loc[ipv, "size"]
            print(f"arrange {ipv} from {selected_route[ipv]} to {change_list}")
            for rt in selected_route[ipv]:
                cap_trans[rt] += sz
            for rt in change_list:
                cap_trans[rt] -= sz
            selected_route[ipv] = change_list
            arrival_date[ipv] = ipdate
            if not all([cap_trans[t] >= 0 for t in T]):
                val = min(cap_trans)
                print(f"idx = {cap_trans.index(val)}, val = {val}")

            arrival_date, cap_trans, selected_route, sol = local_search_1half2best(arrival_date, cap_trans,
                                                                                          selected_route,
                                                                                          new_sol, df, df_tr, V, vehs, trans, T)
            return arrival_date, cap_trans, selected_route, sol
        else:
            print(f"new_val(veh-veh): {sol + res}")
            # accept veh-veh
            v0 = exchange_combi[0]
            v1 = exchange_combi[1]
            size0 = vehs.loc[v0, "size"]
            size1 = vehs.loc[v1, "size"]
            print(f"switch {v0} and {v1}, by {selected_route[v0]}<----->{selected_route[v1]}")
            for rt in selected_route[v0]:
                cap_trans[rt] += size0 - size1
            for rt in selected_route[v1]:
                cap_trans[rt] += size1 - size0
            term_r = selected_route[v0]
            selected_route[v0] = selected_route[v1]
            selected_route[v1] = term_r
            term_t = arrival_date[v0]
            arrival_date[v0] = arrival_date[v1]
            arrival_date[v1] = term_t
            arrival_date, cap_trans, selected_route, sol = local_search_1half2best(arrival_date, cap_trans,
                                                                                          selected_route,
                                                                                          sol + res, df, df_tr, V, vehs, trans, T)
            return arrival_date, cap_trans, selected_route, sol
def local_search_1half2best_remain(arrival_date, cap_trans, selected_route, sol, df, df_tr, optu1 ,optu2, his1, his2, V, vehs, trans, T):
    u1st = 1
    if optu1:
        if optu1[0] not in his2:
            ipv = optu1[0]
            new_sol = optu1[1]
            change_list = optu1[2]
            ipdate = optu1[3]
            u1st=0
    if u1st:
        # consider neighborhood of veh-air
        change_list = []
        ipst = 0
        ipv = -1
        ipdate = max(trans["start_date"] + trans["duration_days"]) + 1
        for v in V:
            if ipst: break
            origin = vehs.loc[v, "origin"]
            destination = vehs.loc[v, "destination"]
            size = vehs.loc[v, "size"]
            dt = arrival_date[v]
            for _, tc in df_tr[df.loc[origin, destination]].iterrows():
                rlist = tc["route"]
                if vehs.loc[v, "produce_time"] <= tc["start_date"] and all(
                        cap_trans[r] >= size for r in rlist) and rlist != selected_route[v] and dt > tc["end_date"]:
                    dt = tc["end_date"]
                    change_list = rlist
                    ipst = 1
                    ipv = v
                    ipdate = dt
        new_sol = sol
        if change_list and ipv != -1:
            s_origin = cost_v(ipv, arrival_date[ipv],vehs)
            s_now = cost_v(ipv, ipdate,vehs)
            new_sol += s_now - s_origin
            # update optu1 and his1
            print("compute optu1")
            optu1 = [ipv, new_sol, change_list, ipdate]
            his1.append(ipv)

    # consider neighborhood of veh-veh
    u2st = 1
    if optu2:
        if (optu2[0]not in his1) and (optu2[1]not in his1):
            exchange_combi = [optu2[0], optu2[1]]
            res = optu2[2]
            u2st = 0
    if u2st:
        exchange_combi = []
        res = 0
        for v0 in V:
            origin0 = vehs.loc[v0, "origin"]
            destination0 = vehs.loc[v0, "destination"]
            size0 = vehs.loc[v0, "size"]
            for v1 in V[v0 + 1:]:
                origin1 = vehs.loc[v1, "origin"]
                destination1 = vehs.loc[v1, "destination"]
                size1 = vehs.loc[v1, "size"]
                st = 0
                if size0 > size1:
                    if all([cap_trans[rt] + size1 >= size0 for rt in selected_route[v1]]):
                        st = 1
                else:
                    if all([cap_trans[rt] + size0 >= size1 for rt in selected_route[v0]]):
                        st = 1
                if st and origin0 == origin1 and destination0 == destination1:
                    if selected_route[v1] and selected_route[v0]:
                        if vehs.loc[v0, "produce_time"] <= trans.loc[selected_route[v1][0], "start_date"] and vehs.loc[
                                v1, "produce_time"] <= trans.loc[selected_route[v0][0], "start_date"]:
                            new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[
                                v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                            if new_res < res:  # cost would be less
                                res = new_res
                                exchange_combi = [v0, v1]
                    elif selected_route[v1]:
                        if vehs.loc[v0, "produce_time"] <= trans.loc[selected_route[v1][0], "start_date"]:
                            new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[
                                v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                            if new_res < res:  # cost would be less
                                res = new_res
                                exchange_combi = [v0, v1]
                    elif  selected_route[v0]:
                        if vehs.loc[v1, "produce_time"] <= trans.loc[selected_route[v0][0], "start_date"]:
                            new_res = cost_v(v0, arrival_date[v1],vehs) + cost_v(v1, arrival_date[v0],vehs) - cost_v(v0, arrival_date[
                                v0],vehs) - cost_v(v1, arrival_date[v1],vehs)
                            if new_res < res:  # cost would be less
                                res = new_res
                                exchange_combi = [v0, v1]

        # update optu2 and his2
        if exchange_combi:
            print("compute optu2")
            optu2 = [exchange_combi[0], exchange_combi[1], res]
            his2.append(exchange_combi[0])
            his2.append(exchange_combi[1])

    if new_sol - sol >= -0.5 and res >= -0.5:
        print(f"this is the best solution, with sol = {sol}")
        return arrival_date, cap_trans, selected_route, sol
    else:
        if new_sol - sol < res:
            print(f"new_val(veh-air): {new_sol}")
            # accept veh-air
            sz = vehs.loc[ipv, "size"]
            print(f"arrange {ipv} from {selected_route[ipv]} to {change_list}")
            for rt in selected_route[ipv]:
                cap_trans[rt] += sz
            for rt in change_list:
                cap_trans[rt] -= sz
            selected_route[ipv] = change_list
            arrival_date[ipv] = ipdate
            if not all([cap_trans[t] >= 0 for t in T]):
                val = min(cap_trans)
                print(f"idx = {cap_trans.index(val)}, val = {val}")

            arrival_date, cap_trans, selected_route, sol = local_search_1half2best_remain(arrival_date, cap_trans, selected_route,
                                                                             new_sol, df, df_tr,[],optu2,his1,[])
            return arrival_date, cap_trans, selected_route, sol
        else:
            print(f"new_val(veh-veh): {sol + res}")
            # accept veh-veh
            v0 = exchange_combi[0]
            v1 = exchange_combi[1]
            size0 = vehs.loc[v0, "size"]
            size1 = vehs.loc[v1, "size"]
            for rt in selected_route[v0]:
                cap_trans[rt] += size0 - size1
            for rt in selected_route[v1]:
                cap_trans[rt] += size1 - size0
            term_r = selected_route[v0]
            selected_route[v0] = selected_route[v1]
            selected_route[v1] = term_r
            term_t = arrival_date[v0]
            arrival_date[v0] = arrival_date[v1]
            arrival_date[v1] = term_t
            arrival_date, cap_trans, selected_route, sol = local_search_1half2best_remain(arrival_date, cap_trans,selected_route,
                                                                                sol + res, df, df_tr,optu1,[],[],his2)
            return arrival_date, cap_trans, selected_route, sol


def greedy_local_op2(V, vehs, T, trans, transportation_combination, plants, deals):
    arrival_date, cap_trans, selected_route, sol, df, df_tr \
        = greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
    arrival_date, cap_trans, selected_route, sol = local_search_op2(arrival_date, cap_trans, selected_route, sol, df, df_tr,V, vehs, trans, T)

    return arrival_date, cap_trans, selected_route, sol
def greedy_local_op12(V, vehs, T, trans, transportation_combination, plants, deals):
    arrival_date, cap_trans, selected_route, sol, df, df_tr \
        = greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
    arrival_date, cap_trans, selected_route, sol = local_search_op12(arrival_date, cap_trans, selected_route, sol, df, df_tr, V, vehs, trans, T)

    return arrival_date, cap_trans, selected_route, sol
def greedy_local_1half2best_remain(V, vehs, T, trans, transportation_combination, plants, deals):
    arrival_date, cap_trans, selected_route, sol, df, df_tr \
        = greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
    arrival_date, cap_trans, selected_route, sol = local_search_1half2best_remain(arrival_date, cap_trans, selected_route, sol, df, df_tr
                                                                                  ,[],[],[],[], V, vehs, trans, T)

    return arrival_date, cap_trans, selected_route, sol
def CL_GRASP(V, vehs, T, trans, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None
    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
        arrival_date, cap_trans, selected_route, sol = local_search_op12(arrival_date, cap_trans, selected_route,
                                                                         sol, df, df_tr, V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def CL_GRASP_1half(V, vehs, T, trans, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None

    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
        arrival_date, cap_trans, selected_route, sol = local_search_1half(arrival_date, cap_trans, selected_route,
                                                                         sol, df, df_tr, V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def CL_GRASP_1best(V, vehs, T, trans, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None

    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
        arrival_date, cap_trans, selected_route, sol = local_search_1best(arrival_date, cap_trans, selected_route,sol, df, df_tr, V, vehs)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def CL_GRASP_1half2best(V, vehs, T, trans, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None

    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
        arrival_date, cap_trans, selected_route, sol = local_search_1half2best(arrival_date, cap_trans,
                                                                                      selected_route,
                                                                                      sol, df, df_tr, V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def CL_GRASP_1half2best_remain(V, vehs, T, trans, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None

    for i in range(MaxIteration):
        t1 = time.process_time()
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = CL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
        arrival_date, cap_trans, selected_route, sol = local_search_1half2best_remain(arrival_date, cap_trans, selected_route,
                                                                         sol, df, df_tr,[],[],[],[], V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def VL_GRASP(V, vehs, T, trans,alpha, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None

    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = VL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals, alpha)
        arrival_date, cap_trans, selected_route, sol = local_search_op12(arrival_date, cap_trans, selected_route,
                                                                         sol, df, df_tr, V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)

    return best_arrival_date, best_cap_trans, best_route, best_sol
def VL_GRASP_1half2best_remain(V, vehs, T, trans,alpha, MaxIteration, transportation_combination, plants, deals):
    best_sol = float('inf')
    best_route = None
    best_arrival_date = None
    best_cap_trans = None
    for i in range(MaxIteration):
        arrival_date, cap_trans, selected_route, sol, df, df_tr \
            = VL_random_greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals, alpha)
        arrival_date, cap_trans, selected_route, sol = local_search_1half2best_remain(arrival_date, cap_trans, selected_route,
                                                                         sol, df, df_tr,[],[],[],[], V, vehs, trans, T)
        if sol < best_sol:
            best_sol = sol
            best_plan = pd.DataFrame(index=V, columns=T, data=0)
            for v in V:
                for rt in selected_route[v]:
                    best_plan.loc[v, rt] = 1
            best_route = selected_route
            best_arrival_date = arrival_date
            best_cap_trans = cap_trans
    print(best_sol)
    return best_arrival_date, best_cap_trans, best_route, best_sol

def cost_v(v, date, vehs):
    if vehs.loc[v,"has_due"]:
        s = 0
        delay = max(0, date - vehs.loc[v, "due_time"])
        if delay > 0.05: s += 100 + 25 * delay
        exceedDoubleNetworkTime = max(0,date - vehs.loc[v, "produce_time"] - 2 * vehs.loc[v, "netTransTime"])
        s += 5 * exceedDoubleNetworkTime
    else:
        s = date - vehs.loc[v, "produce_time"]
    return s
def heu_sol_print(arrival_date, cap_trans, V, T, vehs, trans):
    # count delay
    dl = 0
    for v in V:
        if  vehs.loc[v, "has_due"] and arrival_date[v]>vehs.loc[v, "due_time"]:
            dl += 1
    delay_rate = dl / len(vehs[vehs['has_due'] == 1])
    # print(f"delay rate = {delay_rate}")
    # count utilisation of capacities
    utilisation = pd.DataFrame(index=T, columns=["remain", "capacity", "utilisation"], data=0.)
    utilisation["capacity"] = trans["capacity"]
    utilisation["remain"] = cap_trans
    for t in T:
        utilisation.loc[t, "utilisation"] = 1 - utilisation.loc[t,"remain"]/utilisation.loc[t,"capacity"]
    # print(utilisation)
    # total_utilisation = 1- sum(utilisation["remain"])/sum(utilisation["capacity"])
    # print(f"total utilisation : {total_utilisation}")





