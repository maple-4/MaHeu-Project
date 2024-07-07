
from solve import *
import time

if __name__ == "__main__":
    file_path = 'inst003.txt'
    dataframes = read_inst_file_to_dataframe(file_path)
    V, vehs, T, T_vps, trans, l, P, S, transportation_combination, plants, deals = read_data_from_dataframes(dataframes)
    t1 = time.process_time()
    sol1 = solve_normal(V, vehs, T, T_vps, trans, l, P, S)
    t2 = time.process_time()
    arrival_date, cap_trans, selected_route, sol, df, df_tr = greedy_algorithm(V, vehs, T, trans, transportation_combination, plants, deals)
    t3 = time.process_time()
    print(f"MIP model reach 1% gap by {t2-t1} seconds, final result is {sol1}")
    print(f"Greedy model finish by {t3-t2} seconds, final result is {sol}")