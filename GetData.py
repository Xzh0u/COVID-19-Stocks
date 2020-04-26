import baostock as bs
import pandas as pd
import re

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

f = open("stock_code_sh.txt")
sh = f.read()
f.close()

f = open("stock_code_sz.txt")
sz = f.read()
f.close()

sh_code = re.findall(r'[(](.*?)[)]', sh)
sz_code = re.findall(r'[(](.*?)[)]', sz)

print(sh_code)
print(sz_code)

for i in range(0, int(len(sz_code))):
    rs = bs.query_history_k_data_plus("sz." + sz_code[i],
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                      start_date='2019-12-01', end_date='2020-04-07',
                                      frequency="d", adjustflag="3")  # frequency="d"取日k线，adjustflag="3"默认不复权
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 结果集输出到csv文件

    result.to_csv("data/sh/" + sz_code[i] + "_data_sz_" + str(i) + ".csv",
                  encoding="gbk", index=False)
    print(result)

# 登出系统
bs.logout()
