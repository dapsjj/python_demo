import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dateutil.relativedelta import relativedelta

# 读取用户数据
user_df = pd.read_csv("5用户33个月数据.csv", encoding='gbk', dtype={'用户编码': str})
# user_df = pd.read_csv("36个月不缺失.csv", dtype={'用户编码': str})
# 读取茂名天气数据
weather_df = pd.read_csv("202012-202312茂名天气.csv")

# 将电费年月转换为日期格式
user_df['电费年月'] = pd.to_datetime(user_df['电费年月'], format='%Y%m')
weather_df['年月'] = pd.to_datetime(weather_df['年月'], format='%Y%m')

# 合并用户数据和茂名天气数据
merged_df = pd.merge(user_df, weather_df, left_on='电费年月', right_on='年月', how='left')

# 提取年份和月份
merged_df['year'] = merged_df['电费年月'].dt.year
merged_df['month'] = merged_df['电费年月'].dt.month

# 扩展表示月份特征的列，将布尔值转换为整数
merged_df['is_year_start'] = (merged_df['month'] == 1).astype(int)  # 一年的开始
merged_df['is_year_end'] = (merged_df['month'] == 12).astype(int)  # 一年的结束
merged_df['is_quarter_start'] = (merged_df['month'] % 3 == 1).astype(int)  # 一个季度的开始
merged_df['is_quarter_end'] = (merged_df['month'] % 3 == 0).astype(int)  # 一个季度的结束

merged_df.drop(columns=['年月'], inplace=True)

# 创建 TimeSeriesDataFrame 对象
train_data = TimeSeriesDataFrame.from_data_frame(
    merged_df,
    id_column="用户编码",
    timestamp_column="电费年月"
)
prediction_length = 3
# 创建并训练模型
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,  # 预测未来3个月的数据
    path="ag",
    target="计费电量",
    known_covariates_names=['year','month','is_year_start','is_year_end','is_quarter_start','is_quarter_end','白天平均温度', '夜间平均温度'],  # 添加额外的特征
    eval_metric="MAE",
    freq='MS'
)

predictor.fit(
    train_data=train_data,
    presets="medium_quality",  # other options: ['best_quality', 'high_quality', 'medium_quality', 'fast_training']
    time_limit=3600,
)

# 获取每个用户的最后一个时间戳
last_dates = merged_df.groupby("用户编码")['电费年月'].max()

# 创建待预测的数据集
pred_df_list = []
pred_df = pd.DataFrame(columns=['用户编码', '电费年月', '白天平均温度', '夜间平均温度', 'year', 'month', 'is_year_start', 'is_year_end', 'is_quarter_start', 'is_quarter_end'])
# 循环遍历每个用户的最后一个时间戳
for user_code, last_date in last_dates.items():
    # 生成该用户未来3个月的时间戳
    future_dates = [last_date + relativedelta(months=i) for i in range(1, prediction_length+1)]
    # 将合并后的特征加入待预测数据集
    for future_date in future_dates:
        # 生成年月字符串，格式类似于 202309
        future_month_str = future_date.strftime('%Y%m')
        # 在 weather_df 中找到匹配的行
        matched_row = weather_df[weather_df['年月'].dt.strftime('%Y%m') == future_month_str]
        # 如果匹配到了，则提取对应的白天平均温度和夜间平均温度
        if not matched_row.empty:
            daytime_temp = matched_row['白天平均温度'].values[0]
            nighttime_temp = matched_row['夜间平均温度'].values[0]
        else:
            # 如果未匹配到，将温度设置为缺失值或者采取其他策略
            daytime_temp = None
            nighttime_temp = None
            # 添加待预测数据
        user_pred_df = pd.DataFrame({
            '用户编码': [user_code],
            '电费年月': [future_date],
            '白天平均温度': [daytime_temp],
            '夜间平均温度': [nighttime_temp],
            'year': [future_date.year],
            'month': [future_date.month],
            'is_year_start': [int(future_date.month == 1)],
            'is_year_end': [int(future_date.month == 12)],
            'is_quarter_start': [int(future_date.month % 3 == 1)],
            'is_quarter_end': [int(future_date.month % 3 == 0)]
        })
        # 将用户信息添加到 pred_df 中
        pred_df = pd.concat([pred_df, user_pred_df], ignore_index=True)


known_covariates = TimeSeriesDataFrame.from_data_frame(
    pred_df,
    id_column="用户编码",
    timestamp_column="电费年月"
)
predictor = TimeSeriesPredictor.load("ag")  # 指定保存的模型文件夹
# 进行预测
predictions = predictor.predict(data=train_data, known_covariates=known_covariates)
result_df = predictions.reset_index()[['item_id', 'timestamp', 'mean']]
result_df.columns = ['用户编码', '电费年月', '计费电量']  # 重命名列

# 保存预测结果到文件
result_df.to_csv('future_predictions.csv', index=False)
