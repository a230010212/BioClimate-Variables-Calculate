# bio_climate.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import csv

class DataPreprocessor:
    def __init__(self, filepath):
        """
        初始化資料預處理器
        :param filepath: 原始數據文件路徑
        """
        self.filepath = filepath
        self.output_dir = 'processed_ids'
        self.date_formats = [  # 添加日期格式列表
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%m-%d-%Y', '%m/%d/%Y', '%Y%m%d', '%d.%m.%Y'
        ]
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_batch_data(self, batch_ids, start_date, end_date):
        """高效讀取批次數據並根據日期範圍過濾"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)

            fid_index = headers.index('TARGET_FID')
            date_index = headers.index('Date')
        
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 預轉換為集合提升查詢速度
            target_ids = set(map(str, batch_ids))
            
            # 緩衝區收集數據行
            rows = [headers]  # 保留標題列
            rows.extend(row for row in reader if row[fid_index] in target_ids and start_dt <= pd.to_datetime(row[date_index]) <= end_dt)
#             for row in reader:
#                 if row[fid_index] in target_ids:
#                     dt = pd.to_datetime(row[date_index])
#                     if (dt >= start_dt) & (dt <= end_dt):
#                         rows.append(row)
        
        # 轉換為DataFrame並處理數據類型
        if len(rows) > 1:
            df = pd.DataFrame(rows[1:], columns=headers)
            # 將日期欄轉換為 datetime 型別
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
#             # 篩選日期範圍
#             start_dt = pd.to_datetime(start_date)
#             end_dt = pd.to_datetime(end_date)
#             df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)] 
                     
            numeric_cols = ['temperature_2m_MEAN', 'temperature_2m_max_MEAN',
                           'temperature_2m_min_MEAN', 'total_precipitation_sum_MEAN']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        return pd.DataFrame()

    def process_data(self, total_ids, batch_size=5, start_date="1000-01-01" , end_date = "9999-01-01"):
        """批次處理主程式"""
        
        all_ids = []
        for f in os.listdir(self.output_dir):
            if f:
                all_ids.append(int(f.split('_')[2].split('.')[0]))
            else:
                print("processed_ids中不存在文件")
                
        # 自動檢測ID範圍
        if len(all_ids) > 0: 
            start_ids = max(all_ids)+1
            unique_ids = range(start_ids,total_ids)
            
        else:
            unique_ids = range(total_ids)
            start_ids = 0
            
        for batch_start in tqdm(range(start_ids, total_ids, batch_size),
                               desc="Processing IDs", unit="batch"):
            batch_ids = range(batch_start, min(batch_start+batch_size, total_ids))
            batch_df = self._read_batch_data(batch_ids, start_date, end_date)
            
            if not batch_df.empty:
                try:
                    processed = self._full_preprocess(batch_df)
                    self._save_chunk(processed, f"batch_{batch_start}_{batch_ids[-1]}.csv")
                except Exception as e:
                    print(f"處理批次 {batch_ids} 失敗: {str(e)}")
                finally:
                    del batch_df  # 主動釋放內存

    def _full_preprocess(self, df):
        """完整預處理流程"""
        
        self.df = df.copy()
        self._convert_date()
        self._validate_dates()
        self._add_year_month()
        self._rename_columns()
        self._convert_temp_KtoC()
        self._add_tra_column()
        self._group_monthly()
        return self.df

    def _save_chunk(self, chunk, filename):
        """保存批次處理結果"""
        chunk.to_csv(os.path.join(self.output_dir, filename), 
                    index=False, encoding='utf-8')


    def merge_and_export(self, output_filename):
        """合併並導出最終結果"""
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        if not all_files:
            raise FileNotFoundError("未找到任何批次處理結果文件")
               
        # 逐步合併確保內存效率
        merged = pd.DataFrame()
        for f in tqdm(all_files, desc="Merging files"):
            file_path = os.path.join(self.output_dir, f)
            chunk = pd.read_csv(file_path)

            merged = pd.concat([merged, chunk], ignore_index=True)
        
        # 整合最終結果
        
        calculator = BioClimateCalculator(merged)
        result = calculator.calculate_all()
        final_result = result.groupby(['ID','Year'], as_index=False).mean()
        final_result.to_csv(output_filename, index=False)
        
        # 清理臨時文件
        for f in all_files:
            os.remove(os.path.join(self.output_dir, f))
        os.rmdir(self.output_dir)
        print(f"最終結果已保存至 {output_filename}")

    # 以下預處理方法保持不變但調整為實例方法
    def _convert_date(self):
        """日期格式轉換（優化錯誤處理）"""
        try:
            self.df['Date'] = pd.to_datetime(
                self.df['Date'],
                format='mixed',
                dayfirst=False,
                errors='coerce'
            )
        except Exception as e:
            print(f"日期轉換異常: {str(e)}")
            self.df['Date'] = pd.to_datetime(
                self.df['Date'],
                errors='coerce'
            )

    def _validate_dates(self):
        """驗證日期（增加百分比統計）"""
        na_count = self.df['Date'].isna().sum()
        if na_count > 0:
            total = len(self.df)
            print(f"警告: 發現 {na_count} 個無效日期 ({na_count/total:.2%})")

    def _add_year_month(self):
        """新增年、月欄位到指定位置"""
        self.df.insert(1, 'Year', '')
        self.df.insert(2, 'Month', '')
        self.df['Month'] = self.df['Date'].map(lambda x: x.month)
        self.df['Year'] = self.df['Date'].map(lambda x: x.year)
        
    def _rename_columns(self):
        """標準化欄位名稱"""
        column_mapping = {
            'temperature_2m_MEAN': 'Temp',
            'temperature_2m_max_MEAN': 'Tmax',
            'temperature_2m_min_MEAN': 'Tmin',
            'total_precipitation_sum_MEAN': 'Prec',
#             'InPoly_FID': 'ID'  #自行更換FID名稱
            'TARGET_FID': 'ID' #自行更換FID名稱
        }
        self.df.rename(columns=column_mapping, inplace=True)
    
    def _convert_temp_KtoC(self):
        """K氏溫標轉換成攝氏溫標"""
        if self.df['Temp'].max() > 100:
            self.df['Temp'] = self.df['Temp'] -273.15
            print('已將Temp欄位轉換為攝氏溫標')
        else:
            print('Temp欄位已為攝氏溫標')
            
        if self.df['Tmax'].max() > 100:
            self.df['Tmax'] = self.df['Tmax'] -273.15
            print('已將Tmax欄位轉換為攝氏溫標')
        else:
            print('Tmax欄位已為攝氏溫標')
            
        if self.df['Tmin'].max() > 100:
            self.df['Tmin'] = self.df['Tmin'] -273.15
            print('已將Tmin欄位轉換為攝氏溫標')
        else:
            print('Tmin欄位已為攝氏溫標')

    def _add_tra_column(self):
        """新增溫差欄位"""
        self.df.insert(6, 'Tra', self.df['Tmax'] - self.df['Tmin'])
        
    def _group_monthly(self):
        """計算月均值"""
        self.df = self.df.groupby(['ID', 'Year', 'Month']).agg({
            'Temp': 'mean',
            'Tmax': 'max',
            'Tmin': 'min',
            'Prec': 'sum',
            'Tra': 'mean'
        }).reset_index()


class BioClimateCalculator:
    def __init__(self, data):
        """初始化時執行基礎檢查"""
        required_cols = ['ID', 'Year', 'Month', 'Temp', 'Tmax', 'Tmin', 'Prec', 'Tra']
        if not all(col in data.columns for col in required_cols):
            missing = set(required_cols) - set(data.columns)
            raise ValueError(f"缺失必要欄位: {missing}")
        self.data = data.copy()
        self.results = {}
        
    def calculate(self, indicators):
        """
        根據給定的指標列表來計算 bioclimatic variables。
    
        參數:
            indicators (list): 要計算的指標名稱清單，例如 ['bio1', 'bio12', 'bio5']
    
        回傳:
            DataFrame: 合併後的結果
        """
        # 逐一執行所指定的指標方法
        for ind in indicators:
            if hasattr(self, ind):
                getattr(self, ind)()
            else:
                print(f"警告：找不到指標方法 {ind}，已跳過。")

    # 合併結果
        bio_merge = pd.DataFrame()
        for i in self.results:
            if len(bio_merge) == 0:
                bio_merge = self.results[i]
            else:
                bio_merge = pd.merge(bio_merge, self.results[i], on=['ID', 'Year'])

        return bio_merge
        
        
    # --------------------- 批量計算 ---------------------
    def calculate_all(self):
        """一次計算所有指標"""
        self.bio1()
        self.bio2()
        self.bio4()
        self.bio5()
        self.bio6()
        self.bio12()
        self.bio13()
        self.bio14()
            
        self.bio7()
        self.bio3()
        self.bio15()
        
        self.bio8()
        self.bio9()
        self.bio10()
        self.bio16()
        self.bio17()
        self.bio18()
        self.bio19()
        self.bio11()
        
        bio_merge = pd.DataFrame()
        for i in self.results:

            if len(bio_merge) == 0:
                bio_merge = self.results[i]
            else:
                bio_merge = pd.merge(bio_merge,self.results[i],on=['ID', 'Year'])
                
        return bio_merge
    # --------------------- 基礎指標 ---------------------
    def _basic_metric(self, col, agg_func, bio_name):
        """通用基礎指標計算"""
        result = (
            self.data.groupby(['ID', 'Year'])[[col]]
            .agg(agg_func)
            .reset_index()
            .rename(columns={col: bio_name})
        )
        if str(bio_name)[0:3] in ["Bio", "bio"]:
            self.results[bio_name] = result
            return result
        else:
            return result
    
    def bio1(self):
        """年均溫"""
        return self._basic_metric('Temp', 'mean', 'Bio1')
    
    def bio2(self):
        """月溫差(Tra)均值"""
        return self._basic_metric('Tra', 'mean', 'Bio2')
    
    def bio4(self):
        """溫度季節性(STD)"""
        return self._basic_metric('Temp', 'std', 'Bio4')
    
    def bio5(self):
        """最暖月份最高溫度"""
        return self._basic_metric('Tmax', 'max', 'Bio5')
    
    def bio6(self):
        """最冷月份最低溫度"""
        return self._basic_metric('Tmin', 'min', 'Bio6')
    
    def bio12(self):
        """年降雨量"""
        return self._basic_metric('Prec', 'sum', 'Bio12')
    
    def bio13(self):
        """最濕潤月份之降雨量"""
        return self._basic_metric('Prec', 'max', 'Bio13')
    
    def bio14(self):
        """最乾燥月份之降雨量"""
        return self._basic_metric('Prec', 'min', 'Bio14')
    
    # --------------------- 進階指標 ---------------------
    def bio7(self):
        """年溫差（Bio5 - Bio6）"""
        if 'Bio5' not in self.results or 'Bio6' not in self.results:
            raise ValueError("需要先計算 Bio5 和 Bio6")
            
        bio5 = self.results['Bio5']
        bio6 = self.results['Bio6']
        bio7 = bio5.copy()
        bio7['Bio7'] = bio5['Bio5'] - bio6['Bio6']
        
        bio7 = bio7.drop(columns = "Bio5")
        self.results['Bio7'] = bio7
        return bio7

    def bio3(self):
        """溫度季節性（Bio2/Bio7*100）"""
        if 'Bio2' not in self.results or 'Bio7' not in self.results:
            raise ValueError("需要先計算 Bio2 和 Bio7")
            
        bio2 = self.results['Bio2']
        bio7 = self.results['Bio7']
        bio3 = bio2.copy()
        bio3['Bio3'] = (bio2['Bio2'] / bio7['Bio7']) * 100
        bio3 = bio3.drop(columns = "Bio2")
        self.results['Bio3'] = bio3
        return bio3
    
    def bio15(self):
        """降雨季節性"""
        bio12 = self.results['Bio12']
        Prec_STD = self._basic_metric('Prec', 'std', 'Prec_STD')
        
        bio15 = bio12.copy()
        bio15['Bio15'] = (Prec_STD['Prec_STD']/(1 + bio12['Bio12']/12))*100
        bio15 = bio15.drop(columns = "Bio12")
        self.results['Bio15'] = bio15
        return bio15
        

    # --------------------- 季度指標 ---------------------
    def _quarterly_calculation(self, target_col, metric_col, 
                              agg_type='max', result_name='Bio8',
                              value_type='temp'):
        """季度指標通用計算模板"""
        df_sorted = self.data.sort_values(['ID', 'Year', 'Month']).copy()
        
        # 計算季度移動平均
        roll_col = f'{metric_col}_3m'
        df_sorted[roll_col] = (
            df_sorted.groupby('ID')[metric_col]
            .rolling(3, min_periods=3)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        # 年份分组
        df_sorted['Year_Group'] = df_sorted.apply(
            lambda m: m['Year'] - 1 if m['Month'] < 3 else m['Year'], 
            axis=1)
        
        # 尋找最大/最小季度
        if agg_type == 'max':
            idx = df_sorted.groupby(['ID', 'Year_Group'])[roll_col].idxmax()
        elif agg_type == 'min':
            idx = df_sorted.groupby(['ID', 'Year_Group'])[roll_col].idxmin()
        else:
            raise ValueError("agg_type 必須是 'max' 或 'min'")
            
        quarters = df_sorted.loc[idx.dropna()]

        # 提取目標數據
        values = []
        for _, row in quarters.iterrows():
            start = row.name - 2
            vals = df_sorted.loc[start:row.name, target_col]
            
            if vals.isna().any():
                values.append(np.nan)
            else:
                if value_type == 'temp':
                    values.append(vals.mean())
                elif value_type == 'prec':
                    values.append(vals.sum())
        
        # 輸出結果
        result = quarters[['ID', 'Year_Group']].rename(columns={'Year_Group': 'Year'})
        result[result_name] = values
        self.results[result_name] = result
        return result

    def bio8(self):
        """最濕潤季節之平均溫度"""
        return self._quarterly_calculation(
            target_col='Temp',
            metric_col='Prec',
            agg_type='max',
            result_name='Bio8',
            value_type='temp'
        )
            
            
    def bio9(self):
        """最乾燥季節之平均溫度"""
        return self._quarterly_calculation(
            target_col='Temp',
            metric_col='Prec',
            agg_type='min',
            result_name='Bio9',
            value_type='temp'
        )            
            
    def bio10(self):
        """最溫暖季節之平均溫度"""
        return self._quarterly_calculation(
            target_col='Temp',
            metric_col='Temp',
            agg_type='max',
            result_name='Bio10',
            value_type='temp'
        )
            
    def bio11(self):
        """最寒冷季節之平均溫度"""
        return self._quarterly_calculation(
            target_col='Temp',
            metric_col='Temp',
            agg_type='min',
            result_name='Bio11',
            value_type='temp'
        )
            
    def bio16(self):
        """最濕潤季節之降雨量"""
        return self._quarterly_calculation(
            target_col='Prec',
            metric_col='Prec',
            agg_type='max',
            result_name='Bio16',
            value_type='prec'
        )         
            
    def bio17(self):
        """最乾燥季節之降雨量"""
        return self._quarterly_calculation(
            target_col='Prec',
            metric_col='Prec',
            agg_type='min',
            result_name='Bio17',
            value_type='prec'
        )

    def bio18(self):
        """最溫暖季節之降雨量"""
        return self._quarterly_calculation(
            target_col='Prec',
            metric_col='Temp',
            agg_type='max',
            result_name='Bio18',
            value_type='prec'
        )

    def bio19(self):
        """最寒冷季節之降雨量"""
        return self._quarterly_calculation(
            target_col='Prec',
            metric_col='Temp',
            agg_type='min',
            result_name='Bio19',
            value_type='prec'
        )


# # 使用範例
# if __name__ == "__main__":
#     preprocessor = DataPreprocessor("input_data.csv")
    
#     # 假設已知最大ID為10204
#     preprocessor.process_data(total_ids=10204, batch_size=5)
    
#     # 合併並導出結果
#     preprocessor.merge_and_export("final_results.csv")