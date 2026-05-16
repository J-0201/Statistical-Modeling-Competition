# 全国大学生统计建模大赛参赛项目

## 项目简介
本项目使用 Python 对广东省制造业面板数据进行分析，建立 GM(1,1) 灰色预测模型、OLS 回归模型及贝叶斯回归模拟，研究电信业务量、互联网普及率对制造业总产出的影响，并预测未来趋势。

## 文件说明
- `main.py` — 主程序代码（数据预处理、建模、回归、可视化）
- `final_dataset_6years_real.csv` — 原始数据
- `GM11_mfg_output.png` — 制造业总产出 GM(1,1) 拟合与预测图
- `GM11_fit_forecast.png` — 研发强度 GM(1,1) 拟合与预测图
- `Bayesian_posterior.png` — 贝叶斯回归系数后验分布图
- `OLS_robustness.png` — OLS 回归残差诊断图

## 运行环境
- Python 3.8+
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib

## 运行方法
```bash
python main.py
