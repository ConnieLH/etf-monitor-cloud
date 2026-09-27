# ETF 三指数每日监控（云端）

每个工作日北京时间 17:07、17:27、18:07、19:47（收盘后与兜底重试）自动生成中证A500、科创50、红利低波100监控报告，发布到 GitHub Pages。GitHub 定时任务可能排队延迟，不能保证在某一分钟准时启动。

- 数据源：中证指数官网（价格/PE/股息率）+ 腾讯行情（沪深成交额代理）+ AKShare 中美国债收益率（10年国债）
- 日期保护：按北京时间判断完整收盘日；价格与成交额未齐时停止部署旧报告，PE 如滞后则单独标注其数据日期
- 报告结构：顶部“今天该做什么”（每只指数一句话：买入/卖出/持有/等待），然后是“分批进场进度”（第1步底背离半仓、第2步周线Fr站上0轴满仓，金额按股票资金100万举例），最后每只指数三张关键图：K线与关键价位、Fr趋势动量、估值。默认周K，可切换日K/月K，上下图逐根对齐；周K标出历史买卖点，日K/月K仅供参考。手机默认显示最新行情，上下图同步横向滚动。每张图下面有“怎么看这张图”和“现在说明什么”
- 周线Fr买卖信号：周线Fr跌破0轴即卖，重新上穿0轴买回；下跌周期里出现底背离拐点（价新低、Fr未新低、BAR转红）买半仓，价格和Fr同时再创新低则止损。只以完整的一周确认。规则在 `signal_rules.py`，与回测引擎逐条一致
- 历史数据：每个交易日的快照追加到 `history/daily_snapshots.csv`，红利低波100的股息率与10年国债收益率追加到 `history/dividend_yield_history.csv`（官方接口只给近期数据，靠每日积累形成历史）。这些提交也让仓库保持活跃，避免定时任务因 60 天无活动被 GitHub 停用
- 同一交易日重复运行时，如果没有新数据就跳过部署
- 诊断文件：每次运行都会生成并上传 `source_freshness.json`，列出各数据源最新日期
- 报告入口：https://connielh.github.io/etf-monitor-cloud/
- 手动触发：Actions → daily-report → Run workflow（手动触发总会重新部署）

本地运行：

```bash
pip install -r requirements.txt
python three_index_report.py --output-dir output --history-dir history
python -m unittest discover -s tests -v
```

本报告是按预设规则计算的观察与参考，不是投资建议。
