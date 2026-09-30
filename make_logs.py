from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import pandas as pd
import requests

def process_item(item):
	file_url = item.get('file_path')
	date_str = item.get('file_fromdate')
	
	if not file_url or not date_str:
		return []

	try:
		response = requests.get(file_url, timeout=15)
		if response.status_code == 200:
			xls = pd.ExcelFile(io.BytesIO(response.content))
			df = pd.read_excel(xls, sheet_name='Load Demand Monthly Figures')
			
			hours_data = df.iloc[1, 1:25].dropna().astype(float).values
			base_date = pd.to_datetime(date_str, format='%d.%m.%Y')
			
			local_records = []
			for hour_idx, load_val in enumerate(hours_data):
				current_time = base_date + pd.Timedelta(hours=hour_idx)
				local_records.append({
					'load': load_val,
					'hour': current_time.hour,
					'dayofweek': current_time.dayofweek,
					'month': current_time.month
				})
			return local_records
	except Exception:
		pass
	return []

def main():
	with open('admie_links.json', 'r', encoding='utf-8') as f:
		data = json.load(f)

	records = []
	
	with ThreadPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(process_item, item) for item in data]
		for future in as_completed(futures):
			res = future.result()
			if res:
				records.extend(res)

	with open('admie_dataset.txt', 'w', encoding='utf-8') as txt_file:
		for r in records:
			txt_file.write(f"{r['load']},{r['hour']},{r['dayofweek']},{r['month']}\n")

	print(f"Done. Structured dataset with {len(records)} rows created.")

if __name__ == '__main__':
	main()