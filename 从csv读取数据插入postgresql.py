import psycopg2
import csv


with open(r'D:/aaa.csv', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)
  # your_list = your_list[1:]

# print(your_list)

list_to = [tuple(item) for item in your_list]
insert_list = []
for row in list_to:
    new_row = [None if item.strip() == '' else item for item in row]
    insert_list.append(new_row)
print(insert_list)
sql = ' insert into table_a ' \
      ' values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) '
conn = psycopg2.connect("host=192.168.1.x dbname=xxx user=xxx password=xxx")
cur = conn.cursor()
cur.executemany(sql, insert_list)
conn.commit()
