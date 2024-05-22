import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from torch import tensor
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import ollama
import psycopg2
import pickle

def get_data_table():
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()
    cur.execute("SELECT name, functions FROM brusnika")
    rows = cur.fetchall()
    data_table = {}
    for row in rows:
        data_table[row[0]] = row[1]
    conn.close()
    return data_table

def get_employee_details(employee_name_with_id):
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()
    cur.execute("SELECT mail, phone_number, place, post, date_of_birth FROM brusnika WHERE name = %s", (employee_name_with_id,))
    details = cur.fetchone()
    conn.close()
    return details


def get_embeddings_from_sql():
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM brusnika")
    rows = cur.fetchall()
    embeddings_dict = {}
    for row in rows:
        embeddings_dict[row[0]] = pickle.loads(row[1])
    conn.close()
    return embeddings_dict

def func1(s, data_table):
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    embeddings_from_sql = get_embeddings_from_sql()
    embeddings2 = [embeddings_from_sql[name] for name in data_table.keys()]
    dat = model.encode(s)
    arr = []
    xdd = list(data_table.keys())
    for ind, value in enumerate(embeddings2):
        arr.append([data_table[xdd[ind]], util.pytorch_cos_sim(tensor(embeddings2[ind]), tensor(dat)), xdd[ind]])
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Отправь мне запрос, и я выведу наиболее релевантных сотрудников.')

def echo(update: Update, context: CallbackContext) -> None:
    s = update.message.text
    print(s)
    data_table = get_data_table()
    print(data_table)
    persons = func1(s, data_table)
    diff = tensor([[0.1044]])
    data = [{"functions": i[0][:-1], "name": i[2]} for i in persons if persons[0][1] - diff <= i[1]]

    data1 = data[:10]
    #print(prompt)
    stream = ollama.generate(
        model='llama3',
        prompt=f"Using this data: {data}. Select most suitable employees(name and id)(sort by relevance) and return answer in json format for request: {s}",
        options=
        {
            'temperature': 0.8,
        }
    )
    answer = str(stream["response"])
    print(answer, type(answer))

    employees = json.loads(answer.split('```')[1].replace('json', ''))

    res = []
    employee_names_with_ids = [f'{employee["name"]}id={employee["id"]}' for employee in employees]

    for employee_name_with_id in employee_names_with_ids:
        details = get_employee_details(employee_name_with_id)
        res.append((data_table[employee_name_with_id], employee_name_with_id, details))
    counter = 0
    for i in res:
        details_str = f'✉️ {i[2][0]}\nТел: {i[2][1]}\nКомпания: ООО "Брусника"\nМесто работы: {i[2][2]}\nДолжность: {i[2][3]}\nФункции: {i[0][:-1]}'
        formatted = f'{"".join(i[1]).split("id=")[0]}\n{details_str}\nДата рождения: {i[2][4]}'
        update.message.reply_text(formatted)
        counter += 1
        if counter == 3:
            continue

def main() -> None:
    updater = Updater("7026382273:AAE1emD7ubHMqgRWJy6uJ-4qMEoc_xYomC8")
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
